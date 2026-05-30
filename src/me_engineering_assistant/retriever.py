from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Sequence

from me_engineering_assistant.config import env
from me_engineering_assistant.documents import DocumentChunk


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_MODEL_CACHE: dict[str, Any] = {}
_INDEX_CACHE: dict[tuple[str, tuple[tuple[str, str], ...]], dict[str, Any]] = {}


@dataclass(frozen=True)
class RetrievalResult:
    content: str
    metadata: dict[str, str]
    score: float

    def to_dict(self) -> dict[str, object]:
        return {"content": self.content, "metadata": self.metadata, "score": self.score}


class InMemoryECURetriever:
    """Hybrid dense + BM25 retriever for the small ECU corpus."""

    backend_name = "hybrid-sentence-transformers-faiss-bm25"

    def __init__(self, chunks: Sequence[DocumentChunk], model_name: str | None = None) -> None:
        self.chunks = list(chunks)
        if not self.chunks:
            raise ValueError("InMemoryECURetriever requires at least one document chunk.")

        try:
            import faiss
            import numpy as np
            from rank_bm25 import BM25Okapi
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "The retriever requires sentence-transformers, faiss-cpu, and rank-bm25. "
                "Install them with: python -m pip install -e '.[dev]'"
            ) from exc

        self.np = np
        self.model_name = model_name or env("LOCAL_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        cache_key = _index_cache_key(self.model_name, self.chunks)
        cached = _INDEX_CACHE.get(cache_key)
        if cached is None:
            self.model = _load_sentence_transformer(self.model_name, SentenceTransformer)
            embeddings = self.model.encode(
                [chunk.content for chunk in self.chunks],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            matrix = np.asarray(embeddings, dtype="float32")
            index = faiss.IndexFlatIP(matrix.shape[1])
            index.add(matrix)
            cached = {
                "model": self.model,
                "index": index,
                "tokenized_chunks": [_tokenize(chunk.content) for chunk in self.chunks],
            }
            cached["bm25"] = BM25Okapi(cached["tokenized_chunks"])
            _INDEX_CACHE[cache_key] = cached

        self.model = cached["model"]
        self.index = cached["index"]
        self.tokenized_chunks = cached["tokenized_chunks"]
        self.bm25 = cached["bm25"]

    def retrieve(
        self,
        query: str,
        filters: dict[str, Sequence[str]] | None = None,
        top_k: int = 4,
    ) -> list[RetrievalResult]:
        candidates = [chunk for chunk in self.chunks if _matches_filters(chunk, filters)]
        if not candidates:
            candidates = self.chunks

        query_tokens = _tokenize(query)
        query_embedding = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        query_matrix = self.np.asarray(query_embedding, dtype="float32")
        limit = min(len(self.chunks), max(top_k * 8, top_k))
        scores, indices = self.index.search(query_matrix, limit)
        bm25_scores = self.bm25.get_scores(query_tokens)
        max_bm25 = max(float(score) for score in bm25_scores) if len(bm25_scores) else 0.0
        candidate_ids = {id(chunk) for chunk in candidates}

        dense_scores: dict[int, float] = {}
        for score, index in zip(scores[0], indices[0], strict=False):
            if index < 0:
                continue
            dense_scores[int(index)] = float(score)

        ranked = []
        for index, chunk in enumerate(self.chunks):
            if id(chunk) not in candidate_ids:
                continue
            dense_score = dense_scores.get(index, 0.0)
            bm25_score = float(bm25_scores[index]) if len(bm25_scores) else 0.0
            bm25_normalized = bm25_score / max_bm25 if max_bm25 > 0 else 0.0
            metadata_score = _metadata_score(query_tokens=query_tokens, chunk=chunk, filters=filters)
            combined = (0.55 * dense_score) + (0.35 * bm25_normalized) + metadata_score
            ranked.append((combined, dense_score, bm25_score, metadata_score, chunk))

        ranked.sort(key=lambda item: item[0], reverse=True)
        results: list[RetrievalResult] = []
        for combined, dense_score, bm25_score, metadata_score, chunk in ranked[:top_k]:
            metadata = dict(chunk.metadata)
            metadata["retrieval_method"] = "hybrid"
            metadata["dense_score"] = f"{dense_score:.6f}"
            metadata["bm25_score"] = f"{bm25_score:.6f}"
            metadata["metadata_score"] = f"{metadata_score:.6f}"
            results.append(RetrievalResult(content=chunk.content, metadata=metadata, score=float(combined)))
        return results


def _matches_filters(chunk: DocumentChunk, filters: dict[str, Sequence[str]] | None) -> bool:
    if not filters:
        return True

    models = set(_filter_values(filters.get("models")))
    series = set(_filter_values(filters.get("series")))
    sources = set(_filter_values(filters.get("sources")))
    fields = set(_filter_values(filters.get("fields"))) | set(_filter_values(filters.get("field")))

    metadata = chunk.metadata
    if models and metadata.get("model") not in models:
        return False
    if series and metadata.get("series") not in series:
        return False
    if sources and metadata.get("source") not in sources:
        return False
    if fields and not any(_field_matches(field, metadata.get("field", "")) for field in fields):
        return False
    return True


def _load_sentence_transformer(model_name: str, model_class):
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = model_class(model_name)
    return _MODEL_CACHE[model_name]


def _index_cache_key(model_name: str, chunks: Sequence[DocumentChunk]) -> tuple[str, tuple[tuple[str, str], ...]]:
    return model_name, tuple((chunk.chunk_id, chunk.content) for chunk in chunks)


def _filter_values(value: Sequence[str] | str | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _tokenize(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", value.lower())


def _metadata_score(
    *,
    query_tokens: Sequence[str],
    chunk: DocumentChunk,
    filters: dict[str, Sequence[str]] | None,
) -> float:
    metadata = chunk.metadata
    score = 0.0
    if metadata.get("chunk_type") == "field":
        score += 0.08
    field = metadata.get("field", "")
    if field:
        field_tokens = set(field.split("_"))
        if field_tokens & set(query_tokens):
            score += 0.14
    if filters:
        requested_fields = set(filters.get("fields", ())) | set(filters.get("field", ()))
        if requested_fields and any(_field_matches(requested, field) for requested in requested_fields):
            score += 0.2
        if filters.get("models") and metadata.get("model") in set(filters.get("models", ())):
            score += 0.05
    return score


def _field_matches(requested: str, actual: str) -> bool:
    if not requested:
        return True
    requested_tokens = set(re.findall(r"[a-z0-9]+", requested.lower()))
    actual_tokens = set(re.findall(r"[a-z0-9]+", actual.lower()))
    if not requested_tokens or not actual_tokens:
        return False
    return requested_tokens <= actual_tokens or actual_tokens <= requested_tokens or bool(requested_tokens & actual_tokens)

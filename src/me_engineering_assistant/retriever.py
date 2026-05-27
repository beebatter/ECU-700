from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from me_engineering_assistant.config import env
from me_engineering_assistant.documents import DocumentChunk


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_MODEL_CACHE: dict[str, Any] = {}


@dataclass(frozen=True)
class RetrievalResult:
    content: str
    metadata: dict[str, str]
    score: float

    def to_dict(self) -> dict[str, object]:
        return {"content": self.content, "metadata": self.metadata, "score": self.score}


class InMemoryECURetriever:
    """Sentence-transformers + FAISS retriever for the small ECU corpus."""

    backend_name = "sentence-transformers-faiss"

    def __init__(self, chunks: Sequence[DocumentChunk], model_name: str | None = None) -> None:
        self.chunks = list(chunks)
        if not self.chunks:
            raise ValueError("InMemoryECURetriever requires at least one document chunk.")

        try:
            import faiss
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "The retriever requires sentence-transformers and faiss-cpu. "
                "Install them with: python -m pip install -e '.[dev]'"
            ) from exc

        self.np = np
        self.model_name = model_name or env("LOCAL_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self.model = _load_sentence_transformer(self.model_name, SentenceTransformer)
        embeddings = self.model.encode(
            [chunk.content for chunk in self.chunks],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        matrix = np.asarray(embeddings, dtype="float32")
        self.index = faiss.IndexFlatIP(matrix.shape[1])
        self.index.add(matrix)

    def retrieve(
        self,
        query: str,
        filters: dict[str, Sequence[str]] | None = None,
        top_k: int = 4,
    ) -> list[RetrievalResult]:
        candidates = [chunk for chunk in self.chunks if _matches_filters(chunk, filters)]
        if not candidates:
            candidates = self.chunks

        query_embedding = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        query_matrix = self.np.asarray(query_embedding, dtype="float32")
        limit = min(len(self.chunks), max(top_k * 4, top_k))
        scores, indices = self.index.search(query_matrix, limit)
        candidate_ids = {id(chunk) for chunk in candidates}
        results: list[RetrievalResult] = []
        for score, index in zip(scores[0], indices[0], strict=False):
            if index < 0:
                continue
            chunk = self.chunks[int(index)]
            if id(chunk) not in candidate_ids:
                continue
            results.append(RetrievalResult(content=chunk.content, metadata=chunk.metadata, score=float(score)))
            if len(results) >= top_k:
                break
        return results


def _matches_filters(chunk: DocumentChunk, filters: dict[str, Sequence[str]] | None) -> bool:
    if not filters:
        return True

    models = set(filters.get("models", ()))
    series = set(filters.get("series", ()))
    sources = set(filters.get("sources", ()))

    metadata = chunk.metadata
    if models and metadata.get("model") not in models:
        return False
    if series and metadata.get("series") not in series:
        return False
    if sources and metadata.get("source") not in sources:
        return False
    return True


def _load_sentence_transformer(model_name: str, model_class):
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = model_class(model_name)
    return _MODEL_CACHE[model_name]

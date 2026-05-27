from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence

from me_engineering_assistant.config import env
from me_engineering_assistant.documents import DocumentChunk


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9.+-]+")

QUERY_EXPANSIONS = {
    "ai": ("npu", "accelerator", "tops", "edge"),
    "can": ("interface", "bus", "channel", "fd", "mbps"),
    "memory": ("ram", "lpddr4"),
    "npu": ("ai", "accelerator", "tops"),
    "ota": ("over-the-air", "update"),
    "power": ("consumption", "idle", "load"),
    "ram": ("memory", "lpddr4"),
    "storage": ("flash", "emmc", "capacity"),
    "temperature": ("temp", "operating", "environment"),
    "updates": ("ota", "over-the-air"),
}


@dataclass(frozen=True)
class RetrievalResult:
    content: str
    metadata: dict[str, str]
    score: float

    def to_dict(self) -> dict[str, object]:
        return {"content": self.content, "metadata": self.metadata, "score": self.score}


class OptionalRetrieverBackend(Protocol):
    name: str

    def retrieve(
        self,
        query: str,
        candidates: Sequence[DocumentChunk],
        top_k: int,
    ) -> list[RetrievalResult]:
        ...


class InMemoryECURetriever:
    """Small retriever with optional local/Databricks embedding backends."""

    def __init__(
        self,
        chunks: Sequence[DocumentChunk],
        prefer_langchain: bool = True,
        embedding_endpoint: str | None = None,
    ) -> None:
        self.chunks = list(chunks)
        self._chunk_tokens = [Counter(tokenize(chunk.content)) for chunk in self.chunks]
        self._idf = self._compute_idf(self._chunk_tokens)
        self.embedding_backend: OptionalRetrieverBackend | None = None
        self.backend_name = "keyword-tfidf"
        if prefer_langchain:
            self.embedding_backend = build_optional_local_faiss(self.chunks)
            if self.embedding_backend is None:
                self.embedding_backend = build_optional_langchain_faiss(
                    self.chunks,
                    embedding_endpoint=embedding_endpoint,
                )
            if self.embedding_backend is not None:
                self.backend_name = self.embedding_backend.name

    def retrieve(
        self,
        query: str,
        filters: dict[str, Sequence[str]] | None = None,
        top_k: int = 4,
    ) -> list[RetrievalResult]:
        candidates = [chunk for chunk in self.chunks if _matches_filters(chunk, filters)]
        if not candidates:
            candidates = self.chunks
        if self.embedding_backend is not None:
            return self.embedding_backend.retrieve(query, candidates, top_k)

        query_terms = expand_query(tokenize(query))
        scored: list[RetrievalResult] = []
        for chunk in candidates:
            index = self.chunks.index(chunk)
            score = self._score(query, query_terms, chunk, self._chunk_tokens[index])
            if score > 0:
                scored.append(RetrievalResult(content=chunk.content, metadata=chunk.metadata, score=score))

        if not scored:
            scored = [RetrievalResult(content=chunk.content, metadata=chunk.metadata, score=0.01) for chunk in candidates]

        return sorted(scored, key=lambda result: result.score, reverse=True)[:top_k]

    @staticmethod
    def _compute_idf(counters: Iterable[Counter[str]]) -> dict[str, float]:
        counters = list(counters)
        document_count = max(len(counters), 1)
        document_frequency: Counter[str] = Counter()
        for counter in counters:
            document_frequency.update(counter.keys())
        return {
            token: math.log((1 + document_count) / (1 + frequency)) + 1
            for token, frequency in document_frequency.items()
        }

    def _score(self, query: str, query_terms: Sequence[str], chunk: DocumentChunk, counter: Counter[str]) -> float:
        score = 0.0
        for term in query_terms:
            score += counter.get(term, 0) * self._idf.get(term, 1.0)

        lowered_content = chunk.content.lower()
        lowered_query = query.lower()
        if chunk.metadata.get("model", "").lower() in lowered_query:
            score += 8.0
        if chunk.metadata.get("series", "").lower() in lowered_query:
            score += 3.0
        for phrase in ("can fd", "over-the-air", "operating temp", "power consumption"):
            if phrase in lowered_query and phrase in lowered_content:
                score += 3.0
        return score


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text.replace("_", "-"))]


def expand_query(tokens: Sequence[str]) -> list[str]:
    expanded: list[str] = []
    for token in tokens:
        expanded.append(token)
        expanded.extend(QUERY_EXPANSIONS.get(token, ()))
    return expanded


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


def build_optional_langchain_faiss(
    chunks: Sequence[DocumentChunk],
    embedding_endpoint: str | None = None,
):
    endpoint = embedding_endpoint or os.getenv("DATABRICKS_EMBEDDING_ENDPOINT")
    if not endpoint:
        return None

    try:
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document
        from langchain_databricks import DatabricksEmbeddings
    except ImportError:
        return None

    documents = [Document(page_content=chunk.content, metadata=chunk.metadata) for chunk in chunks]
    embeddings = DatabricksEmbeddings(endpoint=endpoint)
    vectorstore = FAISS.from_documents(documents, embeddings)
    return LangChainFAISSBackend(vectorstore, chunks)


def build_optional_local_faiss(chunks: Sequence[DocumentChunk]):
    backend = (env("ME_RETRIEVER_BACKEND", "auto") or "auto").lower()
    if backend not in {"auto", "local-faiss", "faiss", "sentence-transformers"}:
        return None

    try:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None

    model_name = env("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    model = SentenceTransformer(model_name)
    texts = [chunk.content for chunk in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    matrix = np.asarray(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    return LocalFAISSBackend(index=index, model=model, chunks=list(chunks), np=np)


class LocalFAISSBackend:
    name = "local-faiss-sentence-transformers"

    def __init__(self, index, model, chunks: list[DocumentChunk], np) -> None:
        self.index = index
        self.model = model
        self.chunks = chunks
        self.np = np

    def retrieve(
        self,
        query: str,
        candidates: Sequence[DocumentChunk],
        top_k: int,
    ) -> list[RetrievalResult]:
        query_embedding = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        query_matrix = self.np.asarray(query_embedding, dtype="float32")
        candidate_ids = {id(chunk) for chunk in candidates}
        limit = min(len(self.chunks), max(top_k * 4, top_k))
        scores, indices = self.index.search(query_matrix, limit)
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


class LangChainFAISSBackend:
    name = "databricks-langchain-faiss"

    def __init__(self, vectorstore, chunks: Sequence[DocumentChunk]) -> None:
        self.vectorstore = vectorstore
        self.chunks = list(chunks)

    def retrieve(
        self,
        query: str,
        candidates: Sequence[DocumentChunk],
        top_k: int,
    ) -> list[RetrievalResult]:
        candidate_sources = {chunk.metadata["source"] for chunk in candidates}
        documents_with_scores = self.vectorstore.similarity_search_with_score(query, k=max(top_k * 3, top_k))
        results: list[RetrievalResult] = []
        for document, score in documents_with_scores:
            metadata = dict(document.metadata)
            if metadata.get("source") not in candidate_sources:
                continue
            results.append(
                RetrievalResult(
                    content=document.page_content,
                    metadata=metadata,
                    score=float(score),
                )
            )
            if len(results) >= top_k:
                break
        return results

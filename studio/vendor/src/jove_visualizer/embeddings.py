from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


class Embedder(ABC):
    name: str
    dimension: int

    @abstractmethod
    def encode_documents(self, texts: Sequence[str], batch_size: int = 128) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def encode_queries(self, texts: Sequence[str], batch_size: int = 128) -> np.ndarray:
        raise NotImplementedError


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Run setup_windows.ps1 first."
            ) from exc
        self.name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        self.dimension = int(self._model.get_sentence_embedding_dimension())
        self._is_bge = "bge" in model_name.lower()

    def encode_documents(self, texts: Sequence[str], batch_size: int = 128) -> np.ndarray:
        matrix = self._model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(matrix, dtype=np.float32)

    def encode_queries(self, texts: Sequence[str], batch_size: int = 128) -> np.ndarray:
        prepared = list(texts)
        if self._is_bge:
            prepared = [f"Represent this sentence for searching relevant passages: {text}" for text in prepared]
        matrix = self._model.encode(
            prepared,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(matrix, dtype=np.float32)


class HashingEmbedder(Embedder):
    """Deterministic smoke-test backend. Do not use it for scientific acceptance testing."""

    def __init__(self, dimension: int = 1024) -> None:
        from sklearn.feature_extraction.text import HashingVectorizer

        self.name = f"hashing-{dimension}"
        self.dimension = dimension
        self._vectorizer = HashingVectorizer(
            n_features=dimension,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2),
            lowercase=True,
        )

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        sparse = self._vectorizer.transform(list(texts))
        return np.asarray(sparse.toarray(), dtype=np.float32)

    def encode_documents(self, texts: Sequence[str], batch_size: int = 128) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: Sequence[str], batch_size: int = 128) -> np.ndarray:
        return self._encode(texts)


def create_embedder(backend: str, model_name: str, device: str = "cpu") -> Embedder:
    normalized = backend.strip().lower()
    if normalized == "sentence_transformers":
        return SentenceTransformerEmbedder(model_name, device=device)
    if normalized == "hashing":
        dimension = 1024
        if model_name.startswith("hashing-"):
            try:
                dimension = int(model_name.rsplit("-", 1)[1])
            except ValueError:
                pass
        return HashingEmbedder(dimension=dimension)
    raise ValueError(f"Unsupported embedding backend: {backend}")


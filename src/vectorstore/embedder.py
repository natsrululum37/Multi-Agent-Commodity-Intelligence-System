"""Module untuk embedding teks menggunakan sentence-transformers."""

import os
from pathlib import Path

# Disable telemetry untuk sentence-transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = str(Path.home() / ".cache" / "huggingface")

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List


class Embedder:
    """Class untuk embedding teks menggunakan model lokal."""

    _instance = None
    _model = None

    def __new__(cls, *args, **kwargs):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Inisialisasi embedder.

        Args:
            model_name: Nama model sentence-transformers.
        """
        if self._model is None:
            try:
                self._model = SentenceTransformer(model_name)
                self.model_name = model_name
                self.dimension = self._model.get_embedding_dimension()
            except Exception as e:
                print(f"Warning: Tidak bisa memuat model {model_name}: {e}")
                self._model = None
                self.model_name = model_name
                self.dimension = 384  # Default untuk MiniLM

    def embed(self, texts: List[str]) -> np.ndarray:
        """Buat embedding untuk daftar teks.

        Args:
            texts: Daftar teks yang akan di-embed.

        Returns:
            Array numpy berisi embedding.
        """
        if not texts:
            return np.array([])
        if self._model is None:
            # Fallback: random embedding (tidak direkomendasikan untuk produksi)
            return np.random.rand(len(texts), self.dimension)

        return self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    def embed_query(self, query: str) -> np.ndarray:
        """Buat embedding untuk satu query.

        Args:
            query: Query teks.

        Returns:
            Array numpy 1D berisi embedding.
        """
        return self.embed([query])[0]

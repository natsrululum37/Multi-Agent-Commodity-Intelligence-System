"""Module untuk FAISS index management."""

import json
import pickle
from pathlib import Path
import numpy as np
import faiss
from typing import List, Tuple, Optional, Dict


class FAISSIndex:
    """Class untuk manage FAISS vector index."""

    def __init__(self, dimension: int = 384):
        """Inisialisasi FAISS index.

        Args:
            dimension: Dimensi embedding vector.
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents: List[str] = []
        self.metadata: List[dict] = []

    def add_documents(self, embeddings: np.ndarray, documents: List[str], metadata: Optional[List[dict]] = None) -> None:
        """Tambahkan dokumen ke index.

        Args:
            embeddings: Array numpy berisi embedding vektor.
            documents: List teks dokumen.
            metadata: List metadata untuk setiap dokumen.
        """
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Dimensi embedding ({embeddings.shape[1]}) tidak sesuai dengan index ({self.dimension})")

        embeddings = embeddings.astype(np.float32)
        self.index.add(embeddings)
        self.documents.extend(documents)

        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(documents))

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float, dict]]:
        """Cari dokumen paling relevan.

        Args:
            query_embedding: Embedding dari query.
            k: Jumlah dokumen yang diambil.

        Returns:
            List of (document, score, metadata) tuples.
        """
        if self.index.ntotal == 0:
            return []

        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        distances, indices = self.index.search(query_embedding, min(k, self.index.ntotal))

        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.documents):
                results.append((
                    self.documents[idx],
                    float(dist),
                    self.metadata[idx] if self.metadata else {}
                ))

        return results

    def hybrid_search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        metadata_filter: Optional[Dict] = None,
    ) -> List[Tuple[str, float, dict]]:
        """Hybrid search: semantic similarity + metadata filter.

        Args:
            query_embedding: Embedding dari query.
            k: Jumlah dokumen yang diambil.
            metadata_filter: Dict berisi filter metadata
                (contoh: {"type": "monthly_stats", "period": "2024-03"}).

        Returns:
            List of (document, score, metadata) tuples.
        """
        if self.index.ntotal == 0:
            return []

        # Ambil lebih banyak hasil dari FAISS (5x top_k) untuk filter
        initial_k = min(k * 5, self.index.ntotal)
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        distances, indices = self.index.search(query_embedding, initial_k)

        # Filter hasil berdasarkan metadata
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= len(self.documents):
                continue

            meta = self.metadata[idx] if self.metadata else {}

            # Cek apakah metadata match filter
            if metadata_filter:
                match = True
                for key, value in metadata_filter.items():
                    if key not in meta or meta[key] != value:
                        match = False
                        break
                if not match:
                    continue

            results.append((
                self.documents[idx],
                float(dist),
                meta,
            ))

            if len(results) >= k:
                break

        # Jika hasil filter kurang, ambil sisa dari semantic search
        if len(results) < k:
            for dist, idx in zip(distances[0], indices[0]):
                if idx >= len(self.documents):
                    continue

                doc_tuple = (
                    self.documents[idx],
                    float(dist),
                    self.metadata[idx] if self.metadata else {}
                )
                if doc_tuple not in results:
                    results.append(doc_tuple)
                    if len(results) >= k:
                        break

        return results

    def save(self, index_path: str | Path, documents_path: Optional[str | Path] = None) -> None:
        """Simpan index ke file.

        Args:
            index_path: Path untuk menyimpan FAISS index.
            documents_path: Path untuk menyimpan dokumen dan metadata.
        """
        index_path = Path(index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))

        if documents_path:
            with open(documents_path, "wb") as f:
                pickle.dump({"documents": self.documents, "metadata": self.metadata}, f)

    def load(self, index_path: str | Path, documents_path: Optional[str | Path] = None) -> "FAISSIndex":
        """Muat index dari file.

        Args:
            index_path: Path file FAISS index.
            documents_path: Path file dokumen dan metadata.

        Returns:
            Instance FAISSIndex yang sudah dimuat.
        """
        index_path = Path(index_path)
        self.index = faiss.read_index(str(index_path))
        self.dimension = self.index.d

        if documents_path and Path(documents_path).exists():
            with open(documents_path, "rb") as f:
                data = pickle.load(f)
                self.documents = data["documents"]
                self.metadata = data["metadata"]

        return self

    @property
    def size(self) -> int:
        """Jumlah dokumen di index."""
        return self.index.ntotal

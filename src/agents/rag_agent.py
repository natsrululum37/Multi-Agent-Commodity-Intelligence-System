"""RAGAgent - Untuk menjawab pertanyaan bisnis menggunakan RAG."""

import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import groq
from src.agents.base import BaseAgent
from src.config import Config
from src.vectorstore.index import FAISSIndex
from src.vectorstore.embedder import Embedder


class RAGAgent(BaseAgent):
    """Agent yang menggunakan Retrieval Augmented Generation untuk menjawab pertanyaan."""

    def __init__(self, api_key: Optional[str] = None, cache_dir: str = ".cache"):
        super().__init__(name="RAGAgent")
        self.api_key = api_key or Config.GROQ_API_KEY
        if not self.api_key or self.api_key == "gsk_your_api_key_here":
            print("Warning: GROQ_API_KEY tidak diatur. RAGAgent akan berjalan dalam mode offline (tanpa LLM).")
            self.client = None
        else:
            self.client = groq.Client(api_key=self.api_key)
        self.model = Config.GROQ_MODEL_RAG

        # Setup vector store
        self.embedder = Embedder(Config.EMBEDDING_MODEL)
        self.index = FAISSIndex(dimension=self.embedder.dimension)
        self.context_documents: List[Dict[str, Any]] = []

        # Caching setup
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.cache_dir / "faiss_index.bin"
        self.documents_path = self.cache_dir / "documents.pkl"

    def process(self, input_data: Any) -> Any:
        """Implementasi abstract method untuk BaseAgent.

        Args:
            input_data: Pertanyaan string atau dict dengan key 'question'.

        Returns:
            Jawaban dari RAG system.
        """
        if isinstance(input_data, dict):
            question = input_data.get("question", "")
        else:
            question = str(input_data)
        return self.query(question)

    def build_index(self, documents: List[Dict[str, Any]]) -> None:
        """Build FAISS index dari dokumen dengan caching otomatis.

        Jika cache sudah ada dan data source sama, akan load dari cache.
        Jika tidak, build baru dan simpan ke cache.

        Args:
            documents: List of dictionaries berisi dokumen dan metadata.
        """
        # Cek apakah cache valid (data source sama)
        if self._is_cache_valid(documents):
            print("  ✅ Menggunakan cache RAG index (lebih cepat!)")
            self.index.load(self.index_path, self.documents_path)
            self.context_documents = documents
            self.status = "index_cached"
            return

        print("  🔄 Building RAG index...")
        start_build = time.time()
        texts = [doc["content"] for doc in documents]
        embeddings = self.embedder.embed(texts)
        metadata_list = [doc.get("metadata", {}) for doc in documents]

        self.index.add_documents(embeddings, texts, metadata_list)
        self.context_documents = documents

        # Simpan ke cache
        self.index.save(self.index_path, self.documents_path)

        build_time = time.time() - start_build
        print(f"  ✅ RAG index built & cached ({len(documents)} dokumen, {build_time:.2f}s)")
        self.status = "index_built"

    def _is_cache_valid(self, documents: List[Dict[str, Any]]) -> bool:
        """Cek apakah cache masih valid.

        Cache valid jika:
        - File cache ada
        - Jumlah dokumen sama

        Args:
            documents: Dokumen yang ingin di-build.

        Returns:
            True jika cache valid.
        """
        if not self.index_path.exists() or not self.documents_path.exists():
            return False

        try:
            loaded_index = FAISSIndex(dimension=self.embedder.dimension)
            loaded_index.load(self.index_path, self.documents_path)
            # Valid jika jumlah dokumen sama
            return loaded_index.size == len(documents)
        except Exception:
            return False

    def query(self, question: str, top_k: int = 4) -> Dict[str, Any]:
        """Query pertanyaan dan dapatkan jawaban.

        Args:
            question: Pertanyaan user.
            top_k: Jumlah konteks yang diambil (default 4 untuk coverage yang lebih baik).

        Returns:
            Dictionary berisi jawaban dan konteks yang digunakan.
        """
        if self.index.size == 0:
            return {
                "answer": "Index belum dibangun. Panggil build_index() terlebih dahulu.",
                "context": [],
                "confidence": 0.0,
                "sources": [],
            }

        # Encode question
        question_embedding = self.embedder.embed_query(question)

        # Deteksi keyword di pertanyaan untuk metadata filter
        metadata_filter = self._detect_metadata_filter(question)

        # Hybrid search: semantic + metadata filter
        if metadata_filter:
            print(f"  🔍 Hybrid search with filter: {metadata_filter}")
            results = self.index.hybrid_search(question_embedding, k=top_k, metadata_filter=metadata_filter)
        else:
            results = self.index.search(question_embedding, k=top_k)

        if not results:
            # Fallback: gunakan semantic search tanpa filter
            results = self.index.search(question_embedding, k=top_k)

        if not results:
            return {
                "answer": "Tidak ditemukan informasi yang relevan.",
                "context": [],
                "confidence": 0.0,
                "sources": [],
            }

        # Gabungkan context dengan title yang lebih informatif
        context_parts = []
        for doc, score, meta in results:
            title = meta.get("type", "Konteks")
            context_parts.append(f"[{title.upper()}]: {doc}")

        context_text = "\n\n".join(context_parts)
        context_texts = [doc for doc, score, meta in results]
        context_titles = [meta.get("type", "unknown") for doc, score, meta in results]

        # Hitung confidence berdasarkan cosine similarity scores
        # FAISS search return distance (smaller = more similar)
        # Convert distance to similarity: similarity = 1 / (1 + distance)
        scores = [score for doc, score, meta in results]
        similarities = [1.0 / (1.0 + score) for score in scores]
        confidence = float(sum(similarities) / len(similarities))
        confidence = max(0.0, min(1.0, confidence))

        # Generate jawaban dengan Groq LLM
        answer = self._generate_answer(question, context_text)

        self.last_result = {
            "question": question,
            "answer": answer,
            "context": context_texts,
            "sources": context_titles,
            "confidence": round(confidence, 4),
        }
        self.status = "queried"
        return self.last_result

    def _detect_metadata_filter(self, question: str) -> Optional[Dict]:
        """Deteksi keyword di pertanyaan untuk metadata filter.

        Args:
            question: Pertanyaan user.

        Returns:
            Dictionary metadata filter atau None jika tidak ada.
        """
        question_lower = question.lower()
        metadata_filter = {}

        # Deteksi bulan spesifik (nama bulan lengkap)
        months_map = {
            "januari": "2024-01", "februari": "2024-02", "maret": "2024-03",
            "april": "2024-04", "mei": "2024-05", "juni": "2024-06",
            "juli": "2024-07", "agustus": "2024-08", "september": "2024-09",
            "oktober": "2024-10", "november": "2024-11", "desember": "2024-12",
        }

        for month, month_key in months_map.items():
            if month in question_lower:
                metadata_filter["period"] = month_key
                metadata_filter["type"] = "monthly_stats"
                return metadata_filter

        # Deteksi kata "bulan" tanpa nama spesifik → ambil semua monthly
        if "bulan" in question_lower and ("paling murah" in question_lower or "terendah" in question_lower or "tertinggi" in question_lower or "best" in question_lower):
            metadata_filter["type"] = "monthly_stats"
            return metadata_filter

        # Deteksi kuartal
        if "kuartal" in question_lower or "q1" in question_lower or "q2" in question_lower or \
           "q3" in question_lower or "q4" in question_lower:
            if "q1" in question_lower or "1" in question_lower:
                metadata_filter["period"] = "Q1"
                metadata_filter["type"] = "quarterly_distribution"
            elif "q2" in question_lower or "2" in question_lower:
                metadata_filter["period"] = "Q2"
                metadata_filter["type"] = "quarterly_distribution"
            elif "q3" in question_lower or "3" in question_lower:
                metadata_filter["period"] = "Q3"
                metadata_filter["type"] = "quarterly_distribution"
            elif "q4" in question_lower or "4" in question_lower:
                metadata_filter["period"] = "Q4"
                metadata_filter["type"] = "quarterly_distribution"
            return metadata_filter

        # Deteksi pola mingguan (hanya jika spesifik menyebut "hari kerja" atau nama hari)
        day_names = ["senin", "selasa", "rabu", "kamis", "jumat", "sabtu", "minggu"]
        has_day_name = any(day in question_lower for day in day_names)
        has_pattern_keyword = "pola mingguan" in question_lower or "hari dalam minggu" in question_lower
        
        if has_day_name or has_pattern_keyword:
            metadata_filter["type"] = "weekly_pattern"
            return metadata_filter

        # Deteksi trend mingguan
        if "mingguan" in question_lower or "trend mingguan" in question_lower:
            metadata_filter["type"] = "weekly_trend"
            return metadata_filter

        # Deteksi YoY comparison (kata "tahun", "perbandingan tahun", "year")
        if any(kw in question_lower for kw in ["tahun", "perbandingan tahunan", "year-over-year", "perbandingan tahun"]):
            metadata_filter["type"] = "yoy_comparison"
            return metadata_filter

        return None

    def _generate_answer(self, question: str, context: str) -> str:
        """Generate jawaban menggunakan Groq LLM.

        Args:
            question: Pertanyaan user.
            context: Konteks dari FAISS search.

        Returns:
            String jawaban.
        """
        if self.client is None:
            # Mode offline: ringkasan dari context
            return self._offline_answer(context)

        system_prompt = """Anda adalah analis bisnis senior yang ahli dalam analisis harga komoditas agrikultur, khususnya cabai merah di Indonesia.

TUGAS: Berikan jawaban yang akurat dan informatif berdasarkan data kontek yang diberikan.

PEDOMAN JAWABAN:
1. Gunakan SEMUA informasi dari konteks yang relevan untuk menjawab pertanyaan
2. Berikan angka spesifik (harga, persentase, tanggal) jika tersedia di konteks
3. Jika konteks mengandung sebagian informasi yang dibutuhkan, gunakan itu dan jelaskan keterbatasan data
4. Berikan insight bisnis yang actionable dan praktis
5. Bahasa Indonesia yang formal, profesional, dan mudah dipahami
6. Jawab langsung ke inti pertanyaan, jangan menghindari dengan mengatakan "tidak ada informasi" jika konteks sudah memberikan data yang relevan
7. Jika ada data historis atau tren di konteks, gunakan untuk memberikan analisis yang lebih kaya"""

        prompt = f"""KONTEKS DARI DATABASE HARGA CABAI MERAH PASAR BERINGHARJO:

{context}

---

PERTANYAAN BISNIS: {question}

Berikan jawaban yang informatif berdasarkan konteks di atas. Manfaatkan semua data yang tersedia untuk menjawab pertanyaan secara lengkap dengan angka spesifik.

JAWABAN:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=600,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error saat generate jawaban: {e}")
            return self._offline_answer(context) + f"\n\n[Catatan: API Error: {str(e)[:100]}]"

    def _offline_answer(self, context: str) -> str:
        """Generate jawaban tanpa LLM dari context yang ada.

        Args:
            context: Konteks dari FAISS search.

        Returns:
            String jawaban berbasis context.
        """
        # Extract informasi utama dari context
        lines = context.split("\n")
        summary_lines = []
        for line in lines:
            line = line.strip()
            if line and ":" in line and "[" in line:
                # Extract bagian setelah title dalam brackets
                if "[KONTEKS]:" in line:
                    summary_lines.append(line.split("[KONTEKS]:", 1)[-1].strip())
                else:
                    summary_lines.append(line)

        if not summary_lines:
            return f"Berdasarkan data historis:\n{context[:500]}"

        main_info = summary_lines[0] if summary_lines else context[:500]
        return f"Berdasarkan analisis data historis harga cabai merah:\n\n{main_info}"

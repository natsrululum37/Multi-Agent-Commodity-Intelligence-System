"""Module konfigurasi untuk aplikasi."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables dari .env file
BASE_DIR = Path(__file__).parent.parent
load_dotenv(BASE_DIR / ".env")


class Config:
    """Konfigurasi aplikasi."""

    # Groq API Configuration
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL_COORDINATOR = "llama-3.3-70b-versatile"
    GROQ_MODEL_RAG = "llama-3.3-70b-versatile"
    GROQ_MODEL_EVALUATOR = "llama-3.1-8b-instant"

    # File paths
    DATA_PATH = BASE_DIR / "cabai.csv"
    REPORTS_PATH = BASE_DIR / "reports"
    MODEL_PATH = BASE_DIR / "models"

    # RAG Configuration
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    FAISS_INDEX_PATH = BASE_DIR / "vectorstore" / "faiss_index.bin"
    CONTEXT_DOCUMENTS_PATH = BASE_DIR / "vectorstore" / "documents.json"

    # Model Configuration
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    @classmethod
    def ensure_directories(cls) -> None:
        """Pastikan semua direktori yang dibutuhkan ada."""
        cls.REPORTS_PATH.mkdir(parents=True, exist_ok=True)
        cls.MODEL_PATH.mkdir(parents=True, exist_ok=True)
        cls.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        cls.FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

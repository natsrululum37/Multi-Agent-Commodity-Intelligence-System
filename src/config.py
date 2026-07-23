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
    CACHE_DIR = BASE_DIR / ".cache"
    CACHE_PATH = CACHE_DIR  # Add alias

    # RAG Configuration
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    # Model Configuration
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    @classmethod
    def ensure_directories(cls) -> None:
        """Pastikan semua direktori yang dibutuhkan ada."""
        cls.REPORTS_PATH.mkdir(parents=True, exist_ok=True)
        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)

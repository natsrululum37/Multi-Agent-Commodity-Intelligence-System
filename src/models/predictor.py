"""Module untuk model prediksi harga."""

import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from typing import Tuple, Optional, Dict


class PricePredictor:
    """Class untuk model prediksi harga."""

    def __init__(self, model_type: str = "random_forest"):
        """Inisialisasi predictor.

        Args:
            model_type: Tipe model ('linear' atau 'random_forest').
        """
        self.model_type = model_type
        self.model = self._init_model(model_type)

    def _init_model(self, model_type: str):
        """Buat model berdasarkan tipe."""
        if model_type == "linear":
            return LinearRegression()
        elif model_type == "random_forest":
            return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def train(self, X: pd.DataFrame, y: pd.Series) -> "PricePredictor":
        """Training model.

        Args:
            X: DataFrame fitur training.
            y: Series target training.

        Returns:
            Instance PricePredictor yang sudah ditraining.
        """
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Prediksi harga.

        Args:
            X: DataFrame fitur untuk prediksi.

        Returns:
            Array berisi hasil prediksi.
        """
        return self.model.predict(X)

    def save(self, filepath: str | Path) -> None:
        """Simpan model ke file.

        Args:
            filepath: Path untuk menyimpan model.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, filepath: str | Path) -> "PricePredictor":
        """Muat model dari file.

        Args:
            filepath: Path file model.

        Returns:
            Instance PricePredictor dengan model yang sudah dimuat.
        """
        with open(filepath, "rb") as f:
            self.model = pickle.load(f)
        return self


class PriceTrendPredictor:
    """Class untuk analisis trend harga secara real-time."""

    def __init__(self):
        self.price_history: list = []

    def add_prices(self, prices: list[float]) -> None:
        """Tambahkan harga ke histori."""
        self.price_history.extend(prices)

    def get_trend(self, days: int = 7) -> Dict:
        """Hitung trend harga.

        Args:
            days: Jumlah hari terakhir untuk analisis.

        Returns:
            Dictionary berisi informasi trend.
        """
        if len(self.price_history) < days:
            return {"trend": "insufficient_data", "days": len(self.price_history)}

        recent_prices = self.price_history[-days:]
        older_prices = self.price_history[-2 * days:-days] if len(self.price_history) >= 2 * days else self.price_history[:-days]

        if not older_prices:
            return {"trend": "insufficient_data", "days": len(self.price_history)}

        recent_avg = np.mean(recent_prices)
        older_avg = np.mean(older_prices)
        change_pct = ((recent_avg - older_avg) / older_avg) * 100

        return {
            "trend": "rising" if change_pct > 5 else "falling" if change_pct < -5 else "stable",
            "change_pct": round(change_pct, 2),
            "recent_avg": round(recent_avg, 2),
            "older_avg": round(older_avg, 2),
            "min": round(min(recent_prices), 2),
            "max": round(max(recent_prices), 2),
        }

    def predict_next_price(self) -> Optional[float]:
        """Prediksi harga berikutnya menggunakan moving average."""
        if len(self.price_history) < 7:
            return None
        return float(np.mean(self.price_history[-7:]))

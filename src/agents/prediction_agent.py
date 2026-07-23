"""PredictionAgent - Untuk prediksi harga."""

import numpy as np
from typing import Dict, Any, Optional, List
from src.agents.base import BaseAgent
from src.models.predictor import PricePredictor, PriceTrendPredictor


class PredictionAgent(BaseAgent):
    """Agent untuk memprediksi harga masa depan dengan multiple models."""

    def __init__(self, model_type: str = "random_forest"):
        super().__init__(name="PredictionAgent")
        self.predictor = PricePredictor(model_type)
        self.trend_predictor = PriceTrendPredictor()
        self.model_trained = False
        self.metrics: Optional[Dict] = None

    def process(self, input_data: Any) -> Any:
        """Implementasi abstract method untuk BaseAgent.

        Args:
            input_data: List harga atau dict dengan key 'prices'.

        Returns:
            Hasil prediksi dan rekomendasi.
        """
        if isinstance(input_data, dict):
            prices = input_data.get("prices", [])
        else:
            prices = list(input_data)
        return self.analyze_trend(prices)

    def train_with_evaluation(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_test: np.ndarray, y_test: np.ndarray) -> "PredictionAgent":
        """Training model dan evaluasi performa.

        Args:
            X_train: Fitur training.
            y_train: Target training.
            X_test: Fitur test.
            y_test: Target test.

        Returns:
            Instance PredictionAgent yang sudah ditraining.
        """
        self.predictor.train(X_train, y_train)
        predictions = self.predictor.predict(X_test)

        # Evaluasi model
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        # MAPE
        mask = y_test != 0
        mape = np.mean(np.abs((y_test[mask] - predictions[mask]) / y_test[mask])) * 100

        self.metrics = {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "r2": round(r2, 4),
            "mape": round(mape, 2),
        }
        self.model_trained = True
        self.status = "trained"
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Prediksi harga.

        Args:
            X_test: Fitur test.

        Returns:
            Array hasil prediksi.
        """
        if not self.model_trained:
            raise ValueError("Model belum dilatih. Panggil train_with_evaluation() terlebih dahulu.")
        return self.predictor.predict(X_test)

    def analyze_trend(self, prices: List[float], lookback_days: int = 14) -> Dict[str, Any]:
        """Analisis trend harga dengan pendekatan multi-metode.

        Args:
            prices: List harga historis.
            lookback_days: Jumlah hari terakhir untuk analisis.

        Returns:
            Dictionary berisi analisis trend komprehensif.
        """
        self.trend_predictor.add_prices(prices)

        # Metode 1: Moving Average
        recent = prices[-lookback_days:]
        older = prices[-2*lookback_days:-lookback_days] if len(prices) >= 2*lookback_days else prices[:-lookback_days]

        recent_avg = np.mean(recent)
        older_avg = np.mean(older) if older else recent_avg
        ma_change_pct = ((recent_avg - older_avg) / older_avg) * 100 if older_avg > 0 else 0

        # Metode 2: Linear Regression Trend
        n = len(recent)
        x = np.arange(n)
        y = np.array(recent)
        if n > 1:
            slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
            regression_trend = "rising" if slope > 50 else "falling" if slope < -50 else "stable"
        else:
            regression_trend = "stable"
            slope = 0

        # Metode 3: Momentum (perubahan 7 hari terakhir)
        if len(prices) >= 7:
            week_ago_price = prices[-7]
            momentum = ((prices[-1] - week_ago_price) / week_ago_price) * 100
        else:
            momentum = 0

        # Konsensus dari semua metode
        methods = [
            ("Moving Average", abs(ma_change_pct) > 5, "rising" if ma_change_pct > 5 else "falling" if ma_change_pct < -5 else "stable"),
            ("Regression", abs(slope) > 50, regression_trend),
            ("Momentum", abs(momentum) > 5, "rising" if momentum > 5 else "falling" if momentum < -5 else "stable"),
        ]

        rising_count = sum(1 for _, _, trend in methods if trend == "rising")
        falling_count = sum(1 for _, _, trend in methods if trend == "falling")

        if rising_count >= 2:
            consensus_trend = "rising"
        elif falling_count >= 2:
            consensus_trend = "falling"
        else:
            consensus_trend = "mixed"

        next_price = self.trend_predictor.predict_next_price()

        result = {
            "consensus_trend": consensus_trend,
            "method_results": {
                "moving_average": {
                    "change_pct": round(ma_change_pct, 2),
                    "trend": "rising" if ma_change_pct > 5 else "falling" if ma_change_pct < -5 else "stable",
                },
                "regression": {
                    "slope": round(slope, 2),
                    "trend": regression_trend,
                },
                "momentum": {
                    "7d_change_pct": round(momentum, 2),
                    "trend": "rising" if momentum > 5 else "falling" if momentum < -5 else "stable",
                },
            },
            "next_predicted_price": round(next_price, 2) if next_price else None,
            "current_price": prices[-1] if prices else None,
            "recent_avg": round(recent_avg, 2),
            "older_avg": round(older_avg, 2),
            "price_range": {
                "min": round(float(min(recent)), 2),
                "max": round(float(max(recent)), 2),
            },
        }

        self.last_result = result
        self.status = "predicted"
        return result

    def generate_recommendations(self, analysis: Dict[str, Any], current_price: float) -> List[str]:
        """Generate rekomendasi bisnis dari prediksi.

        Args:
            analysis: Hasil analisis trend.
            current_price: Harga saat ini.

        Returns:
            List of recommendation strings.
        """
        recommendations = []
        trend = analysis.get("consensus_trend", "unknown")
        next_price = analysis.get("next_predicted_price")
        method_results = analysis.get("method_results", {})

        # Rekomendasi berdasarkan trend
        if trend == "falling":
            ma_trend = method_results.get("moving_average", {}).get("trend", "")
            if ma_trend == "falling":
                change = method_results.get("moving_average", {}).get("change_pct", 0)
                recommendations.append(
                    f"📉 TREND TURUN: Analisis moving average menunjukkan penurunan {abs(change):.1f}%. "
                    f"Saran: Tunggu harga stabil sebelum membeli dalam jumlah besar."
                )
            momentum = method_results.get("momentum", {}).get("7d_change_pct", 0)
            if momentum < -5:
                recommendations.append(
                    f"📊 MOMENTUM NEGATIF: 7 hari terakhir turun {abs(momentum):.1f}%. "
                    f"Harga saat ini Rp {current_price:,.0f}, kemungkinan masih ada ruang turun."
                )
            if next_price:
                recommendations.append(
                    f"💰 STRATEGI BELI: Harga berikutnya diprediksi Rp {next_price:,.0f}. "
                    f"Beli bertahap saat harga di bawah Rp {current_price * 0.95:,.0f}."
                )
        elif trend == "rising":
            recommendations.append(
                f"📈 TREND NAIK: Konsensus multi-metode menunjukkan kenaikan. "
                f"Saran: Beli sekarang sebelum harga naik lebih tinggi."
            )
            if next_price:
                recommendations.append(
                    f"💰 STRATEGI BELI: Harga berikutnya diprediksi Rp {next_price:,.0f}. "
                    f"Target beli optimal: Rp {current_price * 0.98:,.0f} (diskon kecil dari harga saat ini)."
                )
        else:
            recommendations.append(
                f"⚖️ TREND CAMPURAN: Beberapa metode menunjukkan arah berbeda. "
                f"Saran: Pertimbangkan strategi hold dan monitor perkembangan harga."
            )

        # Rekomendasi umum
        price_range = analysis.get("price_range", {})
        if price_range:
            recommendations.append(
                f"📏 RANGE HARGA TERAKHIR: Rp {price_range['min']:,.0f} - Rp {price_range['max']:,.0f}. "
                f"Harga saat ini Rp {current_price:,.0f}"
                f"({'di atas' if current_price > price_range['max'] * 0.95 else 'dalam'} range normal)."
            )

        self.last_result = recommendations
        self.status = "recommendations_generated"
        return recommendations

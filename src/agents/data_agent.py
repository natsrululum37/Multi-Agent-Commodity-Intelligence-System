"""DataAgent - Untuk analisis data harga komoditas."""

import pandas as pd
from typing import Dict, Any, Optional
from src.agents.base import BaseAgent
from src.data.loader import load_data, get_summary
from src.data.preprocessing import clean_data, add_features, prepare_training_data


class DataAgent(BaseAgent):
    """Agent untuk membersihkan dan menganalisis data harga."""

    def __init__(self):
        super().__init__(name="DataAgent")
        self.data: Optional[pd.DataFrame] = None
        self.summary: Optional[Dict] = None

    def process(self, input_data: Any) -> Any:
        """Implementasi abstract method untuk BaseAgent.

        Args:
            input_data: Path file CSV atau dict dengan key 'filepath'.

        Returns:
            Hasil analisis data.
        """
        if isinstance(input_data, dict):
            filepath = input_data.get("filepath", "cabai.csv")
        else:
            filepath = str(input_data)
        self.load_and_clean(filepath)
        return self.analyze()

    def load_and_clean(self, filepath: str = "cabai.csv") -> pd.DataFrame:
        """Muat dan bersihkan data.

        Args:
            filepath: Path file CSV.

        Returns:
            DataFrame yang sudah dibersihkan.
        """
        df = load_data(filepath)
        cleaned_df = clean_data(df)
        self.data = cleaned_df
        self.summary = get_summary(cleaned_df)
        self.status = "loaded"
        return cleaned_df

    def analyze(self) -> Dict[str, Any]:
        """Lakukan analisis komprehensif.

        Returns:
            Dictionary berisi hasil analisis.
        """
        if self.data is None:
            raise ValueError("Data belum dimuat. Panggil load_and_clean() terlebih dahulu.")

        featured_df = add_features(self.data)

        analysis = {
            "total_records": len(self.data),
            "date_range": (str(self.data["tanggal"].min()), str(self.data["tanggal"].max())),
            "price_stats": {
                "min": float(self.data["harga"].min()),
                "max": float(self.data["harga"].max()),
                "mean": float(self.data["harga"].mean()),
                "std": float(self.data["harga"].std()),
                "median": float(self.data["harga"].median()),
            },
            "monthly_trend": [],
            "volatility": float(self.data["harga"].std() / self.data["harga"].mean() * 100),
        }

        # Analisis trend bulanan
        df_monthly = self.data.copy()
        df_monthly["month"] = df_monthly["tanggal"].dt.to_period("M").astype(str)
        monthly_avg = df_monthly.groupby("month")["harga"].mean()

        for month, avg_price in monthly_avg.items():
            analysis["monthly_trend"].append({
                "month": month,
                "avg_price": round(float(avg_price), 2),
            })

        self.last_result = analysis
        self.status = "analyzed"
        return analysis

    def generate_insights(self) -> list[str]:
        """Generate insight bisnis dari data.

        Returns:
            List of insight strings.
        """
        if self.last_result is None:
            self.analyze()

        result = self.last_result
        price_stats = result["price_stats"]
        volatility = result["volatility"]

        insights = []

        # Insight tentang volatilitas
        if volatility > 30:
            insights.append(
                f"Harga cabai merah menunjukkan volatilitas TINGGI ({volatility:.2f}%). "
                "Perlu strategi manajemen risiko yang ketat."
            )
        elif volatility > 15:
            insights.append(
                f"Harga cabai merah menunjukkan fluktuasi MODERATE ({volatility:.2f}%). "
                "Monitoring berkala disarankan."
            )
        else:
            insights.append(
                f"Harga cabai merah STABIL ({volatility:.2f}%). "
                "Risiko fluktuasi rendah."
            )

        # Insight tentang range harga
        insights.append(
            f"Range harga: Rp {price_stats['min']:,.0f} - Rp {price_stats['max']:,.0f}/kg. "
            f"Selisih maksimum: Rp {price_stats['max'] - price_stats['min']:,.0f}"
        )

        # Insight tentang trend
        trends = result.get("monthly_trend", [])
        if len(trends) >= 2:
            first_avg = trends[0]["avg_price"]
            last_avg = trends[-1]["avg_price"]
            change_pct = ((last_avg - first_avg) / first_avg) * 100
            direction = "naik" if change_pct > 0 else "turun"
            insights.append(
                f"Trend harga dari awal ke akhir periode: {direction.capitalize()} {abs(change_pct):.2f}%. "
                f"(Rp {first_avg:,.0f} → Rp {last_avg:,.0f})"
            )

        self.last_result = insights
        self.status = "insights_generated"
        return insights

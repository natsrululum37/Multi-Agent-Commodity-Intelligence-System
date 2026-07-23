"""DataAgent - Untuk analisis data harga komoditas."""

import pandas as pd
from typing import Dict, Any, Optional
from src.agents.base import BaseAgent
from src.data.loader import load_data, get_summary
from src.data.preprocessing import clean_data, add_features, prepare_training_data, get_commodity_stats


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
        """Lakukan analisis komprehensif PER KOMODITAS.

        Returns:
            Dictionary berisi hasil analisis per komoditas.
        """
        if self.data is None:
            raise ValueError("Data belum dimuat. Panggil load_and_clean() terlebih dahulu.")

        featured_df = add_features(self.data)

        # Statistik GLOBAL (untuk overview)
        global_stats = {
            "total_records": len(self.data),
            "unique_commodities": int(self.data["nama_komoditas"].nunique()),
            "commodities_list": self.data["nama_komoditas"].unique().tolist(),
            "date_range": (str(self.data["tanggal"].min()), str(self.data["tanggal"].max())),
            "global_price_stats": {
                "min": float(self.data["harga"].min()),
                "max": float(self.data["harga"].max()),
                "mean": float(self.data["harga"].mean()),
                "std": float(self.data["harga"].std()),
                "median": float(self.data["harga"].median()),
            },
        }

        # Statistik PER KOMODITAS
        commodity_stats = get_commodity_stats(self.data)

        # Trend bulanan PER KOMODITAS
        monthly_trends = {}
        for commodity, group in self.data.groupby("nama_komoditas"):
            df_monthly = group.copy()
            df_monthly["month"] = df_monthly["tanggal"].dt.to_period("M").astype(str)
            monthly_avg = df_monthly.groupby("month")["harga"].mean()

            trends = []
            for month, avg_price in monthly_avg.items():
                trends.append({
                    "month": str(month),
                    "avg_price": round(float(avg_price), 2),
                })
            monthly_trends[commodity] = trends

        # Volatilitas PER KOMODITAS
        volatility_per_commodity = {}
        for commodity, group in self.data.groupby("nama_komoditas"):
            avg = group["harga"].mean()
            std = group["harga"].std()
            volatility_per_commodity[commodity] = round(float(std / avg * 100), 2)

        analysis = {
            **global_stats,
            "commodity_stats": commodity_stats,
            "monthly_trends": monthly_trends,
            "volatility_per_commodity": volatility_per_commodity,
        }

        self.last_result = analysis
        self.status = "analyzed"
        return analysis

    def generate_insights(self) -> list[str]:
        """Generate insight bisnis PER KOMODITAS dari data.

        Returns:
            List of insight strings.
        """
        if self.last_result is None:
            self.analyze()

        result = self.last_result
        commodity_stats = result.get("commodity_stats", {})
        volatility_per_commodity = result.get("volatility_per_commodity", {})
        monthly_trends = result.get("monthly_trends", {})

        insights = []

        # Insight per komoditas
        for commodity, stats in commodity_stats.items():
            short_name = commodity.split(",")[0]  # Hapus ",1 kg"
            volatility = volatility_per_commodity.get(commodity, 0)

            # Insight volatilitas
            if volatility > 30:
                insights.append(
                    f"{short_name}: Harga menunjukkan volatilitas TINGGI ({volatility:.2f}%). "
                    "Perlu strategi manajemen risiko yang ketat."
                )
            elif volatility > 15:
                insights.append(
                    f"{short_name}: Harga menunjukkan fluktuasi MODERATE ({volatility:.2f}%). "
                    "Monitoring berkala disarankan."
                )
            else:
                insights.append(
                    f"{short_name}: Harga STABIL ({volatility:.2f}%). "
                    "Risiko fluktuasi rendah."
                )

            # Insight range harga
            insights.append(
                f"{short_name}: Range harga Rp {stats['min']:,.0f} - Rp {stats['max']:,.0f}/kg. "
                f"Rata-rata Rp {stats['mean']:,.0f}/kg."
            )

            # Insight trend
            trends = monthly_trends.get(commodity, [])
            if len(trends) >= 2:
                first_avg = trends[0]["avg_price"]
                last_avg = trends[-1]["avg_price"]
                change_pct = ((last_avg - first_avg) / first_avg) * 100
                direction = "naik" if change_pct > 0 else "turun"
                insights.append(
                    f"{short_name}: Trend harga dari awal ke akhir periode: "
                    f"{direction.capitalize()} {abs(change_pct):.2f}%. "
                    f"(Rp {first_avg:,.0f} → Rp {last_avg:,.0f})"
                )

            insights.append("")  # Spacer antar komoditas

        self.last_result = insights
        self.status = "insights_generated"
        return insights

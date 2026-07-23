"""Module untuk generate dokumen konteks dari data."""

import json
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict
from src.data.loader import load_data
from src.data.preprocessing import clean_data


def generate_documents_from_data(filepath: str = "cabai.csv") -> List[Dict]:
    """Generate dokumen konteks detail dari data CSV untuk RAG.

    Dokumen dibuat granular agar FAISS search lebih akurat.

    Args:
        filepath: Path file CSV.

    Returns:
        List of dictionaries berisi dokumen.
    """
    df = load_data(filepath)
    df = clean_data(df)

    documents = []

    # === 1. RINGKASAN DATA ===
    documents.append({
        "type": "summary",
        "title": "Ringkasan Dataset Harga Cabai Merah",
        "content": (
            f"Dataset harga cabai merah di Pasar Beringharjo, Yogyakarta. "
            f"Total {len(df)} catatan data harian. "
            f"Periode: {df['tanggal'].min().strftime('%d %B %Y')} sampai {df['tanggal'].max().strftime('%d %B %Y')}. "
            f"Harga minimum Rp {df['harga'].min():,.0f}/kg pada {df.loc[df['harga'].idxmin(), 'tanggal'].strftime('%d %B %Y')}. "
            f"Harga maksimum Rp {df['harga'].max():,.0f}/kg pada {df.loc[df['harga'].idxmax(), 'tanggal'].strftime('%d %B %Y')}. "
            f"Rata-rata harga Rp {df['harga'].mean():,.0f}/kg. "
            f"Median harga Rp {df['harga'].median():,.0f}/kg."
        ),
        "metadata": {"source": "summary"},
    })

    # === 2. STATISTIK PER BULAN (granular) ===
    df_copy = df.copy()
    df_copy["month"] = df_copy["tanggal"].dt.to_period("M").astype(str)
    df_copy["month_name"] = df_copy["tanggal"].dt.strftime("%B %Y")
    monthly = df_copy.groupby(["month", "month_name"]).agg(
        avg_price=("harga", "mean"),
        min_price=("harga", "min"),
        max_price=("harga", "max"),
        std_price=("harga", "std"),
        median_price=("harga", "median"),
        record_count=("harga", "count"),
    ).reset_index()

    for _, row in monthly.iterrows():
        month_key = str(row["month"])
        month_label = str(row["month_name"])
        content = (
            f"Bulan {month_label}: "
            f"Rata-rata harga Rp {row['avg_price']:,.0f}/kg, "
            f"minimum Rp {row['min_price']:,.0f}, "
            f"maksimum Rp {row['max_price']:,.0f}, "
            f"median Rp {row['median_price']:,.0f}, "
            f"standar deviasi Rp {row['std_price']:,.0f}. "
            f"Terdapat {int(row['record_count'])} catatan harga."
        )
        documents.append({
            "type": "monthly_stats",
            "title": f"Statistik Harga {month_label}",
            "content": content,
            "metadata": {"period": month_key, "type": "monthly"},
        })

    # === 3. HARGA TERTINGGI DAN TERENDAH ===
    top5 = df.nlargest(5, "harga")[["tanggal", "harga"]]
    top5_str = ", ".join(
        f"{row['tanggal'].strftime('%d %B')}: Rp {row['harga']:,.0f}"
        for _, row in top5.iterrows()
    )
    documents.append({
        "type": "top_prices",
        "title": "5 Harga Tertinggi",
        "content": (
            f"Lima hari dengan harga cabai merah tertinggi: "
            f"{top5_str}. "
            f"Harga tertinggi tercatat Rp {top5.iloc[0]['harga']:,.0f}/kg."
        ),
        "metadata": {"type": "extreme_values"},
    })

    bottom5 = df.nsmallest(5, "harga")[["tanggal", "harga"]]
    bottom5_str = ", ".join(
        f"{row['tanggal'].strftime('%d %B')}: Rp {row['harga']:,.0f}"
        for _, row in bottom5.iterrows()
    )
    documents.append({
        "type": "bottom_prices",
        "title": "5 Harga Terendah",
        "content": (
            f"Lima hari dengan harga cabai merah terendah: "
            f"{bottom5_str}. "
            f"Harga terendah tercatat Rp {bottom5.iloc[0]['harga']:,.0f}/kg."
        ),
        "metadata": {"type": "extreme_values"},
    })

    # === 4. ANALISIS VOLATILITAS ===
    avg_price = df["harga"].mean()
    std_price = df["harga"].std()
    volatility = std_price / avg_price * 100

    # Hitung hari dengan perubahan harga terbesar
    df_sorted = df.sort_values("tanggal").copy()
    df_sorted["daily_change"] = df_sorted["harga"].pct_change() * 100
    max_gain_day = df_sorted.loc[df_sorted["daily_change"].idxmax()]
    max_drop_day = df_sorted.loc[df_sorted["daily_change"].idxmin()]

    documents.append({
        "type": "volatility_analysis",
        "title": "Analisis Volatilitas Harga",
        "content": (
            f"Koefisien variasi harga adalah {volatility:.2f}%. "
            f"Level volatilitas: {'TINGGI (>30%)' if volatility > 30 else 'MODERATE (15-30%)' if volatility > 15 else 'RENDAH (<15%)'}. "
            f"Standar deviasi Rp {std_price:,.0f} dari rata-rata Rp {avg_price:,.0f}. "
            f"Hari dengan kenaikan tertinggi: {max_gain_day['tanggal'].strftime('%d %B %Y')} (+{max_gain_day['daily_change']:.1f}%). "
            f"Hari dengan penurunan tertinggi: {max_drop_day['tanggal'].strftime('%d %B %Y')} ({max_drop_day['daily_change']:.1f}%)."
        ),
        "metadata": {"type": "volatility"},
    })

    # === 5. TREND PERIODE ===
    # Split menjadi Q1, Q2, Q3
    df_copy["quarter"] = df_copy["tanggal"].dt.quarter
    quarterly = df_copy.groupby("quarter")["harga"].agg(["mean", "min", "max", "count"]).reset_index()

    for _, row in quarterly.iterrows():
        q_num = int(row["quarter"])
        content = (
            f"Kuartal {q_num}: Rata-rata harga Rp {row['mean']:,.0f}/kg, "
            f"range Rp {row['min']:,.0f} - Rp {row['max']:,.0f}, "
            f"dari {int(row['count'])} catatan data."
        )
        documents.append({
            "type": "quarterly_stats",
            "title": f"Statistik Kuartal {q_num}",
            "content": content,
            "metadata": {"period": f"Q{q_num}", "type": "quarterly"},
        })

    # Overall trend
    first_half = df_copy[df_copy["tanggal"] < df_copy["tanggal"].median()]["harga"].mean()
    second_half = df_copy[df_copy["tanggal"] >= df_copy["tanggal"].median()]["harga"].mean()
    overall_trend_pct = ((second_half - first_half) / first_half) * 100

    documents.append({
        "type": "overall_trend",
        "title": "Trend Harga Keseluruhan",
        "content": (
            f"Trend harga keseluruhan periode: "
            f"{'NAIK' if overall_trend_pct > 0 else 'TURUN'} {abs(overall_trend_pct):.2f}%. "
            f"Paruh pertama rata-rata Rp {first_half:,.0f}/kg, "
            f"paruh kedua rata-rata Rp {second_half:,.0f}/kg."
        ),
        "metadata": {"type": "trend"},
    })

    # === 6. INSIGHTS PER DIVISI ===
    # Cari bulan dengan harga terendah dan tertinggi
    best_buy_month = monthly.loc[monthly["avg_price"].idxmin()]
    best_sell_month = monthly.loc[monthly["avg_price"].idxmax()]

    business_docs = [
        {
            "type": "business_insight",
            "title": "Rekomendasi Procurement",
            "content": (
                f"Divisi Procurement: Waktu terbaik membeli cabai merah adalah pada bulan "
                f"{best_buy_month['month_name']} dengan rata-rata Rp {best_buy_month['avg_price']:,.0f}/kg. "
                f"Threshold beli optimal: harga < Rp {avg_price * 0.85:,.0f} (15% di bawah rata-rata). "
                f"Jangan beli saat harga > Rp {avg_price * 1.15:,.0f}."
            ),
            "metadata": {"division": "procurement"},
        },
        {
            "type": "business_insight",
            "title": "Rekomendasi Sales & Marketing",
            "content": (
                f"Divisi Sales: Waktu terbaik menjual stok adalah pada bulan "
                f"{best_sell_month['month_name']} dengan rata-rata Rp {best_sell_month['avg_price']:,.0f}/kg. "
                f"Margin potensial: Rp {best_sell_month['avg_price'] - best_buy_month['avg_price']:,.0f}/kg "
                f"(selisih antara harga jual tertinggi dan beli terendah)."
            ),
            "metadata": {"division": "sales"},
        },
        {
            "type": "business_insight",
            "title": "Rekomendasi Finance",
            "content": (
                f"Divisi Finance: Budget yang disarankan per kg adalah Rp {avg_price:,.0f}. "
                f"Buffer keamanan untuk fluktuasi: +/- Rp {std_price:,.0f} "
                f"({volatility:.1f}% dari rata-rata). "
                f"Proyeksi kebutuhan cash flow bulanan untuk 1 ton: Rp {avg_price * 1000:,.0f}."
            ),
            "metadata": {"division": "finance"},
        },
        {
            "type": "business_insight",
            "title": "Rekomendasi Logistik",
            "content": (
                f"Divisi Logistik: Strategi stok optimal - akumulasikan stok saat harga rendah "
                f"(di bawah Rp {avg_price * 0.9:,.0f}) dan keluarkan saat harga tinggi. "
                f"Safety stock disarankan: 2 minggu dari permintaan rata-rata."
            ),
            "metadata": {"division": "logistics"},
        },
        {
            "type": "business_insight",
            "title": "Rekomendasi Manajemen Risiko",
            "content": (
                f"Divisi Manajemen Risiko: Level harga kritis untuk early warning - "
                f"Jika harga > Rp {avg_price * 1.3:,.0f} (30% di atas rata-rata), "
                f"aktivasi protokol respons cepat. "
                f"Level harga aman: antara Rp {avg_price * 0.7:,.0f} dan Rp {avg_price * 1.3:,.0f}."
            ),
            "metadata": {"division": "risk_management"},
        },
    ]
    documents.extend(business_docs)

    # === 7. DATA HARIAN TERTENTUK (sample untuk referensi) ===
    # Ambil beberapa tanggal penting sebagai referensi
    important_dates = df_sorted.iloc[[0, -1]]  # First and last day
    for _, row in important_dates.iterrows():
        documents.append({
            "type": "reference_date",
            "title": f"Data Referensi {row['tanggal'].strftime('%d %B %Y')}",
            "content": (
                f"Data referensi: {row['tanggal'].strftime('%d %B %Y')} - "
                f"Harga cabai merah di Pasar Beringharjo: Rp {row['harga']:,.0f}/kg."
            ),
            "metadata": {"type": "reference", "date": str(row["tanggal"].date())},
        })

    # === 8. POLA HARIAN MINGGUAN (Day-of-Week Analysis) ===
    df_copy["day_name"] = df_copy["tanggal"].dt.strftime("%A")
    df_copy["week_number"] = df_copy["tanggal"].dt.isocalendar().week.astype(int)
    df_copy["year"] = df_copy["tanggal"].dt.year
    df_copy["month_num"] = df_copy["tanggal"].dt.month

    daily_patterns = df_copy.groupby("day_name")["harga"].agg(["mean", "min", "max", "std"]).reset_index()
    daily_patterns.columns = ["day", "avg_price", "min_price", "max_price", "std_price"]

    # Sort berdasarkan hari dalam seminggu
    day_order = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
    daily_patterns["day_order"] = daily_patterns["day"].map(
        {day: i for i, day in enumerate(day_order)}
    )
    daily_patterns = daily_patterns.dropna(subset=["day_order"]).sort_values("day_order").drop(columns=["day_order"])

    if len(daily_patterns) > 0:
        daily_pattern_str = "; ".join([
            f"{row['day']}: rata-rata Rp {row['avg_price']:,.0f} (min Rp {row['min_price']:,.0f}, max Rp {row['max_price']:,.0f})"
            for _, row in daily_patterns.iterrows()
        ])

        # Cari hari paling murah dan paling mahal
        cheapest_day = daily_patterns.loc[daily_patterns["avg_price"].idxmin()]
        most_expensive_day = daily_patterns.loc[daily_patterns["avg_price"].idxmax()]

        documents.append({
            "type": "weekly_pattern",
            "title": "Pola Harga Berdasarkan Hari dalam Minggu",
            "content": (
                f"Analisis pola harga cabai merah per hari dalam minggu:\n"
                f"{daily_pattern_str}\n\n"
                f"📌 Hari paling murah untuk beli: {cheapest_day['day']} (rata-rata Rp {cheapest_day['avg_price']:,.0f}/kg)\n"
                f"📌 Hari paling mahal: {most_expensive_day['day']} (rata-rata Rp {most_expensive_day['avg_price']:,.0f}/kg)\n"
                f"Selisih harian: Rp {most_expensive_day['avg_price'] - cheapest_day['avg_price']:,.0f}"
            ),
            "metadata": {"type": "weekly_pattern"},
        })

    # === 9. POLA PER MINGGU (Weekly Trend) ===
    weekly = df_copy.groupby("week_number")["harga"].agg(["mean", "min", "max", "count"]).reset_index()
    weekly.columns = ["week", "avg_price", "min_price", "max_price", "count"]

    if len(weekly) > 4:
        # Bandingkan 4 minggu terakhir vs 4 minggu sebelumnya
        mid = len(weekly) // 2
        recent_4_weeks = weekly.tail(4)
        prev_4_weeks = weekly.iloc[mid:mid+4] if len(weekly) >= 8 else weekly.head(4)

        recent_avg = recent_4_weeks["avg_price"].mean()
        prev_avg = prev_4_weeks["avg_price"].mean()
        weekly_trend_pct = ((recent_avg - prev_avg) / prev_avg) * 100 if prev_avg > 0 else 0

        documents.append({
            "type": "weekly_trend",
            "title": "Tren Mingguan Harga Cabai Merah",
            "content": (
                f"Tren mingguan harga cabai merah (berdasarkan {len(weekly)} minggu data):\n"
                f"4 minggu terakhir rata-rata: Rp {recent_avg:,.0f}/kg\n"
                f"Periode sebelumnya rata-rata: Rp {prev_avg:,.0f}/kg\n"
                f"Perubahan: {'NAIK' if weekly_trend_pct > 0 else 'TURUN'} {abs(weekly_trend_pct):.2f}%\n\n"
                f"Minggu dengan harga tertinggi: Minggu ke-{weekly.loc[weekly['avg_price'].idxmax(), 'week']} "
                f"(Rp {weekly['avg_price'].max():,.0f}/kg)\n"
                f"Minggu dengan harga terendah: Minggu ke-{weekly.loc[weekly['avg_price'].idxmin(), 'week']} "
                f"(Rp {weekly['avg_price'].min():,.0f}/kg)"
            ),
            "metadata": {"type": "weekly_trend"},
        })

    # === 10. DISTRIBUSI HARGA PER QUARTER ===
    quarterly_detail = df_copy.groupby("quarter")["harga"].agg(
        ["mean", "median", "std", "min", "max", "count",
         lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
    ).reset_index()
    quarterly_detail.columns = ["quarter", "mean", "median", "std", "min", "max", "count", "q1", "q3"]

    for _, row in quarterly_detail.iterrows():
        q_num = int(row["quarter"])
        content = (
            f"Distribusi harga kuartal {q_num}:\n"
            f"- Rata-rata: Rp {row['mean']:,.0f}/kg | Median: Rp {row['median']:,.0f}/kg\n"
            f"- Range: Rp {row['min']:,.0f} - Rp {row['max']:,.0f}/kg\n"
            f"- Quartile 25%: Rp {row['q1']:,.0f} | Quartile 75%: Rp {row['q3']:,.0f}\n"
            f"- Standar deviasi: Rp {row['std']:,.0f}\n"
            f"- Jumlah data: {int(row['count'])} catatan"
        )
        documents.append({
            "type": "quarterly_distribution",
            "title": f"Distribusi Harga Kuartal {q_num}",
            "content": content,
            "metadata": {"period": f"Q{q_num}", "type": "quarterly_distribution"},
        })

    # === 11. ANALISIS YEAR-OVER-YEAR (jika ada data > 1 tahun) ===
    yearly = df_copy.groupby("year")["harga"].agg(["mean", "min", "max", "count"]).reset_index()

    if len(yearly) > 1:
        for _, row in yearly.iterrows():
            year = int(row["year"])
            documents.append({
                "type": "yearly_stats",
                "title": f"Statistik Tahunan {year}",
                "content": (
                    f"Statistik harga cabai merah tahun {year}:\n"
                    f"- Rata-rata: Rp {row['mean']:,.0f}/kg\n"
                    f"- Minimum: Rp {row['min']:,.0f}/kg\n"
                    f"- Maksimum: Rp {row['max']:,.0f}/kg\n"
                    f"- Total catatan: {int(row['count'])} hari"
                ),
                "metadata": {"period": str(year), "type": "yearly"},
            })

        # Perbandingan antar tahun
        yearly_sorted = yearly.sort_values("year")
        first_year = yearly_sorted.iloc[0]
        last_year = yearly_sorted.iloc[-1]
        yoy_change = ((last_year["mean"] - first_year["mean"]) / first_year["mean"]) * 100

        if len(yearly_sorted) >= 2:
            documents.append({
                "type": "yoy_comparison",
                "title": "Perbandingan Tahunan (Year-over-Year)",
                "content": (
                    f"Perbandingan tahunan harga cabai merah:\n"
                    f"- Tahun {int(first_year['year'])}: rata-rata Rp {first_year['mean']:,.0f}/kg\n"
                    f"- Tahun {int(last_year['year'])}: rata-rata Rp {last_year['mean']:,.0f}/kg\n"
                    f"- Perubahan YoY: {'NAIK' if yoy_change > 0 else 'TURUN'} {abs(yoy_change):.2f}%\n"
                    f"Catatan: Data mungkin belum lengkap jika periode < 2 tahun penuh."
                ),
                "metadata": {"type": "yoy_comparison"},
            })

    return documents


def save_documents(documents: List[Dict], filepath: str | Path) -> None:
    """Simpan dokumen ke file JSON.

    Args:
        documents: List dokumen.
        filepath: Path file output.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)


def load_documents(filepath: str | Path) -> List[Dict]:
    """Muat dokumen dari file JSON.

    Args:
        filepath: Path file JSON.

    Returns:
        List of dictionaries berisi dokumen.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

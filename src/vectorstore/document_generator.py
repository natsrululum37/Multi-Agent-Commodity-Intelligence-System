"""Module untuk generate dokumen konteks dari data - PER KOMODITAS."""

import json
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict
from src.data.loader import load_data
from src.data.preprocessing import clean_data


def generate_documents_from_data(filepath: str = "cabai.csv") -> List[Dict]:
    """Generate dokumen konteks detail dari data CSV untuk RAG - OPTIMIZED.

    Dokumen dibuat lebih ringkas dan efisien untuk mempercepat RAG processing.
    
    Returns:
        List of dictionaries berisi dokumen yang sudah dioptimasi.
    """
    import time
    start_time = time.time()
    
    df = load_data(filepath)
    df = clean_data(df)

    documents = []

    # === 1. RINGKASAN DATA GLOBAL (RINGKAS) ===
    documents.append({
        "type": "summary",
        "title": f"Ringkasan Dataset Harga Komoditas Pasar Beringharjo",
        "content": (
            f"Dataset harga komoditas di Pasar Beringharjo, Yogyakarta. "
            f"Total {len(df)} catatan dari {df['nama_komoditas'].nunique()} komoditas. "
            f"Periode: {df['tanggal'].min().strftime('%d %B %Y')} - {df['tanggal'].max().strftime('%d %B %Y')}."
        ),
        "metadata": {"source": "summary_global"},
    })

    commodities = df["nama_komoditas"].unique()
    
    # Pre-compute semua data per komoditas dalam satu loop
    for commodity in commodities:
        short_name = commodity.split(",")[0]
        commodity_df = df[df["nama_komoditas"] == commodity].sort_values("tanggal")

        # --- STATISTIK DASAR PER KOMODITAS (DIGABUNG) ---
        avg_price = commodity_df["harga"].mean()
        std_price = commodity_df["harga"].std()
        min_price = commodity_df["harga"].min()
        max_price = commodity_df["harga"].max()
        median_price = commodity_df["harga"].median()
        volatility = std_price / avg_price * 100 if avg_price > 0 else 0
        
        documents.append({
            "type": "summary",
            "title": f"Statistik Dasar {short_name}",
            "content": (
                f"{short_name}: {len(commodity_df)} data harian. "
                f"Range: Rp {min_price:,.0f} - Rp {max_price:,.0f}/kg. "
                f"Rata-rata: Rp {avg_price:,.0f}, Median: Rp {median_price:,.0f}. "
                f"Volatilitas: {volatility:.2f}% ({'TINGGI' if volatility > 30 else 'MODERATE' if volatility > 15 else 'RENDAH'}). "
                f"Std Dev: Rp {std_price:,.0f}."
            ),
            "metadata": {"source": "basic_stats", "commodity": commodity},
        })

        # --- STATISTIK PER BULAN (KOMPRASI) ---
        df_copy = commodity_df.copy()
        df_copy["month"] = df_copy["tanggal"].dt.to_period("M").astype(str)
        monthly = df_copy.groupby("month")["harga"].agg(
            avg_price="mean", min_price="min", max_price="max", record_count="count"
        ).reset_index()
        
        if len(monthly) > 0:
            best_month = monthly.loc[monthly["avg_price"].idxmin()]
            worst_month = monthly.loc[monthly["avg_price"].idxmax()]
            
            documents.append({
                "type": "monthly_summary",
                "title": f"Ringkasan Bulanan {short_name}",
                "content": (
                    f"Statistik bulanan {short_name}: {len(monthly)} bulan tercatat. "
                    f"Bulan termurah: Rp {best_month['avg_price']:,.0f}/kg. "
                    f"Bulan termahal: Rp {worst_month['avg_price']:,.0f}/kg. "
                    f"Record total: {int(best_month['record_count'] + worst_month['record_count'])} hari."
                ),
                "metadata": {"type": "monthly_summary", "commodity": commodity},
            })

        # --- HARGA EKSTREM (DIGABUNG) ---
        top5 = commodity_df.nlargest(3, "harga")[["tanggal", "harga"]]
        bottom3 = commodity_df.nsmallest(3, "harga")[["tanggal", "harga"]]
        
        documents.append({
            "type": "extreme_values",
            "title": f"Harga Ekstrem {short_name}",
            "content": (
                f"Harga {short_name} tertinggi: Rp {top5.iloc[0]['harga']:,.0f}/kg. "
                f"Terendah: Rp {bottom3.iloc[0]['harga']:,.0f}/kg. "
                f"Spread: Rp {top5.iloc[0]['harga'] - bottom3.iloc[0]['harga']:,.0f}/kg."
            ),
            "metadata": {"type": "extreme_values", "commodity": commodity},
        })

        # --- TREND OVERALL ---
        first_half = df_copy[df_copy["tanggal"] < df_copy["tanggal"].median()]["harga"].mean()
        second_half = df_copy[df_copy["tanggal"] >= df_copy["tanggal"].median()]["harga"].mean()
        overall_trend_pct = ((second_half - first_half) / first_half) * 100 if first_half > 0 else 0

        documents.append({
            "type": "trend",
            "title": f"Trend Harga {short_name}",
            "content": (
                f"Trend {short_name}: {'NAIK' if overall_trend_pct > 0 else 'TURUN'} {abs(overall_trend_pct):.2f}%. "
                f"Paruh 1: Rp {first_half:,.0f}/kg → Paruh 2: Rp {second_half:,.0f}/kg."
            ),
            "metadata": {"type": "trend", "commodity": commodity},
        })

        # --- INSIGHT BISNIS (DIGABUNG SEMUA DIVISI) ---
        documents.append({
            "type": "business_insights",
            "title": f"Insight Bisnis {short_name}",
            "content": (
                f"Rekomendasi bisnis {short_name}:\n"
                f"- Procurement: Beli saat harga < Rp {avg_price * 0.85:,.0f}\n"
                f"- Sales: Jual saat harga > Rp {avg_price * 1.15:,.0f}\n"
                f"- Finance: Budget Rp {avg_price:,.0f}/kg ± Rp {std_price:,.0f}\n"
                f"- Risk: Warning jika harga > Rp {avg_price * 1.3:,.0f}"
            ),
            "metadata": {"type": "business_insights", "commodity": commodity},
        })

        # --- DATA REFERENSI (FIRST & LAST ONLY) ---
        if len(commodity_df) > 0:
            first_row = commodity_df.iloc[0]
            last_row = commodity_df.iloc[-1]
            documents.append({
                "type": "reference",
                "title": f"Data Referensi {short_name}",
                "content": (
                    f"Data referensi {short_name}: "
                    f"Awal periode {first_row['tanggal'].strftime('%d %B %Y')}: Rp {first_row['harga']:,.0f}/kg. "
                    f"Akhir periode {last_row['tanggal'].strftime('%d %B %Y')}: Rp {last_row['harga']:,.0f}/kg."
                ),
                "metadata": {"type": "reference", "date_range": f"{first_row['tanggal'].date()} to {last_row['tanggal'].date()}", "commodity": commodity},
            })

    # === COMPARISON GLOBAL ===
    avg_per_commodity = df.groupby("nama_komoditas")["harga"].mean()
    comparison_str = ", ".join([f"{c.split(',')[0]}: Rp {v:,.0f}" for c, v in avg_per_commodity.items()])
    
    documents.append({
        "type": "comparison",
        "title": "Perbandingan Semua Komoditas",
        "content": (
            f"Perbandingan rata-rata harga:\n{comparison_str}\n\n"
            f"Termahal: {avg_per_commodity.idxmax().split(',')[0]} (Rp {avg_per_commodity.max():,.0f}/kg)\n"
            f"Termurah: {avg_per_commodity.idxmin().split(',')[0]} (Rp {avg_per_commodity.min():,.0f}/kg)"
        ),
        "metadata": {"type": "comparison_global"},
    })

    print(f"  ✅ Generated {len(documents)} dokumen dalam {time.time() - start_time:.2f}s")
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

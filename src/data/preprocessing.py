"""Module untuk preprocessing dan feature engineering data."""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Bersihkan data dari missing values dan outlier.

    Args:
        df: DataFrame yang akan dibersihkan.

    Returns:
        DataFrame yang sudah dibersihkan.
    """
    cleaned_df = df.copy()

    # Hapus baris dengan harga NaN
    cleaned_df = cleaned_df.dropna(subset=["harga"])

    # Hapus baris dengan harga <= 0
    cleaned_df = cleaned_df[cleaned_df["harga"] > 0]

    return cleaned_df.reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan fitur baru untuk analisis - PER KOMODITAS.

    Fitur rolling window dihitung terpisah per komoditas agar tidak tercampur.

    Args:
        df: DataFrame yang sudah dibersihkan.

    Returns:
        DataFrame dengan fitur tambahan.
    """
    featured_dfs = []

    # Group by komoditas untuk memastikan rolling window tidak tercampur
    for commodity, group in df.groupby("nama_komoditas"):
        g = group.sort_values("tanggal").copy()

        # Ekstrak fitur waktu
        g["year"] = g["tanggal"].dt.year
        g["month"] = g["tanggal"].dt.month
        g["day_of_week"] = g["tanggal"].dt.dayofweek
        g["day_of_month"] = g["tanggal"].dt.day

        # Rolling average (7 hari dan 30 hari) - PER KOMODITAS
        g["rolling_avg_7d"] = g["harga"].rolling(window=7, min_periods=1).mean()
        g["rolling_avg_30d"] = g["harga"].rolling(window=30, min_periods=1).mean()

        # Perubahan harga harian (%)
        g["price_change"] = g["harga"].pct_change().fillna(0)

        # Perubahan harga mingguan (%)
        g["weekly_change"] = g["harga"].pct_change(periods=7).fillna(0)

        featured_dfs.append(g)

    # Gabungkan semua komoditas kembali
    return pd.concat(featured_dfs, ignore_index=True)


def prepare_training_data(
    df: pd.DataFrame,
    target_col: str = "harga",
    lookback_days: int = 7,
) -> tuple[pd.DataFrame, pd.Series]:
    """Siapkan data untuk training model prediksi - PER KOMODITAS.

    Args:
        df: DataFrame dengan fitur tambahan.
        target_col: Nama kolom target.
        lookback_days: Jumlah hari sebelumnya sebagai fitur.

    Returns:
        Tuple (X, y) untuk training.
    """
    featured_dfs = []

    # Proses per komoditas
    for commodity, group in df.groupby("nama_komoditas"):
        g = group.copy()

        # Buat fitur lag per komoditas
        for i in range(1, lookback_days + 1):
            g[f"lag_{i}d"] = g[target_col].shift(i)

        featured_dfs.append(g)

    featured_df = pd.concat(featured_dfs, ignore_index=True)

    # Drop rows yang memiliki NaN karena lag
    featured_df = featured_df.dropna()

    feature_cols = [
        "month", "day_of_week", "rolling_avg_7d", "rolling_avg_30d",
        "price_change", "weekly_change",
    ] + [f"lag_{i}d" for i in range(1, lookback_days + 1)]

    X = featured_df[feature_cols]
    y = featured_df[target_col]

    return X, y


def get_price_statistics(df: pd.DataFrame) -> dict:
    """Hitung statistik harga per bulan - PER KOMODITAS.

    Args:
        df: DataFrame berisi data harga.

    Returns:
        Dictionary berisi statistik per bulan per komoditas.
    """
    result = {}

    for commodity, group in df.groupby("nama_komoditas"):
        g = group.copy()
        g["month"] = g["tanggal"].dt.to_period("M")

        monthly_stats = g.groupby("month")["harga"].agg(["min", "max", "mean", "std"]).reset_index()

        result[commodity] = {}
        for _, row in monthly_stats.iterrows():
            month_key = str(row["month"])
            result[commodity][month_key] = {
                "min": float(row["min"]),
                "max": float(row["max"]),
                "mean": float(row["mean"]),
                "std": float(row["std"]),
            }

    return result


def get_commodity_stats(df: pd.DataFrame) -> Dict[str, Dict]:
    """Hitung statistik ringkas per komoditas.

    Args:
        df: DataFrame berisi data harga.

    Returns:
        Dictionary berisi statistik per komoditas.
        Contoh: {"Cabai Merah Besar,1 kg": {"min": ..., "max": ..., "mean": ..., "std": ...}, ...}
    """
    result = {}

    for commodity, group in df.groupby("nama_komoditas"):
        result[commodity] = {
            "min": float(group["harga"].min()),
            "max": float(group["harga"].max()),
            "mean": float(group["harga"].mean()),
            "median": float(group["harga"].median()),
            "std": float(group["harga"].std()),
            "record_count": len(group),
            "date_range": (
                str(group["tanggal"].min()),
                str(group["tanggal"].max()),
            ),
        }

    return result

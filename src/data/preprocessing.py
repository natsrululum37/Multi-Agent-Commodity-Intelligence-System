"""Module untuk preprocessing dan feature engineering data."""

import pandas as pd
import numpy as np
from typing import Optional


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
    """Tambahkan fitur baru untuk analisis.

    Args:
        df: DataFrame yang sudah dibersihkan.

    Returns:
        DataFrame dengan fitur tambahan.
    """
    featured_df = df.copy()

    # Ekstrak fitur waktu
    featured_df["year"] = featured_df["tanggal"].dt.year
    featured_df["month"] = featured_df["tanggal"].dt.month
    featured_df["day_of_week"] = featured_df["tanggal"].dt.dayofweek
    featured_df["day_of_month"] = featured_df["tanggal"].dt.day

    # Rolling average (7 hari dan 30 hari)
    featured_df["rolling_avg_7d"] = featured_df["harga"].rolling(window=7, min_periods=1).mean()
    featured_df["rolling_avg_30d"] = featured_df["harga"].rolling(window=30, min_periods=1).mean()

    # Perubahan harga harian (%)
    featured_df["price_change"] = featured_df["harga"].pct_change().fillna(0)

    # Perubahan harga mingguan (%)
    featured_df["weekly_change"] = featured_df["harga"].pct_change(periods=7).fillna(0)

    return featured_df


def prepare_training_data(
    df: pd.DataFrame,
    target_col: str = "harga",
    lookback_days: int = 7,
) -> tuple[pd.DataFrame, pd.Series]:
    """Siapkan data untuk training model prediksi.

    Args:
        df: DataFrame dengan fitur tambahan.
        target_col: Nama kolom target.
        lookback_days: Jumlah hari sebelumnya sebagai fitur.

    Returns:
        Tuple (X, y) untuk training.
    """
    featured_df = df.copy()

    # Buat fitur lag
    for i in range(1, lookback_days + 1):
        featured_df[f"lag_{i}d"] = featured_df[target_col].shift(i)

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
    """Hitung statistik harga per bulan.

    Args:
        df: DataFrame berisi data harga.

    Returns:
        Dictionary berisi statistik per bulan.
    """
    df_copy = df.copy()
    df_copy["month"] = df_copy["tanggal"].dt.to_period("M")

    monthly_stats = df_copy.groupby("month")["harga"].agg(["min", "max", "mean", "std"]).reset_index()

    result = {}
    for _, row in monthly_stats.iterrows():
        month_key = str(row["month"])
        result[month_key] = {
            "min": float(row["min"]),
            "max": float(row["max"]),
            "mean": float(row["mean"]),
            "std": float(row["std"]),
        }

    return result

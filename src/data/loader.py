"""Module untuk memuat dan memvalidasi data CSV."""

import pandas as pd
from typing import Tuple


def load_data(filepath: str = "cabai.csv") -> pd.DataFrame:
    """Muat data harga komoditas dari file CSV.

    Args:
        filepath: Path ke file CSV. Default: 'cabai.csv'

    Returns:
        DataFrame yang berisi data harga komoditas.

    Raises:
        FileNotFoundError: Jika file CSV tidak ditemukan.
        ValueError: Jika kolom yang dibutuhkan tidak ada.
    """
    df = pd.read_csv(filepath)

    # Validasi kolom yang dibutuhkan
    required_columns = ["tanggal", "nama_komoditas", "harga"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Kolom yang hilang: {missing_columns}")

    # Konversi tipe data
    df["tanggal"] = pd.to_datetime(df["tanggal"])
    df["harga"] = pd.to_numeric(df["harga"], errors="coerce")

    return df


def get_summary(df: pd.DataFrame) -> dict:
    """Buat ringkasan statistik dari dataset.

    Args:
        df: DataFrame berisi data harga komoditas.

    Returns:
        Dictionary berisi statistik deskriptif.
    """
    return {
        "total_records": len(df),
        "date_range": (str(df["tanggal"].min()), str(df["tanggal"].max())),
        "unique_commodities": df["nama_komoditas"].nunique(),
        "unique_markets": df["nama_pasar"].nunique() if "nama_pasar" in df.columns else 0,
        "price_stats": df["harga"].describe().to_dict(),
    }


def split_time_series(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data berdasarkan waktu untuk train/test.

    Args:
        df: DataFrame berisi data harga komoditas.
        test_size: Proporsi data untuk testing.

    Returns:
        Tuple (train_df, test_df)
    """
    sorted_df = df.sort_values("tanggal")
    split_idx = int(len(sorted_df) * (1 - test_size))
    return sorted_df.iloc[:split_idx], sorted_df.iloc[split_idx:]

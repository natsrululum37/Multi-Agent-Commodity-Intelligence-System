"""Tests untuk data loading dan preprocessing."""

import pytest
import pandas as pd
import sys
from pathlib import Path

# Tambahkan root directory ke Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_data, get_summary, split_time_series
from src.data.preprocessing import clean_data, add_features, prepare_training_data


@pytest.fixture
def sample_df():
    """Sample DataFrame untuk testing."""
    return pd.DataFrame({
        "tanggal": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
        "harga": [80000, 85000, 90000, 82000, 88000],
        "nama_komoditas": ["Cabai Merah"] * 5,
        "nama_pasar": ["Pasar Beringharjo"] * 5,
    })


def test_load_data():
    """Test load_data function."""
    df = load_data("cabai.csv")
    assert len(df) > 0
    assert "tanggal" in df.columns
    assert "harga" in df.columns


def test_get_summary(sample_df):
    """Test get_summary function."""
    summary = get_summary(sample_df)
    assert summary["total_records"] == 5
    assert summary["unique_commodities"] == 1


def test_split_time_series(sample_df):
    """Test time series split."""
    train, test = split_time_series(sample_df, test_size=0.2)
    assert len(train) + len(test) == len(sample_df)
    assert len(train) > 0
    assert len(test) > 0


def test_clean_data(sample_df):
    """Test data cleaning."""
    cleaned = clean_data(sample_df)
    assert len(cleaned) > 0
    assert cleaned["harga"].min() > 0


def test_add_features(sample_df):
    """Test feature engineering."""
    featured = add_features(sample_df)
    assert "rolling_avg_7d" in featured.columns
    assert "price_change" in featured.columns


def test_prepare_training_data(sample_df):
    """Test training data preparation."""
    featured = add_features(sample_df)
    X, y = prepare_training_data(featured, lookback_days=3)
    assert len(X) > 0
    assert len(y) == len(X)

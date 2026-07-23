"""Tests untuk model prediksi dan evaluator."""

import pytest
import numpy as np
import sys
from pathlib import Path

# Tambahkan root directory ke Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.predictor import PricePredictor, PriceTrendPredictor
from src.models.evaluator import (
    evaluate_regression_model,
    evaluate_response_effectiveness,
    measure_efficiency,
)


def test_price_predictor_training():
    """Test training PricePredictor."""
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    predictor = PricePredictor(model_type="linear")
    predictor.train(X, y)
    assert predictor is not None


def test_price_predictor_prediction():
    """Test prediction."""
    X_train = np.random.rand(100, 5)
    y_train = np.random.rand(100)
    X_test = np.random.rand(10, 5)

    predictor = PricePredictor(model_type="linear")
    predictor.train(X_train, y_train)
    predictions = predictor.predict(X_test)

    assert len(predictions) == 10


def test_price_trend_predictor():
    """Test trend analysis."""
    predictor = PriceTrendPredictor()
    prices = [80000 + i * 1000 for i in range(30)]
    predictor.add_prices(prices)
    trend = predictor.get_trend(days=7)
    assert "trend" in trend
    assert "change_pct" in trend


def test_evaluate_regression_model():
    """Test regression evaluation."""
    y_true = np.array([100, 200, 300, 400, 500])
    y_pred = np.array([95, 195, 305, 395, 495])
    metrics = evaluate_regression_model(y_true, y_pred)
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert "mape" in metrics


def test_evaluate_response_effectiveness():
    """Test response effectiveness evaluation."""
    question = "Berapa harga?"
    response = "Harga cabai merah rata-rata Rp 50000/kg"
    context = "Harga cabai merah rata-rata Rp 50000/kg adalah rata-rata harga harian"
    result = evaluate_response_effectiveness(question, response, context)
    assert "effectiveness" in result


def test_measure_efficiency():
    """Test efficiency measurement."""
    import time
    start = time.time()
    time.sleep(0.01)
    end = time.time()
    result = measure_efficiency(start, end)
    assert "duration_seconds" in result
    assert result["duration_seconds"] > 0

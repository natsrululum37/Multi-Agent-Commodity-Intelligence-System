"""Module untuk evaluasi model."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict


def evaluate_regression_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Evaluasi model regresi.

    Args:
        y_true: Nilai aktual.
        y_pred: Nilai prediksi.

    Returns:
        Dictionary berisi metrik evaluasi.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # MAPE (Mean Absolute Percentage Error)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4),
        "mape": round(mape, 2),
    }


def evaluate_classification_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Evaluasi model klasifikasi.

    Args:
        y_true: Label aktual.
        y_pred: Label prediksi.

    Returns:
        Dictionary berisi metrik evaluasi.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
    }


def evaluate_response_effectiveness(question: str, response: str, context: str) -> Dict:
    """Evaluasi kualitas respons RAG.

    Args:
        question: Pertanyaan user.
        response: Respons yang dihasilkan agent.
        context: Konteks yang digunakan.

    Returns:
        Dictionary berisi skor efektivitas.
    """
    if not response or not context:
        return {"effectiveness": 0.0, "context_used": False}

    # Cek apakah respons menggunakan konteks
    context_used = any(word.lower() in response.lower() for word in context.split()[:20] if len(word) > 3)

    # Hitung panjang respons dan konteks
    return {
        "effectiveness": 1.0 if context_used else 0.5,
        "response_length": len(response),
        "context_used": context_used,
    }


def measure_efficiency(start_time: float, end_time: float) -> Dict:
    """Ukur efisiensi waktu respons.

    Args:
        start_time: Timestamp awal.
        end_time: Timestamp akhir.

    Returns:
        Dictionary berisi metrik efisiensi.
    """
    duration = end_time - start_time
    return {
        "duration_seconds": round(duration, 4),
        "is_fast": duration < 5.0,
        "is_acceptable": duration < 10.0,
    }


def calculate_hallucination_rate(predictions: list, actuals: list, context: list) -> Dict:
    """Hitung tingkat halusinasi.

    Args:
        predictions: List prediksi/respons.
        actuals: List data aktual.
        context: List konteks yang digunakan.

    Returns:
        Dictionary berisi metrik halusinasi.
    """
    if not predictions or not actuals:
        return {"hallucination_rate": 0.0, "total": 0}

    hallucination_count = 0
    total = len(predictions)

    for pred, ctx in zip(predictions, context):
        # Respons dianggap halusinasi jika tidak ada kata dari konteks yang muncul
        if ctx and not any(word.lower() in pred.lower() for word in ctx.split()[:30] if len(word) > 3):
            hallucination_count += 1

    rate = (hallucination_count / total) * 100
    return {
        "hallucination_rate": round(rate, 2),
        "total": total,
        "hallucinated_count": hallucination_count,
    }

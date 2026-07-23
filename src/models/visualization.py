"""Module untuk membuat visualisasi data."""

import matplotlib
matplotlib.use("Agg")  # Non-GUI backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


def plot_price_history(df: pd.DataFrame, output_path: str | Path) -> None:
    """Visualisasi historis harga cabai merah.

    Args:
        df: DataFrame berisi kolom 'tanggal' dan 'harga'.
        output_path: Path file output.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Analisis Harga Cabai Merah - Pasar Beringharjo", fontsize=14, fontweight="bold")

    # Plot 1: Time Series dengan Rolling Average
    ax1 = axes[0, 0]
    df_sorted = df.sort_values("tanggal").copy()
    df_sorted["rolling_7d"] = df_sorted["harga"].rolling(window=7, min_periods=1).mean()
    df_sorted["rolling_30d"] = df_sorted["harga"].rolling(window=30, min_periods=1).mean()
    ax1.plot(df_sorted["tanggal"], df_sorted["harga"], alpha=0.4, label="Harga Harian", color="#e74c3c")
    ax1.plot(df_sorted["tanggal"], df_sorted["rolling_7d"], label="Rata-rata 7 Hari", linewidth=2, color="#f39c12")
    ax1.plot(df_sorted["tanggal"], df_sorted["rolling_30d"], label="Rata-rata 30 Hari", linewidth=2, color="#2ecc71")
    ax1.set_title("Trend Harga Harian")
    ax1.set_xlabel("Tanggal")
    ax1.set_ylabel("Harga (Rp)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    # Plot 2: Distribusi Harga
    ax2 = axes[0, 1]
    ax2.hist(df["harga"], bins=50, edgecolor="black", alpha=0.7, color="#e74c3c")
    ax2.axvline(df["harga"].mean(), color="orange", linestyle="--", label=f"Mean: Rp {df['harga'].mean():,.0f}")
    ax2.axvline(df["harga"].median(), color="green", linestyle="--", label=f"Median: Rp {df['harga'].median():,.0f}")
    ax2.set_title("Distribusi Harga")
    ax2.set_xlabel("Harga (Rp/kg)")
    ax2.set_ylabel("Frekuensi")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Harga per Bulan
    ax3 = axes[1, 0]
    df_copy = df.copy()
    df_copy["month"] = df_copy["tanggal"].dt.to_period("M").astype(str)
    monthly_avg = df_copy.groupby("month")["harga"].agg(["mean", "min", "max"])
    x_pos = range(len(monthly_avg))
    ax3.bar(x_pos, monthly_avg["mean"], yerr=[monthly_avg["mean"] - monthly_avg["min"], monthly_avg["max"] - monthly_avg["mean"]],
            capsize=3, color="#3498db", alpha=0.7, edgecolor="black")
    ax3.set_xticks(list(x_pos))
    ax3.set_xticklabels(monthly_avg.index, rotation=45, ha="right", fontsize=8)
    ax3.set_title("Harga Rata-rata per Bulan")
    ax3.set_xlabel("Bulan")
    ax3.set_ylabel("Harga (Rp)")
    ax3.grid(True, alpha=0.3, axis="y")

    # Plot 4: Perubahan Harga Harian
    ax4 = axes[1, 1]
    df_sorted["daily_change"] = df_sorted["harga"].pct_change() * 100
    colors = ["#2ecc71" if c > 0 else "#e74c3c" for c in df_sorted["daily_change"]]
    ax4.bar(range(len(df_sorted["daily_change"])), df_sorted["daily_change"], color=colors, alpha=0.6, edgecolor="none")
    ax4.axhline(0, color="black", linewidth=0.5)
    ax4.set_title("Perubahan Harga Harian (%)")
    ax4.set_xlabel("Hari ke-")
    ax4.set_ylabel("Perubahan (%)")
    ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Visualisasi disimpan ke: {output_path}")


def plot_comparison(actual: list, predicted: list, output_path: str | Path) -> None:
    """Visualisasi perbandingan actual vs predicted.

    Args:
        actual: List harga aktual.
        predicted: List harga prediksi.
        output_path: Path file output.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(min(len(actual), len(predicted)))
    ax.plot(x, actual[:len(x)], label="Actual", marker="o", markersize=3, linewidth=2, color="#e74c3c")
    ax.plot(x, predicted[:len(x)], label="Predicted", marker="x", markersize=4, linewidth=2, color="#3498db")
    ax.set_title("Perbandingan Actual vs Predicted Price")
    ax.set_xlabel("Data Point")
    ax.set_ylabel("Harga (Rp)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Perbandingan disimpan ke: {output_path}")


def plot_evaluation_metrics(metrics: dict, output_path: str | Path) -> None:
    """Visualisasi metrik evaluasi.

    Args:
        metrics: Dictionary berisi hasil evaluasi.
        output_path: Path file output.
    """
    categories = [m.get("metric", "").split(" - ")[0] if " - " in m.get("metric", "") else m.get("metric", "")
                  for m in metrics]
    scores = [m.get("score", 0) for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2ecc71" if s >= 0.7 else "#f39c12" if s >= 0.5 else "#e74c3c" for s in scores]
    bars = ax.barh(categories, scores, color=colors, edgecolor="black")

    # Tambahkan nilai di setiap bar
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f"{score:.2%}", va="center", fontsize=10, fontweight="bold")

    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Score")
    ax.set_title("Evaluation Metrics Score")
    ax.axvline(0.7, color="green", linestyle="--", alpha=0.5, label="Target (70%)")
    ax.legend()
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Metrik evaluasi disimpan ke: {output_path}")

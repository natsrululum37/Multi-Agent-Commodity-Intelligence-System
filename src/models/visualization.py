"""Module untuk membuat visualisasi data."""

import matplotlib
matplotlib.use("Agg")  # Non-GUI backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List


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
    print(f"  ✅ Visualisasi disimpan ke: {output_path}")


def plot_price_comparison(df: pd.DataFrame, output_path: str | Path) -> None:
    """Visualisasi perbandingan harga semua komoditas dalam 1 grafik.
    
    Args:
        df: DataFrame berisi kolom 'tanggal', 'nama_komoditas', dan 'harga'.
        output_path: Path file output.
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Perbandingan Harga Komoditas Cabai - Pasar Beringharjo", fontsize=16, fontweight="bold")
    
    commodities = df['nama_komoditas'].unique()
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    
    # === Plot 1: Line Chart Perbandingan Semua Komoditas ===
    ax1 = axes[0, 0]
    for i, commodity in enumerate(commodities):
        sub_df = df[df['nama_komoditas'] == commodity].sort_values('tanggal')
        label_short = commodity.split(',')[0]  # Ambil nama tanpa ",1 kg"
        ax1.plot(sub_df['tanggal'], sub_df['harga'], alpha=0.6, label=label_short, 
                color=colors[i % len(colors)], linewidth=1.5)
    ax1.set_title("Perbandingan Trend Harga Harian")
    ax1.set_xlabel("Tanggal")
    ax1.set_ylabel("Harga (Rp)")
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    
    # === Plot 2: Bar Chart Perbandingan Rata-rata Harga ===
    ax2 = axes[0, 1]
    avg_prices = df.groupby('nama_komoditas')['harga'].mean().sort_values()
    short_names = [c.split(',')[0] for c in avg_prices.index]
    bars = ax2.barh(range(len(avg_prices)), avg_prices.values, color=colors[:len(avg_prices)])
    ax2.set_yticks(range(len(avg_prices)))
    ax2.set_yticklabels(short_names)
    ax2.set_title("Rata-rata Harga per Komoditas")
    ax2.set_xlabel("Harga Rata-rata (Rp/kg)")
    # Tambahkan nilai di setiap bar
    for bar, val in zip(bars, avg_prices.values):
        ax2.text(bar.get_width() + 500, bar.get_y() + bar.get_height()/2, 
                f"Rp {val:,.0f}", va='center', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # === Plot 3: Box Plot Distribusi Harga ===
    ax3 = axes[1, 0]
    box_data = [df[df['nama_komoditas'] == c]['harga'].values for c in commodities]
    short_labels = [c.split(',')[0] for c in commodities]
    bp = ax3.boxplot(box_data, tick_labels=short_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors[:len(commodities)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax3.set_title("Distribusi Harga per Komoditas")
    ax3.set_xlabel("Komoditas")
    ax3.set_ylabel("Harga (Rp/kg)")
    ax3.tick_params(axis='x', rotation=15)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # === Plot 4: Statistik Ringkas per Komoditas ===
    ax4 = axes[1, 1]
    stats_df = df.groupby('nama_komoditas')['harga'].agg(['min', 'max', 'mean', 'std']).reset_index()
    stats_df['short_name'] = stats_df['nama_komoditas'].apply(lambda x: x.split(',')[0])
    
    x_pos = range(len(stats_df))
    width = 0.2
    ax4.bar([p - width*1.5 for p in x_pos], stats_df['min'], width, label='Min', color='#95a5a6')
    ax4.bar([p - width/2 for p in x_pos], stats_df['mean'], width, label='Mean', color='#3498db')
    ax4.bar([p + width/2 for p in x_pos], stats_df['max'], width, label='Max', color='#e74c3c')
    ax4.bar([p + width*1.5 for p in x_pos], stats_df['std'], width, label='Std Dev', color='#f39c12')
    ax4.set_xticks(list(x_pos))
    ax4.set_xticklabels(stats_df['short_name'], rotation=15, ha='right')
    ax4.set_title("Ringkasan Statistik Harga")
    ax4.set_ylabel("Harga (Rp)")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Perbandingan harga disimpan ke: {output_path}")


def plot_per_commodity(df: pd.DataFrame, commodity_name: str, output_path: str | Path) -> None:
    """Visualisasi detail harga per komoditas tertentu.
    
    Args:
        df: DataFrame berisi data harga.
        commodity_name: Nama komoditas (contoh: "Cabai Merah Besar,1 kg").
        output_path: Path file output.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Analisis Detail {commodity_name} - Pasar Beringharjo", fontsize=14, fontweight="bold")
    
    # Filter data komoditas
    sub_df = df[df['nama_komoditas'] == commodity_name].sort_values('tanggal').copy()
    
    if len(sub_df) == 0:
        print(f"  ⚠️  Data tidak ditemukan untuk {commodity_name}")
        return
    
    # Plot 1: Time Series dengan Rolling Average
    ax1 = axes[0, 0]
    sub_df["rolling_7d"] = sub_df["harga"].rolling(window=7, min_periods=1).mean()
    sub_df["rolling_30d"] = sub_df["harga"].rolling(window=30, min_periods=1).mean()
    ax1.plot(sub_df["tanggal"], sub_df["harga"], alpha=0.4, label="Harga Harian", color="#e74c3c")
    ax1.plot(sub_df["tanggal"], sub_df["rolling_7d"], label="Rata-rata 7 Hari", linewidth=2, color="#f39c12")
    ax1.plot(sub_df["tanggal"], sub_df["rolling_30d"], label="Rata-rata 30 Hari", linewidth=2, color="#2ecc71")
    ax1.set_title("Trend Harga Harian")
    ax1.set_xlabel("Tanggal")
    ax1.set_ylabel("Harga (Rp)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    
    # Plot 2: Distribusi Harga
    ax2 = axes[0, 1]
    ax2.hist(sub_df["harga"], bins=30, edgecolor="black", alpha=0.7, color="#e74c3c")
    ax2.axvline(sub_df["harga"].mean(), color="orange", linestyle="--", 
               label=f"Mean: Rp {sub_df['harga'].mean():,.0f}")
    ax2.axvline(sub_df["harga"].median(), color="green", linestyle="--", 
               label=f"Median: Rp {sub_df['harga'].median():,.0f}")
    ax2.set_title("Distribusi Harga")
    ax2.set_xlabel("Harga (Rp/kg)")
    ax2.set_ylabel("Frekuensi")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Harga per Bulan
    ax3 = axes[1, 0]
    sub_df_copy = sub_df.copy()
    sub_df_copy["month"] = sub_df_copy["tanggal"].dt.to_period("M").astype(str)
    monthly_avg = sub_df_copy.groupby("month")["harga"].agg(["mean", "min", "max"])
    x_pos = range(len(monthly_avg))
    ax3.bar(x_pos, monthly_avg["mean"], 
           yerr=[monthly_avg["mean"] - monthly_avg["min"], monthly_avg["max"] - monthly_avg["mean"]],
           capsize=3, color="#3498db", alpha=0.7, edgecolor="black")
    ax3.set_xticks(list(x_pos))
    ax3.set_xticklabels(monthly_avg.index, rotation=45, ha="right", fontsize=8)
    ax3.set_title("Harga Rata-rata per Bulan")
    ax3.set_xlabel("Bulan")
    ax3.set_ylabel("Harga (Rp)")
    ax3.grid(True, alpha=0.3, axis="y")
    
    # Plot 4: Perubahan Harga Harian
    ax4 = axes[1, 1]
    sub_df["daily_change"] = sub_df["harga"].pct_change() * 100
    colors = ["#2ecc71" if c > 0 else "#e74c3c" for c in sub_df["daily_change"]]
    ax4.bar(range(len(sub_df["daily_change"])), sub_df["daily_change"], color=colors, alpha=0.6, edgecolor="none")
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
    print(f"  ✅ Visualisasi {commodity_name} disimpan ke: {output_path}")


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

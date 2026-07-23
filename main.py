"""Main entry point untuk Multi-Agent Commodity Intelligence System."""

import sys
import json
from pathlib import Path

# Tambahkan root directory ke Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.agents.coordinator import CoordinatorAgent
from src.config import Config


def run_demo():
    """Jalankan demo sistem."""
    print("=" * 70)
    print("Multi-Agent Commodity Intelligence System")
    print("UAS Data Mining ST167 - Universitas AMIKOM Yogyakarta")
    print("=" * 70)

    # Pastikan direktori ada
    Config.ensure_directories()

    # Inisialisasi coordinator
    coord = CoordinatorAgent()

    # Jalankan analisis penuh
    results = coord.run_full_analysis("cabai.csv")

    # Generate visualisasi
    print("\n📊 Generating visualizations...")
    from src.models.visualization import plot_price_history, plot_price_comparison, plot_per_commodity, plot_comparison, plot_evaluation_metrics
    
    try:
        # 1. Visualisasi perbandingan semua komoditas
        plot_price_comparison(coord.data_agent.data, Config.REPORTS_PATH / "price_comparison.png")
        
        # 2. Visualisasi detail per komoditas
        commodities = coord.data_agent.data['nama_komoditas'].unique()
        for commodity in commodities:
            short_name = commodity.split(',')[0]  # Hapus ",1 kg"
            filename = f"price_{short_name.lower().replace(' ', '_')}.png"
            plot_per_commodity(coord.data_agent.data, commodity, Config.REPORTS_PATH / filename)
            
    except Exception as e:
        print(f"  ⚠️  Warning: Gagal membuat visualisasi: {e}")

    # Tampilkan hasil
    print("\n" + "=" * 70)
    print("HASIL ANALISIS")
    print("=" * 70)

    # Data Analysis
    analysis = results.get("data_analysis", {})
    print("\n📊 DATA ANALYSIS:")
    print(f"  Total records: {analysis.get('total_records', 'N/A')}")
    print(f"  Unique commodities: {analysis.get('unique_commodities', 'N/A')}")
    date_range = analysis.get('date_range', ('N/A', 'N/A'))
    if isinstance(date_range, (list, tuple)):
        print(f"  Date range: {date_range[0]} → {date_range[1]}")
    else:
        print(f"  Date range: {date_range}")
    
    # Tampilkan statistik PER KOMODITAS
    commodity_stats = analysis.get("commodity_stats", {})
    if commodity_stats:
        print("\n  📈 Statistik per Komoditas:")
        for name, stats in commodity_stats.items():
            short_name = name.split(',')[0]
            print(f"    • {short_name}:")
            print(f"      Avg: Rp {stats['mean']:,.0f} | Min: Rp {stats['min']:,.0f} | Max: Rp {stats['max']:,.0f}")
            print(f"      Volatility: {analysis.get('volatility_per_commodity', {}).get(name, 0):.2f}%")

    # Insights
    insights = results.get("data_insights", [])
    print("\n💡 INSIGHTS:")
    if insights:
        for i, insight in enumerate(insights, 1):
            if insight.strip():  # Skip empty lines
                print(f"  {i}. {insight}")
    else:
        print("  (tidak ada insights)")

    # Predictions PER KOMODITAS
    predictions = results.get("predictions", {})
    print("\n🔮 PREDICTIONS PER KOMODITAS:")
    if predictions:
        for commodity, trend in predictions.items():
            short_name = commodity.split(",")[0]
            print(f"\n  📈 {short_name}:")
            print(f"     Consensus Trend: {trend.get('consensus_trend', 'unknown').upper()}")
            method_results = trend.get("method_results", {})
            if method_results:
                ma = method_results.get("moving_average", {})
                print(f"     Moving Average: {ma.get('change_pct', 0):+.2f}% ({ma.get('trend', '-')})")
                rg = method_results.get("regression", {})
                print(f"     Regression Slope: {rg.get('slope', 0):+.0f} ({rg.get('trend', '-')})")
                mm = method_results.get("momentum", {})
                print(f"     7-day Momentum: {mm.get('7d_change_pct', 0):+.2f}% ({mm.get('trend', '-')})")
            next_price = trend.get("next_predicted_price")
            if next_price:
                print(f"     Next predicted price: Rp {next_price:,.0f}")
            if trend.get("current_price"):
                print(f"     Current price: Rp {trend['current_price']:,.0f}")
    else:
        print("  (tidak ada prediksi)")

    # Model Metrics PER KOMODITAS
    model_metrics = results.get("model_metrics", {})
    if model_metrics:
        print("\n📊 MODEL PERFORMANCE (ML Training) PER KOMODITAS:")
        for commodity, metrics in model_metrics.items():
            short_name = commodity.split(",")[0]
            print(f"\n  🤖 {short_name}:")
            print(f"     MAE: Rp {metrics.get('mae', 0):,.0f} | RMSE: Rp {metrics.get('rmse', 0):,.0f}")
            print(f"     R² Score: {metrics.get('r2', 0):.4f} | MAPE: {metrics.get('mape', 0):.2f}%")

    # Recommendations PER KOMODITAS
    all_recommendations = results.get("recommendations", {})
    if all_recommendations:
        print("\n📋 RECOMMENDATIONS PER KOMODITAS:")
        for commodity, recs in all_recommendations.items():
            short_name = commodity.split(",")[0]
            print(f"\n  📌 {short_name}:")
            if recs:
                for i, rec in enumerate(recs, 1):
                    print(f"     {i}. {rec}")
            else:
                print("     (tidak ada rekomendasi)")

    # RAG Responses
    rag_results = results.get("rag_responses", [])
    print("\n🤖 RAG RESPONSES:")
    if rag_results:
        for rag in rag_results:
            question = rag.get('question', '')
            # Tampilkan hanya jika pertanyaan mengandung komoditas spesifik
            for commodity in coord.data_agent.data["nama_komoditas"].unique():
                if commodity in question:
                    short_name = commodity.split(",")[0]
                    print(f"\n  📌 {short_name}:")
                    break
            
            answer = rag.get("answer", "")
            print(f"  Q: {rag.get('question', '')}")
            print(f"  A: {answer[:300]}{'...' if len(answer) > 300 else ''}")
            confidence = rag.get("confidence", 0)
            sources = rag.get("sources", [])
            if sources:
                unique_sources = list(dict.fromkeys(sources))[:3]
                source_str = ", ".join(unique_sources)
                print(f"  Sources: [{source_str}]")
            print(f"  Confidence: {confidence:.2%}")
    else:
        print("  (tidak ada jawaban)")

    # Evaluation
    evaluations = results.get("evaluations", [])
    avg_score = results.get("average_score", 0)
    weighted_score = results.get("weighted_score", 0)

    print("\n📈 EVALUATION METRICS:")
    if evaluations:
        for ev in evaluations:
            category = ev.get('category', 'unknown')
            score = ev.get('score', 0)
            metric = ev.get('metric', 'Unknown')
            icon = "✅" if score >= 0.7 else ("⚠️" if score >= 0.5 else "❌")
            print(f"  {icon} {metric}: {score:.2%}")
    else:
        print("  (tidak ada evaluasi)")
    
    # Generate evaluation metrics visualization
    if evaluations:
        try:
            plot_evaluation_metrics(evaluations, Config.REPORTS_PATH / "evaluation_metrics.png")
        except Exception as e:
            print(f"  ⚠️  Warning: Gagal membuat visualisasi evaluasi: {e}")

    print(f"\n  Average Score: {avg_score:.2%}")
    print(f"  Weighted Score: {weighted_score:.2%}")

    # Execution time
    exec_time = results.get("execution_time", 0)
    print(f"\n⏱️  Execution Time: {exec_time:.2f} seconds")

    print("\n" + "=" * 70)
    print("Demo selesai!")
    print("=" * 70)

    return results


def interactive_mode():
    """Mode interaktif untuk bertanya ke sistem."""
    coord = CoordinatorAgent()

    print("\n" + "=" * 70)
    print("Interactive Mode - Ketik 'quit' atau 'exit' untuk keluar")
    print("=" * 70)

    while True:
        question = input("\n🔍 Pertanyaan Anda: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Sampai jumpa!")
            break

        try:
            result = coord.ask_question(question)
            print(f"\n🤖 Jawaban:\n{result.get('answer', 'Tidak ada jawaban')[:800]}")
            print(f"\n📊 Confidence: {result.get('confidence', 0):.2%}")
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        run_demo()

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
    from src.models.visualization import plot_price_history, plot_comparison, plot_evaluation_metrics
    try:
        plot_price_history(coord.data_agent.data, Config.REPORTS_PATH / "price_history.png")
        if "evaluations" in results and results["evaluations"]:
            plot_evaluation_metrics(results["evaluations"], Config.REPORTS_PATH / "evaluation_metrics.png")
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
    date_range = analysis.get('date_range', ('N/A', 'N/A'))
    if isinstance(date_range, (list, tuple)):
        print(f"  Date range: {date_range[0]} → {date_range[1]}")
    else:
        print(f"  Date range: {date_range}")

    price_stats = analysis.get("price_stats", {})
    print(f"  Min price: Rp {price_stats.get('min', 0):,.0f}")
    print(f"  Max price: Rp {price_stats.get('max', 0):,.0f}")
    print(f"  Mean price: Rp {price_stats.get('mean', 0):,.0f}")
    print(f"  Volatility: {analysis.get('volatility', 0):.2f}%")

    # Insights
    insights = results.get("data_insights", [])
    print("\n💡 INSIGHTS:")
    if insights:
        for i, insight in enumerate(insights, 1):
            print(f"  {i}. {insight}")
    else:
        print("  (tidak ada insights)")

    # Predictions
    prediction = results.get("prediction", {})
    print("\n🔮 PREDICTIONS:")
    print(f"  Consensus Trend: {prediction.get('consensus_trend', 'unknown').upper()}")
    method_results = prediction.get("method_results", {})
    if method_results:
        ma = method_results.get("moving_average", {})
        print(f"  Moving Average: {ma.get('change_pct', 0):+.2f}% ({ma.get('trend', '-')})")
        rg = method_results.get("regression", {})
        print(f"  Regression Slope: {rg.get('slope', 0):+.0f} ({rg.get('trend', '-')})")
        mm = method_results.get("momentum", {})
        print(f"  7-day Momentum: {mm.get('7d_change_pct', 0):+.2f}% ({mm.get('trend', '-')})")
    next_price = prediction.get("next_predicted_price")
    if next_price:
        print(f"  Next predicted price: Rp {next_price:,.0f}")
    if prediction.get("current_price"):
        print(f"  Current price: Rp {prediction['current_price']:,.0f}")

    # Recommendations
    recommendations = results.get("recommendations", [])
    print("\n📋 RECOMMENDATIONS:")
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    else:
        print("  (tidak ada rekomendasi)")

    # RAG Responses
    rag_results = results.get("rag_responses", [])
    print("\n🤖 RAG RESPONSES:")
    if rag_results:
        for rag in rag_results:
            print(f"\n  Q: {rag.get('question', '')}")
            answer = rag.get("answer", "")
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

    print(f"\n  Average Score: {avg_score:.2%}")
    print(f"  Weighted Score: {weighted_score:.2%}")

    # Execution time
    exec_time = results.get("execution_time", 0)
    print(f"\n⏱️  Execution Time: {exec_time:.2f} seconds")

    # Model metrics
    if "model_metrics" in results:
        print("\n📊 MODEL PERFORMANCE (ML Training):")
        metrics = results["model_metrics"]
        for key, value in metrics.items():
            print(f"  {key}: {value}")

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

"""EvaluatorAgent - Untuk evaluasi kualitas output agent lain."""

import os
from typing import Dict, Any, List, Optional

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import groq
import numpy as np
from src.agents.base import BaseAgent
from src.config import Config


class EvaluatorAgent(BaseAgent):
    """Agent untuk mengevaluasi kualitas output dari agent lain."""

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(name="EvaluatorAgent")
        self.api_key = api_key or Config.GROQ_API_KEY
        if not self.api_key or self.api_key == "gsk_your_api_key_here":
            print("Warning: GROQ_API_KEY tidak diatur. EvaluatorAgent akan berjalan dalam mode offline.")
            self.client = None
        else:
            self.client = groq.Client(api_key=self.api_key)
        self.model = Config.GROQ_MODEL_EVALUATOR

    def process(self, input_data: Any) -> Any:
        """Implementasi abstract method untuk BaseAgent.

        Args:
            input_data: Dict berisi 'actual', 'predicted' untuk evaluasi prediksi.

        Returns:
            Hasil evaluasi.
        """
        if isinstance(input_data, dict):
            return self.evaluate_prediction(
                actual=input_data.get("actual", 0),
                predicted=input_data.get("predicted", 0),
                context=input_data.get("context", ""),
            )
        return None

    def evaluate_prediction(self, actual: float, predicted: float, context: str = "") -> Dict[str, Any]:
        """Evaluasi akurasi prediksi harga.

        Args:
            actual: Harga aktual.
            predicted: Harga prediksi.
            context: Konteks tambahan.

        Returns:
            Dictionary berisi evaluasi.
        """
        error = abs(actual - predicted)
        error_pct = (error / abs(actual)) * 100 if actual != 0 else 100

        # Tentukan akurasi berdasarkan percentage error
        if error_pct < 5:
            accuracy_level = "sangat_akurat"
            score = 1.0 - (error_pct / 20.0)
        elif error_pct < 15:
            accuracy_level = "akurat"
            score = 0.85 - ((error_pct - 5) / 100.0)
        elif error_pct < 30:
            accuracy_level = "moderat"
            score = 0.65 - ((error_pct - 15) / 100.0)
        elif error_pct < 50:
            accuracy_level = "kurang_akurat"
            score = 0.40 - ((error_pct - 30) / 100.0)
        else:
            accuracy_level = "tidak_akurat"
            score = max(0.1, 0.2 - ((error_pct - 50) / 200.0))

        return {
            "actual": round(actual, 2),
            "predicted": round(predicted, 2),
            "absolute_error": round(error, 2),
            "percentage_error": round(error_pct, 2),
            "accuracy_level": accuracy_level,
            "score": round(score, 4),
            "acceptable": error_pct < 20,  # Threshold acceptable: < 20%
        }

    def evaluate_rag_response(self, question: str, response: str, context: str) -> Dict[str, Any]:
        """Evaluasi kualitas respons RAG secara detail.

        Args:
            question: Pertanyaan user.
            response: Respons yang dihasilkan.
            context: Konteks yang digunakan.

        Returns:
            Dictionary berisi evaluasi kualitas.
        """
        # Evaluasi dasar tanpa LLM
        response_lower = response.lower()
        context_lower = context.lower()

        # Cek relevansi: apakah respons menjawab pertanyaan?
        question_keywords = set(question.lower().split())
        relevant_words = sum(1 for word in question_keywords if word in response_lower and len(word) > 3)
        relevance_score = min(relevant_words / max(len(question_keywords), 1), 1.0)

        # Cek groundedness: apakah respons berbasis konteks?
        context_words = [w for w in context_lower.split() if len(w) > 4]
        matched_context = sum(1 for word in context_words[:50] if word in response_lower)
        groundedness_score = matched_context / max(len(context_words[:50]), 1)

        # Cek kelengkapan
        has_numbers = bool(response_lower.replace(",", "").replace(".", "").replace("-", "").replace("rp", "").isdigit())
        completeness_score = min(relevance_score + groundedness_score * 0.5, 1.0)

        # Cek kejelasan (response length dan struktur)
        words = response.split()
        clarity_score = min(len(words) / 100.0, 1.0)  # Idealnya 100+ kata

        # Overall score (weighted average)
        overall_score = (
            relevance_score * 0.35 +
            groundedness_score * 0.30 +
            completeness_score * 0.20 +
            clarity_score * 0.15
        )

        return {
            "relevance": round(relevance_score, 4),
            "groundedness": round(groundedness_score, 4),
            "completeness": round(completeness_score, 4),
            "clarity": round(clarity_score, 4),
            "overall_score": round(overall_score, 4),
            "has_numbers": has_numbers,
        }

    def evaluate_effectiveness(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluasi efektivitas analisis bisnis.

        Args:
            analysis_results: Hasil analisis dari semua agent.

        Returns:
            Dictionary berisi evaluasi efektivitas.
        """
        insights = analysis_results.get("data_insights", [])
        recommendations = analysis_results.get("recommendations", [])
        rag_responses = analysis_results.get("rag_responses", [])

        # Hitung coverage: berapa persen query yang terjawab dengan baik
        answered_count = sum(
            1 for r in rag_responses
            if r.get("confidence", 0) > 0.3 and len(r.get("answer", "")) > 50
        )
        rag_coverage = answered_count / max(len(rag_responses), 1)

        # Score keseluruhan efektivitas
        effectiveness_score = (
            min(len(insights) / 5.0, 1.0) * 0.3 +
            min(len(recommendations) / 5.0, 1.0) * 0.3 +
            rag_coverage * 0.4
        )

        return {
            "insights_count": len(insights),
            "recommendations_count": len(recommendations),
            "rag_queries_answered": answered_count,
            "rag_total_queries": len(rag_responses),
            "rag_coverage": round(rag_coverage, 4),
            "effectiveness_score": round(effectiveness_score, 4),
            "effective": effectiveness_score >= 0.7,
        }

    def evaluate_explainability(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluasi explainability dari sistem.

        Args:
            analysis_results: Hasil analisis dari semua agent.

        Returns:
            Dictionary berisi evaluasi explainability.
        """
        insights = analysis_results.get("data_insights", [])
        recommendations = analysis_results.get("recommendations", [])

        # Cek apakah ada penjelasan numerik
        numeric_explanations = 0
        for item in insights + recommendations:
            if any(char.isdigit() for char in item):
                numeric_explanations += 1

        total_explanations = len(insights) + len(recommendations)
        explainability_score = numeric_explanations / max(total_explanations, 1)

        return {
            "total_explanations": total_explanations,
            "with_numbers": numeric_explanations,
            "explainability_score": round(explainability_score, 4),
            "good_explainability": explainability_score >= 0.5,
        }

    def evaluate_hallucination(self, rag_responses: List[Dict], ground_truth: str = "") -> Dict[str, Any]:
        """Estimasi tingkat halusinasi dalam respons RAG.

        Args:
            rag_responses: List hasil query RAG.
            ground_truth: Ground truth untuk validasi (opsional).

        Returns:
            Dictionary berisi metrik halusinasi.
        """
        if not rag_responses:
            return {"hallucination_rate": 0.0, "total": 0}

        hallucination_indicators = 0
        total = len(rag_responses)

        for resp in rag_responses:
            answer = resp.get("answer", "")
            sources = resp.get("sources", [])
            confidence = resp.get("confidence", 0)

            # Indikator halusinasi:
            # 1. Confidence sangat rendah tapi ada jawaban
            if confidence < 0.2 and len(answer) > 100:
                hallucination_indicators += 1
            # 2. Tidak ada source yang jelas
            if not sources or all(s == "unknown" for s in sources):
                hallucination_indicators += 0.5
            # 3. Jawaban terlalu panjang tanpa angka (curiga hallucination)
            words = answer.split()
            has_numbers = any(c.isdigit() for c in answer)
            if len(words) > 300 and not has_numbers:
                hallucination_indicators += 0.5

        hallucination_rate = min(hallucination_indicators / total, 1.0)

        return {
            "hallucination_rate": round(hallucination_rate, 4),
            "estimated_hallucinated": round(hallucination_indicators, 2),
            "total_responses": total,
            "low_hallucination": hallucination_rate < 0.3,
        }

    def generate_comprehensive_report(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate laporan evaluasi komprehensif.

        Args:
            analysis_results: Hasil lengkap dari CoordinatorAgent.

        Returns:
            Dictionary berisi laporan evaluasi menyeluruh.
        """
        evaluations = []

        # 1. Prediction Accuracy
        prediction = analysis_results.get("prediction", {})
        next_price = prediction.get("next_predicted_price")
        if next_price:
            prices = analysis_results.get("prices_history", [])
            if prices:
                current_price = prices[-1]
                pred_eval = self.evaluate_prediction(current_price, next_price)
                evaluations.append({
                    "metric": "Accuracy - Prediksi Harga",
                    "category": "accuracy",
                    "score": pred_eval["score"],
                    "details": pred_eval,
                })

        # 2. RAG Effectiveness
        rag_responses = analysis_results.get("rag_responses", [])
        if rag_responses:
            rag_scores = []
            for resp in rag_responses:
                rag_eval = self.evaluate_rag_response(
                    resp.get("question", ""),
                    resp.get("answer", ""),
                    "\n".join(resp.get("context", [])),
                )
                rag_scores.append(rag_eval["overall_score"])
            avg_rag_score = sum(rag_scores) / len(rag_scores) if rag_scores else 0.5
            evaluations.append({
                "metric": "Effectiveness - RAG Responses",
                "category": "effectiveness",
                "score": round(avg_rag_score, 4),
                "details": {"individual_scores": [round(s, 4) for s in rag_scores]},
            })

        # 3. Efficiency
        exec_time = analysis_results.get("execution_time", 0)
        efficiency_score = max(0, min(1.0, 1.0 - exec_time / 30.0))
        evaluations.append({
            "metric": "Efficiency - Waktu Eksekusi",
            "category": "efficiency",
            "score": round(efficiency_score, 4),
            "details": {"execution_time_seconds": exec_time},
        })

        # 4. Explainability
        explain_eval = self.evaluate_explainability(analysis_results)
        evaluations.append({
            "metric": "Explainability - Penjelasan Bisnis",
            "category": "explainability",
            "score": explain_eval["explainability_score"],
            "details": explain_eval,
        })

        # 5. Hallucination
        halluc_eval = self.evaluate_hallucination(rag_responses)
        hallucination_score = 1.0 - halluc_eval["hallucination_rate"]
        evaluations.append({
            "metric": "Hallucination - Kualitas Konten",
            "category": "hallucination",
            "score": round(hallucination_score, 4),
            "details": halluc_eval,
        })

        # Hitung rata-rata keseluruhan
        all_scores = [e["score"] for e in evaluations]
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0

        # Hitung weighted score (Accuracy dan Effectiveness lebih penting)
        weights = {
            "accuracy": 0.25,
            "effectiveness": 0.25,
            "efficiency": 0.15,
            "explainability": 0.20,
            "hallucination": 0.15,
        }
        weighted_score = sum(
            e["score"] * weights.get(e["category"], 0.2)
            for e in evaluations
        )

        report = {
            "total_metrics": len(evaluations),
            "metrics": evaluations,
            "average_score": round(avg_score, 4),
            "weighted_score": round(weighted_score, 4),
            "summary": self._generate_summary(avg_score, weighted_score, evaluations),
        }

        self.last_result = report
        self.status = "report_generated"
        return report

    def _generate_summary(self, avg_score: float, weighted_score: float, metrics: list) -> str:
        """Generate ringkasan evaluasi."""
        status = "LULUS" if avg_score >= 0.6 else "PERLU PERBAIKAN"

        lines = [
            f"=== LAPORAN EVALUASI SISTEM MULTI-AGENT ===",
            f"",
            f"Skor Rata-rata: {avg_score:.2%}",
            f"Skor Weighted: {weighted_score:.2%}",
            f"Status: {status}",
            f"",
            f"Detail per Metrik:",
        ]

        for m in metrics:
            icon = "✅" if m["score"] >= 0.6 else "⚠️"
            lines.append(f"  {icon} {m['metric']}: {m['score']:.2%}")

        lines.append("")
        lines.append("=" * 50)

        return "\n".join(lines)

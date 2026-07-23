"""CoordinatorAgent - Untuk mengkoordinasi semua agent."""

import time
from typing import Dict, Any, List, Optional
from src.agents.base import BaseAgent
from src.agents.data_agent import DataAgent
from src.agents.prediction_agent import PredictionAgent
from src.agents.rag_agent import RAGAgent
from src.agents.evaluator_agent import EvaluatorAgent
from src.config import Config


class CoordinatorAgent(BaseAgent):
    """Agent yang mengkoordinasi semua agent dalam sistem."""

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(name="CoordinatorAgent")
        self.api_key = api_key or Config.GROQ_API_KEY
        self.data_agent = DataAgent()
        self.prediction_agent = PredictionAgent()
        self.rag_agent = RAGAgent(api_key=self.api_key)
        self.evaluator_agent = EvaluatorAgent(api_key=self.api_key)

    def process(self, input_data: Any) -> Any:
        """Implementasi abstract method untuk BaseAgent.

        Args:
            input_data: Pertanyaan string atau dict.

        Returns:
            Hasil proses dari agent.
        """
        if isinstance(input_data, str):
            return self.ask_question(input_data)
        elif isinstance(input_data, dict) and "filepath" in input_data:
            return self.run_full_analysis(input_data["filepath"])
        return self.run_full_analysis()

    def run_full_analysis(self, filepath: str = "cabai.csv") -> Dict[str, Any]:
        """Jalankan analisis penuh dengan semua agent.

        Args:
            filepath: Path file CSV.

        Returns:
            Dictionary berisi hasil analisis komprehensif.
        """
        start_time = time.time()
        results = {}

        # Step 1: Data Agent - Load dan analisis data
        print("📊 DataAgent: Memuat dan menganalisis data...")
        self.data_agent.load_and_clean(filepath)
        analysis = self.data_agent.analyze()
        insights = self.data_agent.generate_insights()
        results["data_analysis"] = analysis
        results["data_insights"] = insights
        results["prices_history"] = self.data_agent.data["harga"].tolist()

        # Step 2: Prediction Agent - Prediksi harga
        print("🔮 PredictionAgent: Memprediksi harga...")
        prices = self.data_agent.data["harga"].tolist()

        # Siapkan training data dengan feature engineering
        from src.data.preprocessing import add_features, prepare_training_data
        featured_df = add_features(self.data_agent.data.copy())
        X_train, y_train = prepare_training_data(featured_df, lookback_days=7)

        # Split train/test
        split_idx = int(len(X_train) * 0.8)
        X_tr, X_te = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
        y_tr, y_te = y_train.iloc[:split_idx], y_train.iloc[split_idx:]

        # Training dan evaluasi model
        if len(X_te) > 0:
            self.prediction_agent.train_with_evaluation(X_tr.values, y_tr.values, X_te.values, y_te.values)
            test_predictions = self.prediction_agent.predict(X_te.values)
            results["model_metrics"] = self.prediction_agent.metrics

        # Analisis trend dengan multi-metode
        recent_prices = prices[-60:]  # 60 hari terakhir
        trend = self.prediction_agent.analyze_trend(recent_prices)
        recommendations = self.prediction_agent.generate_recommendations(trend, prices[-1])
        results["prediction"] = trend
        results["recommendations"] = recommendations

        # Step 3: RAG Agent - Build index dan jawab pertanyaan
        print("🤖 RAGAgent: Membangun index dan menjawab pertanyaan...")
        from src.vectorstore.document_generator import generate_documents_from_data
        documents = generate_documents_from_data(filepath)
        self.rag_agent.build_index(documents)

        # Query bisnis yang relevan
        queries = [
            "Berapa rata-rata harga cabai merah di Pasar Beringharjo berdasarkan data historis?",
            "Apa tren harga cabai merah secara keseluruhan dari awal hingga akhir periode data?",
            "Bulan mana waktu terbaik untuk membeli cabai merah dan berapa rata-rata harganya?",
            "Bagaimana volatilitas harga cabai merah? Berapa koefisien variasi dan standar deviasinya?",
        ]
        rag_results = []
        for query in queries:
            result = self.rag_agent.query(query)
            rag_results.append({"question": query, **result})

        results["rag_responses"] = rag_results

        # Step 4: Evaluation - Evaluasi semua output
        print("📈 EvaluatorAgent: Mengevaluasi hasil...")
        evaluations = []

        # 4a. Evaluasi prediksi (Accuracy)
        pred_eval = self.evaluator_agent.evaluate_prediction(
            actual=prices[-1],
            predicted=trend.get("next_predicted_price", prices[-1]),
        )
        evaluations.append({
            "metric": "Accuracy - Prediksi Harga",
            "category": "accuracy",
            "score": pred_eval["score"],
            "details": pred_eval,
        })

        # 4b. Evaluasi RAG responses (Effectiveness)
        rag_scores = []
        for rag_result in rag_results:
            rag_eval = self.evaluator_agent.evaluate_rag_response(
                rag_result["question"],
                rag_result["answer"],
                "\n".join(rag_result.get("context", [])),
            )
            rag_scores.append(rag_eval["overall_score"])

        avg_rag_score = sum(rag_scores) / len(rag_scores) if rag_scores else 0.5
        evaluations.append({
            "metric": "Effectiveness - RAG Responses",
            "category": "effectiveness",
            "score": round(avg_rag_score, 4),
            "details": {"individual_scores": [round(s, 4) for s in rag_scores]},
        })

        # 4c. Evaluasi efisiensi (Efficiency)
        duration = time.time() - start_time
        efficiency_score = max(0, min(1.0, 1.0 - duration / 30.0))
        evaluations.append({
            "metric": "Efficiency - Waktu Eksekusi",
            "category": "efficiency",
            "score": round(efficiency_score, 4),
            "details": {"execution_time_seconds": round(duration, 2)},
        })

        # 4d. Evaluasi explainability
        insights_text = " ".join(insights + recommendations)
        numeric_explanations = sum(
            1 for item in insights + recommendations
            if any(c.isdigit() for c in item)
        )
        total_explanations = len(insights) + len(recommendations)
        explainability_score = numeric_explanations / max(total_explanations, 1)
        evaluations.append({
            "metric": "Explainability - Penjelasan Bisnis",
            "category": "explainability",
            "score": round(explainability_score, 4),
            "details": {
                "total_explanations": total_explanations,
                "with_numbers": numeric_explanations,
            },
        })

        # 4e. Evaluasi hallucination
        halluc_indicators = 0
        for resp in rag_results:
            answer = resp.get("answer", "")
            confidence = resp.get("confidence", 0)
            sources = resp.get("sources", [])

            if confidence < 0.2 and len(answer) > 100:
                halluc_indicators += 1
            if not sources or all(s == "unknown" for s in sources):
                halluc_indicators += 0.5
            words = answer.split()
            has_numbers = any(c.isdigit() for c in answer)
            if len(words) > 300 and not has_numbers:
                halluc_indicators += 0.5

        hallucination_rate = min(halluc_indicators / max(len(rag_results), 1), 1.0)
        evaluations.append({
            "metric": "Hallucination - Kualitas Konten",
            "category": "hallucination",
            "score": round(1.0 - hallucination_rate, 4),
            "details": {
                "hallucination_rate": round(hallucination_rate, 4),
                "estimated_hallucinated": round(halluc_indicators, 2),
                "total_responses": len(rag_results),
            },
        })

        # Hitung skor akhir
        all_scores = [e["score"] for e in evaluations]
        avg_score = sum(all_scores) / len(all_scores)

        # Weighted score
        weights = {
            "accuracy": 0.25,
            "effectiveness": 0.25,
            "efficiency": 0.15,
            "explainability": 0.20,
            "hallucination": 0.15,
        }
        weighted_score = sum(e["score"] * weights.get(e["category"], 0.2) for e in evaluations)

        results["evaluations"] = evaluations
        results["average_score"] = round(avg_score, 4)
        results["weighted_score"] = round(weighted_score, 4)
        results["execution_time"] = round(time.time() - start_time, 2)

        # Generate comprehensive report
        eval_report = self.evaluator_agent.generate_comprehensive_report(results)
        results["evaluation_report"] = eval_report

        self.last_result = results
        self.status = "completed"
        return results

    def ask_question(self, question: str) -> Dict[str, Any]:
        """Tanyakan pertanyaan bisnis ke sistem.

        Args:
            question: Pertanyaan user.

        Returns:
            Dictionary berisi jawaban.
        """
        return self.rag_agent.query(question)

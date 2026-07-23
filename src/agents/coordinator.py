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
        """Jalankan analisis penuh dengan semua agent - PER KOMODITAS.

        Args:
            filepath: Path file CSV.

        Returns:
            Dictionary berisi hasil analisis komprehensif.
        """
        start_time = time.time()
        results = {}

        # Step 1: Data Agent - Load dan analisis data PER KOMODITAS
        print("📊 DataAgent: Memuat dan menganalisis data...")
        self.data_agent.load_and_clean(filepath)
        analysis = self.data_agent.analyze()
        insights = self.data_agent.generate_insights()
        results["data_analysis"] = analysis
        results["data_insights"] = insights

        # Step 2: Prediction Agent - Prediksi harga PER KOMODITAS (OPTIMIZED)
        print("🔮 PredictionAgent: Memprediksi harga per komoditas...")
        
        # Siapkan training dan prediksi PER KOMODITAS
        from src.data.preprocessing import add_features, prepare_training_data
        
        all_model_metrics = {}
        all_trends = {}
        all_recommendations = {}
        all_prices = {}
        
        # Pre-compute features untuk semua komoditas sekaligus (OPTIMIZED)
        featured_dfs = {}
        for commodity, group in self.data_agent.data.groupby("nama_komoditas"):
            short_name = commodity.split(",")[0]
            print(f"\n  📈 Processing {short_name}...")
            
            # Ekstrak harga per komoditas
            group_sorted = group.sort_values("tanggal")
            prices = group_sorted["harga"].tolist()
            all_prices[commodity] = prices
            
            # Feature engineering HANYA SEKALI (cached)
            if commodity not in featured_dfs:
                featured_dfs[commodity] = add_features(group_sorted.copy())
        
        # Training model dengan data yang sudah di-feature
        for commodity, featured_df in featured_dfs.items():
            X_train, y_train = prepare_training_data(featured_df, lookback_days=7)
            
            # Split train/test
            if len(X_train) > 10:
                split_idx = int(len(X_train) * 0.8)
                X_tr, X_te = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
                y_tr, y_te = y_train.iloc[:split_idx], y_train.iloc[split_idx:]
                
                # Training dan evaluasi model
                if len(X_te) > 0:
                    self.prediction_agent.train_with_evaluation(
                        X_tr.values, y_tr.values, X_te.values, y_te.values
                    )
                    all_model_metrics[commodity] = self.prediction_agent.metrics
                    
                    # Analisis trend
                    prices = all_prices[commodity]
                    recent_prices = prices[-60:]
                    trend = self.prediction_agent.analyze_trend(recent_prices)
                    recommendations = self.prediction_agent.generate_recommendations(trend, prices[-1])
                    
                    all_trends[commodity] = trend
                    all_recommendations[commodity] = recommendations
        
        results["model_metrics"] = all_model_metrics
        results["predictions"] = all_trends
        results["recommendations"] = all_recommendations

        # Step 3: RAG Agent - Build index dan jawab pertanyaan (OPTIMIZED)
        print("🤖 RAGAgent: Membangun index dan menjawab pertanyaan...")
        from src.vectorstore.document_generator import generate_documents_from_data
        
        # Gunakan cache jika tersedia
        import pickle
        cache_path = Config.CACHE_PATH / "rag_index.pkl"
        
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    cached_time, cached_docs = pickle.load(f)
                # Validasi cache (max 1 jam)
                if time.time() - cached_time < 3600:
                    print("  🔄 Menggunakan cache RAG index...")
                    documents = cached_docs
                else:
                    documents = generate_documents_from_data(filepath)
            except Exception:
                documents = generate_documents_from_data(filepath)
        else:
            documents = generate_documents_from_data(filepath)
        
        self.rag_agent.build_index(documents)

        # Query bisnis yang relevan (OPTIMIZED - kurangi query redundan)
        queries = []
        for commodity in self.data_agent.data["nama_komoditas"].unique():
            short_name = commodity.split(",")[0]
            # Hanya 2 query penting per komoditas (dari 4 menjadi 2)
            queries.extend([
                f"Berapa rata-rata dan tren harga {short_name} berdasarkan data historis?",
                f"Kapan waktu terbaik untuk membeli {short_name} dan berapa volatilitasnya?",
            ])
        
        rag_results = []
        for query in queries:
            result = self.rag_agent.query(query)
            rag_results.append({"question": query, **result})

        results["rag_responses"] = rag_results

        # Step 4: Evaluation - Evaluasi semua output
        print("📈 EvaluatorAgent: Mengevaluasi hasil...")
        evaluations = []

        # 4a. Evaluasi prediksi (Accuracy) - rata-rata semua komoditas
        avg_accuracy = 0
        accuracy_count = 0
        for commodity, trend in all_trends.items():
            pred_eval = self.evaluator_agent.evaluate_prediction(
                actual=all_prices[commodity][-1],
                predicted=trend.get("next_predicted_price", all_prices[commodity][-1]),
            )
            avg_accuracy += pred_eval["score"]
            accuracy_count += 1
        
        avg_accuracy = avg_accuracy / max(accuracy_count, 1)
        evaluations.append({
            "metric": "Accuracy - Prediksi Harga",
            "category": "accuracy",
            "score": round(avg_accuracy, 4),
            "details": {"avg_score": round(avg_accuracy, 4)},
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
        
        # Formula efficiency yang lebih realistis:
        # - < 5 detik: perfect score (1.0)
        # - 5-30 detik: linear decay dari 1.0 ke 0.5
        # - 30-60 detik: linear decay dari 0.5 ke 0.0
        # - > 60 detik: minimum 0.0
        if duration <= 5:
            efficiency_score = 1.0
        elif duration <= 30:
            efficiency_score = 1.0 - ((duration - 5) / 25) * 0.5  # 1.0 → 0.5
        elif duration <= 60:
            efficiency_score = 0.5 - ((duration - 30) / 30) * 0.5  # 0.5 → 0.0
        else:
            efficiency_score = 0.0
        
        evaluations.append({
            "metric": "Efficiency - Waktu Eksekusi",
            "category": "efficiency",
            "score": round(efficiency_score, 4),
            "details": {"execution_time_seconds": round(duration, 2)},
        })

        # 4d. Evaluasi explainability
        all_insights = list(insights)
        all_recs = []
        for recs in all_recommendations.values():
            all_recs.extend(recs)
        
        insights_text = " ".join(all_insights + all_recs)
        numeric_explanations = sum(
            1 for item in all_insights + all_recs
            if any(c.isdigit() for c in item)
        )
        total_explanations = len(all_insights) + len(all_recs)
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

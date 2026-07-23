"""FastAPI server untuk demo Multi-Agent Commodity Intelligence System."""

import time
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.agents.coordinator import CoordinatorAgent
from src.config import Config


# Inisialisasi aplikasi
app = FastAPI(
    title="Multi-Agent Commodity Intelligence System",
    description="Sistem multi-agent untuk analisis harga komoditas agrikultur Indonesia.",
    version="0.1.0",
)

# Inisialisasi coordinator (akan dibuat saat pertama kali digunakan)
coordinator: Optional[CoordinatorAgent] = None


def get_coordinator() -> CoordinatorAgent:
    """Dapatkan instance CoordinatorAgent."""
    global coordinator
    if coordinator is None:
        coordinator = CoordinatorAgent()
    return coordinator


# Request/Response models
class QuestionRequest(BaseModel):
    question: str


class AnalysisResponse(BaseModel):
    data_analysis: dict
    prediction: dict
    recommendations: list
    rag_responses: list
    evaluation: dict
    execution_time: float


class HealthResponse(BaseModel):
    status: str
    version: str


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", version="0.1.0")


@app.post("/analyze", response_model=AnalysisResponse)
async def run_full_analysis(filepath: str = "cabai.csv"):
    """Jalankan analisis penuh dengan semua agent."""
    try:
        coord = get_coordinator()
        results = coord.run_full_analysis(filepath)
        return AnalysisResponse(**results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=dict)
async def ask_question(request: QuestionRequest):
    """Tanyakan pertanyaan bisnis ke sistem."""
    try:
        coord = get_coordinator()
        result = coord.ask_question(request.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/price-stats", response_model=dict)
async def get_price_stats():
    """Dapatkan statistik harga."""
    try:
        coord = get_coordinator()
        coord.data_agent.load_and_clean("cabai.csv")
        analysis = coord.data_agent.analyze()
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

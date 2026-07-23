"""Tests untuk multi-agent framework."""

import pytest
import sys
from pathlib import Path

# Tambahkan root directory ke Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.data_agent import DataAgent
from src.agents.prediction_agent import PredictionAgent
from src.agents.base import BaseAgent


def test_data_agent_load():
    """Test DataAgent loading data."""
    agent = DataAgent()
    df = agent.load_and_clean("cabai.csv")
    assert len(df) > 0
    assert agent.status == "loaded"


def test_data_agent_analysis():
    """Test DataAgent analysis."""
    agent = DataAgent()
    agent.load_and_clean("cabai.csv")
    analysis = agent.analyze()
    assert "price_stats" in analysis
    assert "volatility" in analysis


def test_data_agent_insights():
    """Test DataAgent insights generation."""
    agent = DataAgent()
    agent.load_and_clean("cabai.csv")
    insights = agent.generate_insights()
    assert len(insights) > 0
    assert all(isinstance(i, str) for i in insights)


def test_prediction_agent_trend():
    """Test PredictionAgent trend analysis."""
    agent = PredictionAgent()
    prices = [80000 + i * 1000 for i in range(30)]
    result = agent.analyze_trend(prices)
    assert "consensus_trend" in result
    assert "method_results" in result
    assert "next_predicted_price" in result


def test_prediction_agent_recommendations():
    """Test PredictionAgent recommendations."""
    agent = PredictionAgent()
    prices = [80000 + i * 1000 for i in range(30)]
    analysis = agent.analyze_trend(prices)
    recommendations = agent.generate_recommendations(analysis, 80000)
    assert len(recommendations) > 0


def test_base_agent_abstract():
    """Test BaseAgent is abstract."""
    # BaseAgent tidak bisa diinstansiasi langsung
    with pytest.raises(TypeError):
        BaseAgent()

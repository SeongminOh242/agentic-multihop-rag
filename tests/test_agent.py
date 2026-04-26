from unittest.mock import patch

import pytest

from src.agent import AgenticRAG


CORPUS = [
    "Albert Einstein was born in Ulm, Germany.",
    "Ulm is a city in the state of Baden-Württemberg.",
    "Einstein developed the theory of relativity.",
]


def test_agent_returns_answer_with_hop_count():
    agent = AgenticRAG(CORPUS, max_hops=3)
    with patch.object(agent, "_call_llm", side_effect=[
        '{"sub_query": "Einstein birthplace", "sufficient": false}',
        '{"sub_query": "Ulm Germany state", "sufficient": false}',
        '{"sub_query": "", "sufficient": true, "final_answer": "Baden-Württemberg"}',
    ]):
        result = agent.answer("What state in Germany was Einstein born in?")
    assert "answer" in result
    assert "num_hops" in result
    assert result["num_hops"] <= 3


def test_agent_stops_at_max_hops():
    corpus = ["unrelated document " + str(i) for i in range(10)]
    agent = AgenticRAG(corpus, max_hops=2)
    with patch.object(agent, "_call_llm", return_value='{"sub_query": "still searching", "sufficient": false}'):
        result = agent.answer("impossible question")
    assert result["num_hops"] <= 2


def test_agent_result_has_required_fields():
    agent = AgenticRAG(CORPUS, max_hops=2)
    with patch.object(agent, "_call_llm", return_value='{"sufficient": true, "final_answer": "Baden-Württemberg"}'):
        result = agent.answer("Where was Einstein born?")
    assert "answer" in result
    assert "num_hops" in result
    assert "retrieved_docs" in result
    assert "per_hop_docs" in result
    assert "sub_queries" in result
    assert "hop_traces" in result
    assert result["hop_traces"][0]["decision"]["final_answer"] == "Baden-Württemberg"


def test_agent_handles_malformed_llm_json():
    agent = AgenticRAG(CORPUS, max_hops=1)
    with patch.object(agent, "_call_llm", return_value="not valid json at all"):
        result = agent.answer("some question")
    assert "answer" in result
    assert result["num_hops"] <= 1


def test_agent_returns_final_answer_on_sufficient():
    agent = AgenticRAG(CORPUS, max_hops=3)
    with patch.object(agent, "_call_llm", return_value='{"sufficient": true, "final_answer": "Baden-Württemberg"}'):
        result = agent.answer("What state is Ulm in?")
    assert result["answer"] == "Baden-Württemberg"
    assert result["num_hops"] == 2

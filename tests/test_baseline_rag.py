from unittest.mock import patch

from src.baseline_rag import StaticRAG


def test_static_rag_returns_answer_and_docs() -> None:
    corpus = [
        "The Eiffel Tower is located in Paris, France.",
        "Paris is the capital of France.",
        "Python is a programming language.",
    ]
    rag = StaticRAG(corpus)

    with patch.object(rag, "_call_llm", return_value="Paris, France"):
        result = rag.answer("Where is the Eiffel Tower?")

    assert "answer" in result
    assert "retrieved_docs" in result
    assert isinstance(result["retrieved_docs"], list)


def test_static_rag_retrieved_docs_count() -> None:
    corpus = [f"doc {index}" for index in range(20)]
    rag = StaticRAG(corpus, top_k=5)

    with patch.object(rag, "_call_llm", return_value="some answer"):
        result = rag.answer("test query")

    assert len(result["retrieved_docs"]) <= 5

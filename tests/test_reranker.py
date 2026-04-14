from src.reranker import CrossEncoderReranker


def test_reranker_returns_sorted_results() -> None:
    reranker = CrossEncoderReranker()
    query = "Where is the Eiffel Tower?"
    docs = [
        {"text": "Python is a programming language.", "score": 0.9, "doc_id": 0},
        {"text": "The Eiffel Tower is located in Paris, France.", "score": 0.5, "doc_id": 1},
    ]

    reranked = reranker.rerank(query, docs, top_k=2)

    assert len(reranked) == 2
    assert "Eiffel" in reranked[0]["text"]


def test_reranker_top_k_limits_results() -> None:
    reranker = CrossEncoderReranker()
    query = "capital of France"
    docs = [{"text": f"doc {index}", "score": float(index), "doc_id": index} for index in range(5)]

    reranked = reranker.rerank(query, docs, top_k=3)

    assert len(reranked) == 3

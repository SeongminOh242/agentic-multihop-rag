from src.retriever import BM25Retriever


def test_bm25_returns_top_k_docs() -> None:
    corpus = [
        "The Eiffel Tower is in Paris France.",
        "Python is a programming language.",
        "Paris is the capital of France.",
        "Machine learning uses statistical methods.",
    ]
    retriever = BM25Retriever(corpus)

    results = retriever.retrieve("Eiffel Tower Paris", top_k=2)

    assert len(results) == 2
    assert isinstance(results[0], dict)
    assert "text" in results[0]
    assert "score" in results[0]


def test_bm25_most_relevant_doc_ranked_first() -> None:
    corpus = [
        "The Eiffel Tower is in Paris France.",
        "Python is a programming language.",
    ]
    retriever = BM25Retriever(corpus)

    results = retriever.retrieve("Eiffel Tower", top_k=2)

    assert "Eiffel" in results[0]["text"]

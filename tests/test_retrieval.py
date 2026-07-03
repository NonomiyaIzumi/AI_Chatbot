from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from healthcare_assistant.retrieval import Retriever, cosine_similarity_matrix, rank_top_k
from healthcare_assistant.schemas import KnowledgeEntry


def test_cosine_similarity_identical_vectors_is_one():
    v = np.array([1.0, 2.0, 3.0])
    doc_matrix = np.array([v])
    scores = cosine_similarity_matrix(v, doc_matrix)
    assert scores[0] == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero():
    query = np.array([1.0, 0.0])
    doc_matrix = np.array([[0.0, 1.0]])
    scores = cosine_similarity_matrix(query, doc_matrix)
    assert scores[0] == pytest.approx(0.0, abs=1e-9)


def test_cosine_similarity_opposite_vectors_is_negative_one():
    query = np.array([1.0, 0.0])
    doc_matrix = np.array([[-1.0, 0.0]])
    scores = cosine_similarity_matrix(query, doc_matrix)
    assert scores[0] == pytest.approx(-1.0)


def test_rank_top_k_orders_by_similarity_descending():
    query = np.array([1.0, 0.0])
    doc_matrix = np.array(
        [
            [0.0, 1.0],  # orthogonal -> 0
            [1.0, 0.0],  # identical -> 1
            [-1.0, 0.0],  # opposite -> -1
            [0.7, 0.7],  # partial match -> ~0.707
        ]
    )
    top_indices = rank_top_k(query, doc_matrix, top_k=2)
    assert top_indices == [1, 3]


def _entry(entry_id: str, condition: str) -> KnowledgeEntry:
    return KnowledgeEntry(
        id=entry_id,
        symptoms=["symptom a", "symptom b"],
        condition=condition,
        advice="advice text",
    )


def test_retriever_returns_entries_in_similarity_order():
    entries = [_entry("a", "Condition A"), _entry("b", "Condition B"), _entry("c", "Condition C")]
    doc_matrix = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])

    fake_client = MagicMock()
    fake_client.models.embed_content.return_value = SimpleNamespace(
        embeddings=[SimpleNamespace(values=[1.0, 0.0])]
    )

    retriever = Retriever(
        client=fake_client,
        embedding_model="fake-embedding-model",
        entries=entries,
        doc_matrix=doc_matrix,
        top_k=2,
    )

    result = retriever.retrieve("some query")

    assert [e.id for e in result] == ["a", "b"]
    call_kwargs = fake_client.models.embed_content.call_args.kwargs
    assert call_kwargs["model"] == "fake-embedding-model"
    assert call_kwargs["contents"] == ["some query"]
    assert call_kwargs["config"].task_type == "RETRIEVAL_QUERY"

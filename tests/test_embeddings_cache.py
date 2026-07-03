from types import SimpleNamespace
from unittest.mock import MagicMock

from healthcare_assistant.embeddings import get_or_build_kb_embeddings
from healthcare_assistant.schemas import KnowledgeEntry


def _fake_embed_response(n: int):
    return SimpleNamespace(embeddings=[SimpleNamespace(values=[float(i), 0.0]) for i in range(n)])


def test_unchanged_entries_are_not_recomputed_on_second_call(tmp_path):
    cache_path = tmp_path / "cache.json"
    entries = [
        KnowledgeEntry(id="a", symptoms=["s1"], condition="A", advice="advice a"),
        KnowledgeEntry(id="b", symptoms=["s2"], condition="B", advice="advice b"),
    ]

    client = MagicMock()
    client.models.embed_content.return_value = _fake_embed_response(2)

    matrix1 = get_or_build_kb_embeddings(client, "fake-model", entries, cache_path)
    assert client.models.embed_content.call_count == 1
    assert matrix1.shape == (2, 2)

    client.models.embed_content.reset_mock()
    matrix2 = get_or_build_kb_embeddings(client, "fake-model", entries, cache_path)

    client.models.embed_content.assert_not_called()
    assert (matrix1 == matrix2).all()


def test_changed_entry_triggers_recompute_for_that_entry_only(tmp_path):
    cache_path = tmp_path / "cache.json"
    entries = [
        KnowledgeEntry(id="a", symptoms=["s1"], condition="A", advice="advice a"),
        KnowledgeEntry(id="b", symptoms=["s2"], condition="B", advice="advice b"),
    ]

    client = MagicMock()
    client.models.embed_content.return_value = _fake_embed_response(2)
    get_or_build_kb_embeddings(client, "fake-model", entries, cache_path)

    entries[1] = KnowledgeEntry(id="b", symptoms=["s2", "s3"], condition="B", advice="advice b")
    client.models.embed_content.reset_mock()
    client.models.embed_content.return_value = _fake_embed_response(1)

    get_or_build_kb_embeddings(client, "fake-model", entries, cache_path)

    client.models.embed_content.assert_called_once()
    call_kwargs = client.models.embed_content.call_args.kwargs
    assert len(call_kwargs["contents"]) == 1

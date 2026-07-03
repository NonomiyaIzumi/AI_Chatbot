import hashlib
import json
import os
from pathlib import Path
from typing import Literal

import numpy as np
from google import genai
from google.genai import types
from loguru import logger

from healthcare_assistant.knowledge_base import doc_text
from healthcare_assistant.schemas import KnowledgeEntry

TaskType = Literal["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]


def embed_texts(client: genai.Client, model: str, texts: list[str], task_type: TaskType) -> list[list[float]]:
    resp = client.models.embed_content(
        model=model,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    assert resp.embeddings is not None
    return [e.values for e in resp.embeddings]  # type: ignore[misc]


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_or_build_kb_embeddings(
    client: genai.Client,
    model: str,
    entries: list[KnowledgeEntry],
    cache_path: Path,
) -> np.ndarray:
    cache: dict = {"embedding_model": model, "entries": {}}
    if cache_path.exists():
        try:
            loaded = json.loads(cache_path.read_text())
            if loaded.get("embedding_model") == model:
                cache = loaded
        except json.JSONDecodeError:
            logger.warning("Embeddings cache at {} is corrupt; rebuilding", cache_path)

    cached_entries: dict = cache.get("entries", {})
    to_embed: list[KnowledgeEntry] = []
    hashes: dict[str, str] = {}

    for entry in entries:
        text = doc_text(entry)
        content_hash = _content_hash(text)
        hashes[entry.id] = content_hash
        cached = cached_entries.get(entry.id)
        if cached is None or cached.get("content_hash") != content_hash:
            to_embed.append(entry)

    if to_embed:
        vectors = embed_texts(client, model, [doc_text(e) for e in to_embed], "RETRIEVAL_DOCUMENT")
        for entry, vector in zip(to_embed, vectors):
            cached_entries[entry.id] = {"content_hash": hashes[entry.id], "embedding": vector}

    logger.info(
        "Knowledge base embeddings: {} reused, {} recomputed",
        len(entries) - len(to_embed),
        len(to_embed),
    )

    new_cache = {"embedding_model": model, "entries": cached_entries}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(new_cache))
    os.replace(tmp_path, cache_path)

    matrix = np.array([cached_entries[entry.id]["embedding"] for entry in entries])
    return matrix

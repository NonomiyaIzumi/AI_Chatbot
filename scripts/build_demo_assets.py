import json
from pathlib import Path

from healthcare_assistant.config import load_settings
from healthcare_assistant.embeddings import embed_texts
from healthcare_assistant.gemini_client import build_client
from healthcare_assistant.knowledge_base import doc_text, load_knowledge_base

OUTPUT_PATH = Path("demo/kb_embeddings.json")


def main() -> None:
    settings = load_settings()
    client = build_client(settings)
    entries = load_knowledge_base()

    vectors = embed_texts(
        client, settings.gemini.embedding_model, [doc_text(e) for e in entries], "RETRIEVAL_DOCUMENT"
    )

    payload = {
        "embedding_model": settings.gemini.embedding_model,
        "entries": [
            {
                "id": entry.id,
                "symptoms": entry.symptoms,
                "condition": entry.condition,
                "advice": entry.advice,
                "urgent": entry.urgent,
                "embedding": vector,
            }
            for entry, vector in zip(entries, vectors)
        ],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload))
    print(f"Wrote {len(entries)} entries to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

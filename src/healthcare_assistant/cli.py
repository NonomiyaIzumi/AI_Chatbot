import uuid

from loguru import logger

from healthcare_assistant.assistant import HealthcareAssistant
from healthcare_assistant.config import load_settings
from healthcare_assistant.db import init_db
from healthcare_assistant.embeddings import get_or_build_kb_embeddings
from healthcare_assistant.gemini_client import build_client
from healthcare_assistant.knowledge_base import load_knowledge_base
from healthcare_assistant.retrieval import Retriever


def main() -> None:
    settings = load_settings()
    init_db(settings.storage.db_path)

    client = build_client(settings)
    entries = load_knowledge_base()
    doc_matrix = get_or_build_kb_embeddings(
        client, settings.gemini.embedding_model, entries, settings.storage.embeddings_cache_path
    )
    retriever = Retriever(
        client, settings.gemini.embedding_model, entries, doc_matrix, settings.retrieval.top_k
    )
    assistant = HealthcareAssistant(client, retriever, settings)

    user_id = str(uuid.uuid4())
    logger.info("Starting healthcare assistant session user_id={}", user_id)

    print("Healthcare Assistant (educational demo, not a real diagnostic tool).")
    print("Describe your symptoms, or type 'exit' to quit.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        answer = assistant.ask(user_id, query)
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()

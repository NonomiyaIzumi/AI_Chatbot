import json
from importlib import resources

from healthcare_assistant.schemas import KnowledgeEntry


def load_knowledge_base() -> list[KnowledgeEntry]:
    raw = resources.files("healthcare_assistant").joinpath("data", "knowledge_base.json").read_text()
    return [KnowledgeEntry(**entry) for entry in json.loads(raw)]


def doc_text(entry: KnowledgeEntry) -> str:
    return (
        f"Symptoms: {', '.join(entry.symptoms)}. "
        f"Likely condition: {entry.condition}. "
        f"Advice: {entry.advice}"
    )

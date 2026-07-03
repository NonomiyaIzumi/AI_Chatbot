import numpy as np
from google import genai

from healthcare_assistant.embeddings import embed_texts
from healthcare_assistant.schemas import KnowledgeEntry


def cosine_similarity_matrix(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    d = doc_matrix / (np.linalg.norm(doc_matrix, axis=1, keepdims=True) + 1e-10)
    return d @ q


def rank_top_k(query_vec: np.ndarray, doc_matrix: np.ndarray, top_k: int) -> list[int]:
    scores = cosine_similarity_matrix(query_vec, doc_matrix)
    return list(np.argsort(scores)[::-1][:top_k])


class Retriever:
    def __init__(
        self,
        client: genai.Client,
        embedding_model: str,
        entries: list[KnowledgeEntry],
        doc_matrix: np.ndarray,
        top_k: int,
    ) -> None:
        self.client = client
        self.embedding_model = embedding_model
        self.entries = entries
        self.doc_matrix = doc_matrix
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[KnowledgeEntry]:
        query_vec = np.array(
            embed_texts(self.client, self.embedding_model, [query], "RETRIEVAL_QUERY")[0]
        )
        idx = rank_top_k(query_vec, self.doc_matrix, top_k or self.top_k)
        return [self.entries[i] for i in idx]


def format_context(entries: list[KnowledgeEntry]) -> str:
    lines = []
    for i, entry in enumerate(entries, start=1):
        lines.append(
            f"Reference {i}: symptoms=[{', '.join(entry.symptoms)}] "
            f"condition={entry.condition!r} advice={entry.advice!r} urgent={entry.urgent}"
        )
    return "\n".join(lines)

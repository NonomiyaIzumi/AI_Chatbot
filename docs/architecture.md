# Architecture

```
User query
   │
   ▼
Retriever.retrieve()  ──►  Gemini embeddings (RETRIEVAL_QUERY)
   │                              │
   │                              ▼
   │                    cosine similarity vs. cached
   │                    knowledge-base document embeddings
   │                    (RETRIEVAL_DOCUMENT, gemini-embedding-001)
   ▼
Top-k KnowledgeEntry references
   │
   ▼
HealthcareAssistant.ask()
   │  builds prompt: [context references] + [user question]
   ▼
Gemini generate_content (system instruction + HEALTH_TOOLS)
   │
   ├─ no function_calls ─► return resp.text
   │
   └─ function_calls ─► dispatch to tools.py
           │              ├─ log_symptom_check   → SQLite symptom_logs
           │              ├─ get_patient_history  → SQLite symptom_logs (read)
           │              └─ schedule_appointment → SQLite appointments
           │
           └─ function response fed back into `contents`, loop continues
              (bounded by assistant.max_function_call_rounds)
```

Knowledge base entries live in `src/healthcare_assistant/data/knowledge_base.json` and are
embedded once; embeddings are cached in `data/embeddings_cache.json`, keyed by entry `id` and a
content hash, so unchanged entries are never re-embedded.

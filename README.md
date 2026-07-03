# Healthcare Assistant

> **Disclaimer:** This is an educational demo. It does not provide medical diagnoses and is not
> a substitute for professional medical advice. Always consult a qualified healthcare
> professional for real medical concerns.

A Gemini-powered assistant demonstrating **RAG** (retrieval-augmented generation over a small
symptom knowledge base) combined with **function calling** against a **SQLite** database, inspired
by the Kaggle "AI-Powered Healthcare Assistant" notebook / Gen AI Intensive capstone pattern.

See [docs/architecture.md](docs/architecture.md) for how retrieval and function calling fit together.

## Setup

1. Install dependencies:
   ```bash
   uv sync
   ```
2. Copy `.env.example` to `.env` and set your Gemini API key:
   ```bash
   cp .env.example .env
   # then edit .env and set GOOGLE_API_KEY=<your key>
   ```
   Get a key from [Google AI Studio](https://aistudio.google.com/apikey).
3. (Optional) adjust `configs/app.toml` for model names, retrieval `top_k`, or storage paths.

## Run

```bash
uv run python scripts/chat.py
# or
uv run healthcare-assistant
```

This starts an interactive REPL. Describe symptoms in plain language; the assistant retrieves
relevant reference entries, asks Gemini for a grounded response, and may call tools to log the
interaction or request a follow-up appointment. Type `exit` to quit.

## Development

```bash
uv run pytest              # run tests (offline, no API key needed)
uv run ruff check .        # lint
uv run ruff format .       # format
uv run pyright             # type check
uv run pre-commit install  # enable pre-commit hooks
```

## Project layout

```
configs/        non-secret app config (model names, top_k, paths)
docs/           architecture notes
scripts/        CLI entry point (scripts/chat.py)
src/healthcare_assistant/
  schemas.py          pydantic models (KnowledgeEntry, SymptomLog, Appointment)
  config.py           pydantic-settings Settings + load_settings()
  knowledge_base.py   loads the bundled synthetic symptom knowledge base
  embeddings.py       Gemini embeddings + on-disk cache
  retrieval.py        cosine-similarity retrieval (Retriever)
  db.py               SQLite schema + CRUD
  tools.py            Gemini function declarations + per-session dispatch
  gemini_client.py    genai.Client construction
  prompts.py          system instruction (incl. medical disclaimer)
  assistant.py        HealthcareAssistant orchestration loop
  cli.py              interactive REPL entry point
tests/          offline unit tests (no network / API key required)
```

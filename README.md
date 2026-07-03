# Healthcare Assistant

**[Try the live demo →](https://nonomiyaizumi.github.io/AI_Chatbot/)** (bring your own free Gemini API key, nothing is stored)

> **Disclaimer:** This is an educational demo. It does not provide medical diagnoses and is not
> a substitute for professional medical advice. Always consult a qualified healthcare
> professional for real medical concerns.

An AI assistant that lets you describe symptoms in plain language and get back grounded,
educational guidance — inspired by the Kaggle ["AI-Powered Healthcare Assistant"](https://www.kaggle.com/code/nishithatirumalaraju/ai-powered-healthcare-assistant)
notebook / Gen AI Intensive capstone pattern, rebuilt as a proper application.

## What it does

- **Understands symptoms in natural language.** Describe how you feel — "I have a runny nose
  and sneezing" — and the assistant identifies the relevant reference condition(s).
- **Retrieval-augmented generation (RAG).** Every answer is grounded in a curated knowledge base
  of 14 common conditions (symptoms, likely diagnosis, self-care advice, urgency flag). The
  assistant embeds your message with Gemini, ranks the knowledge base by cosine similarity, and
  only answers from what it retrieves — it doesn't invent conditions.
- **Function calling.** The model can call three tools mid-conversation:
  - `log_symptom_check` — records what was discussed for later reference
  - `get_patient_history` — recalls a user's past symptom checks
  - `schedule_appointment` — flags urgent/severe cases for human follow-up
- **Built-in safety guardrails.** The system prompt forces every response to end with a
  "not a medical diagnosis" disclaimer and to escalate urgent symptoms toward real care.
- **Two ways to use it:**
  1. A **local web app** (FastAPI + a small chat UI) backed by the full Python pipeline, with
     real SQLite persistence — run it yourself with your own API key server-side.
  2. A **public static demo** (plain HTML/JS, hosted on GitHub Pages) that calls Gemini directly
     from your browser using a key you paste in yourself; nothing is sent to any server but
     Google's own API.

See [docs/architecture.md](docs/architecture.md) for how retrieval and function calling fit
together end to end.

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

**Web UI** (recommended — chat in your browser):
```bash
uv run healthcare-assistant-web
# open http://127.0.0.1:8000
```

**Terminal REPL:**
```bash
uv run python scripts/chat.py
# or
uv run healthcare-assistant
```

Either way: describe symptoms in plain language, the assistant retrieves relevant reference
entries, asks Gemini for a grounded response, and may call tools to log the interaction or
request a follow-up appointment.

## Development

```bash
uv run pytest              # run tests (offline, no API key needed)
uv run ruff check .        # lint
uv run ruff format .       # format
uv run pyright             # type check
uv run pre-commit install  # enable pre-commit hooks
```

To regenerate the static demo's precomputed knowledge-base embeddings after editing
`src/healthcare_assistant/data/knowledge_base.json`:
```bash
uv run python scripts/build_demo_assets.py
```

## Project layout

```
configs/        non-secret app config (model names, top_k, paths)
demo/           static, client-side-only demo deployed to GitHub Pages
docs/           architecture notes
scripts/        CLI entry points (scripts/chat.py, scripts/serve.py, scripts/build_demo_assets.py)
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
  webapp.py           FastAPI app (local web UI backend)
  web_static/         local web UI frontend (HTML/CSS/JS)
tests/          offline unit tests (no network / API key required)
```

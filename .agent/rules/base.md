# Agent rules for healthcare-assistant

- Package name: `healthcare_assistant` (src layout under `src/healthcare_assistant/`).
- `Settings` / `load_settings()` must never be instantiated at module import time — only inside
  `cli.main()` — so tests, ruff, and pyright can run without `GOOGLE_API_KEY` set.
- Tool wrapper functions in `tools.py` must return plain JSON-serializable `dict`s (they go
  straight into `types.Part.from_function_response`).
- `user_id` is always an app-injected keyword-only argument bound via `functools.partial` in
  `build_dispatch`, never part of the model-facing function-calling JSON schema.
- The system instruction in `prompts.py` must always retain the "not a medical diagnosis"
  disclaimer and the instruction to escalate urgent symptoms.
- Tests must never require `GOOGLE_API_KEY` or network access — mock `genai.Client` where needed.

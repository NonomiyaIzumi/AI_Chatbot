from google import genai
from google.genai import types
from loguru import logger

from healthcare_assistant.config import Settings
from healthcare_assistant.prompts import SYSTEM_INSTRUCTION
from healthcare_assistant.retrieval import Retriever, format_context
from healthcare_assistant.tools import HEALTH_TOOLS, build_dispatch


class HealthcareAssistant:
    def __init__(self, client: genai.Client, retriever: Retriever, settings: Settings) -> None:
        self.client = client
        self.retriever = retriever
        self.settings = settings

    def ask(self, user_id: str, query: str) -> str:
        context = self.retriever.retrieve(query)
        context_block = format_context(context)
        contents: list[types.Content] = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=f"{context_block}\n\nUser question: {query}")],
            )
        ]
        dispatch = build_dispatch(user_id, self.settings.storage.db_path)
        collected_text: list[str] = []

        for _ in range(self.settings.assistant.max_function_call_rounds):
            resp = self.client.models.generate_content(
                model=self.settings.gemini.chat_model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    tools=[HEALTH_TOOLS],
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                ),
            )
            # A single turn can contain both a text part (the actual answer) and a
            # function_call part (e.g. logging that answer) — keep the text from every
            # round, since the final round's text is often just the trailing disclaimer.
            if resp.text:
                collected_text.append(resp.text)

            if not resp.function_calls:
                return "\n\n".join(collected_text)

            model_content = resp.candidates[0].content  # type: ignore[index]
            assert model_content is not None
            contents.append(model_content)

            for call in resp.function_calls:
                logger.info("Gemini tool call: {} args={}", call.name, call.args)
                fn = dispatch.get(call.name or "")
                result = fn(**(call.args or {})) if fn else {"error": f"unknown tool {call.name}"}
                contents.append(
                    types.Content(
                        role="tool",
                        parts=[
                            types.Part.from_function_response(
                                name=call.name or "unknown", response=result
                            )
                        ],
                    )
                )

        logger.warning("Max function-call rounds reached without a final answer")
        fallback = "I couldn't finish that after several tool calls — please rephrase or consult a clinician."
        return "\n\n".join([*collected_text, fallback])

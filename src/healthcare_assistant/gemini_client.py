from google import genai

from healthcare_assistant.config import Settings


def build_client(settings: Settings) -> genai.Client:
    return genai.Client(api_key=settings.google_api_key)

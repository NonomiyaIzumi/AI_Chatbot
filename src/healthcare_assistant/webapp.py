import uuid
from contextlib import asynccontextmanager
from importlib import resources

from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from loguru import logger
from pydantic import BaseModel

from healthcare_assistant.assistant import HealthcareAssistant
from healthcare_assistant.config import load_settings
from healthcare_assistant.db import init_db
from healthcare_assistant.embeddings import get_or_build_kb_embeddings
from healthcare_assistant.gemini_client import build_client
from healthcare_assistant.knowledge_base import load_knowledge_base
from healthcare_assistant.retrieval import Retriever

SESSION_COOKIE = "healthcare_assistant_session"


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


@asynccontextmanager
async def lifespan(app: FastAPI):
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
    app.state.assistant = HealthcareAssistant(client, retriever, settings)
    logger.info("Healthcare assistant web app ready")
    yield


app = FastAPI(title="Healthcare Assistant", lifespan=lifespan)


def _static_path(filename: str):
    return resources.files("healthcare_assistant").joinpath("web_static", filename)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(_static_path("index.html")))


@app.get("/app.js")
def app_js() -> FileResponse:
    return FileResponse(str(_static_path("app.js")), media_type="application/javascript")


@app.get("/style.css")
def style_css() -> FileResponse:
    return FileResponse(str(_static_path("style.css")), media_type="text/css")


@app.post("/api/chat")
def chat(payload: ChatRequest, request: Request, response: Response) -> ChatResponse:
    user_id = request.cookies.get(SESSION_COOKIE)
    if not user_id:
        user_id = str(uuid.uuid4())
        response.set_cookie(SESSION_COOKIE, user_id, httponly=True, samesite="lax")

    assistant: HealthcareAssistant = request.app.state.assistant
    reply = assistant.ask(user_id, payload.message)
    return ChatResponse(reply=reply)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error while serving {}", request.url.path)
    return JSONResponse(status_code=500, content={"error": "Something went wrong. Please try again."})


def run() -> None:
    import uvicorn

    uvicorn.run("healthcare_assistant.webapp:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    run()

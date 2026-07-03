from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Callable

from google.genai import types
from loguru import logger

from healthcare_assistant import db

LOG_SYMPTOM_CHECK_DECL = types.FunctionDeclaration(
    name="log_symptom_check",
    description=(
        "Record a symptom-check interaction for the current user: the symptoms reported "
        "and the condition/advice suggested. Call this after giving the user a suggested "
        "condition and advice."
    ),
    parameters_json_schema={
        "type": "object",
        "properties": {
            "symptoms": {
                "type": "string",
                "description": "Comma-separated symptoms, e.g. 'fever, sore throat'.",
            },
            "predicted_condition": {
                "type": "string",
                "description": "Likely condition suggested, e.g. 'Common cold'.",
            },
            "advice_given": {
                "type": "string",
                "description": "Advice given to the user.",
            },
        },
        "required": ["symptoms", "predicted_condition", "advice_given"],
    },
)

GET_PATIENT_HISTORY_DECL = types.FunctionDeclaration(
    name="get_patient_history",
    description="Retrieve the current user's most recent symptom-check history, if any.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 20,
                "description": "Maximum number of past entries to return.",
            },
        },
        "required": [],
    },
)

SCHEDULE_APPOINTMENT_DECL = types.FunctionDeclaration(
    name="schedule_appointment",
    description=(
        "Request a follow-up appointment with a human clinician for the current user. "
        "Use this for symptoms that are urgent or beyond general self-care advice."
    ),
    parameters_json_schema={
        "type": "object",
        "properties": {
            "reason": {"type": "string", "description": "Reason for the appointment request."},
            "preferred_date": {
                "type": "string",
                "description": "Preferred date in YYYY-MM-DD format, or 'unspecified'.",
            },
        },
        "required": ["reason", "preferred_date"],
    },
)

HEALTH_TOOLS = types.Tool(
    function_declarations=[
        LOG_SYMPTOM_CHECK_DECL,
        GET_PATIENT_HISTORY_DECL,
        SCHEDULE_APPOINTMENT_DECL,
    ]
)


def tool_log_symptom_check(
    *, user_id: str, db_path: Path, symptoms: str, predicted_condition: str, advice_given: str
) -> dict:
    timestamp = datetime.now(timezone.utc).isoformat()
    log_id = db.insert_symptom_log(
        db_path, user_id, timestamp, symptoms, predicted_condition, advice_given
    )
    logger.info("Logged symptom check id={} user={}", log_id, user_id)
    return {"status": "ok", "log_id": log_id}


def tool_get_patient_history(*, user_id: str, db_path: Path, limit: int = 5) -> dict:
    logs = db.fetch_symptom_logs(db_path, user_id, limit)
    logger.info("Fetched {} history entries for user={}", len(logs), user_id)
    return {"status": "ok", "history": logs}


def tool_schedule_appointment(
    *, user_id: str, db_path: Path, reason: str, preferred_date: str
) -> dict:
    requested_at = datetime.now(timezone.utc).isoformat()
    appointment_id = db.insert_appointment(db_path, user_id, requested_at, reason, preferred_date)
    logger.info("Scheduled appointment id={} user={}", appointment_id, user_id)
    return {"status": "ok", "appointment_id": appointment_id}


def build_dispatch(user_id: str, db_path: Path) -> dict[str, Callable[..., dict]]:
    return {
        "log_symptom_check": partial(tool_log_symptom_check, user_id=user_id, db_path=db_path),
        "get_patient_history": partial(tool_get_patient_history, user_id=user_id, db_path=db_path),
        "schedule_appointment": partial(tool_schedule_appointment, user_id=user_id, db_path=db_path),
    }

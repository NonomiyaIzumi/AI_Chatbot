import json
import sqlite3

from healthcare_assistant import db
from healthcare_assistant.tools import (
    tool_get_patient_history,
    tool_log_symptom_check,
    tool_schedule_appointment,
)


def test_init_db_creates_expected_tables(db_path):
    db.init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
    assert {"symptom_logs", "appointments"} <= tables


def test_symptom_log_round_trip_respects_limit_and_order(db_path):
    db.init_db(db_path)
    for i in range(3):
        db.insert_symptom_log(
            db_path, "user-1", f"2026-01-0{i + 1}T00:00:00", f"symptom-{i}", "Condition", "advice"
        )

    logs = db.fetch_symptom_logs(db_path, "user-1", limit=2)

    assert len(logs) == 2
    assert logs[0]["symptoms"] == "symptom-2"
    assert logs[1]["symptoms"] == "symptom-1"


def test_symptom_log_filters_by_user_id(db_path):
    db.init_db(db_path)
    db.insert_symptom_log(db_path, "user-1", "ts", "symptoms", "Condition", "advice")
    db.insert_symptom_log(db_path, "user-2", "ts", "other symptoms", "Condition", "advice")

    logs = db.fetch_symptom_logs(db_path, "user-1", limit=5)

    assert len(logs) == 1
    assert logs[0]["user_id"] == "user-1"


def test_appointment_round_trip(db_path):
    db.init_db(db_path)
    appointment_id = db.insert_appointment(db_path, "user-1", "2026-01-01T00:00:00", "fever", "2026-01-05")

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT user_id, reason, preferred_date, status FROM appointments WHERE id = ?",
            (appointment_id,),
        ).fetchone()

    assert row == ("user-1", "fever", "2026-01-05", "pending")


def test_tool_wrappers_return_json_serializable_dicts(db_path):
    db.init_db(db_path)

    log_result = tool_log_symptom_check(
        user_id="user-1",
        db_path=db_path,
        symptoms="cough, fever",
        predicted_condition="Common cold",
        advice_given="Rest and hydrate.",
    )
    history_result = tool_get_patient_history(user_id="user-1", db_path=db_path, limit=5)
    appointment_result = tool_schedule_appointment(
        user_id="user-1", db_path=db_path, reason="follow-up", preferred_date="unspecified"
    )

    for result in (log_result, history_result, appointment_result):
        json.dumps(result)  # must not raise

    assert log_result["status"] == "ok"
    assert history_result["history"][0]["predicted_condition"] == "Common cold"
    assert appointment_result["status"] == "ok"

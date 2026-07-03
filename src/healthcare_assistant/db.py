import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

_SCHEMA = """
CREATE TABLE IF NOT EXISTS symptom_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    symptoms TEXT NOT NULL,
    predicted_condition TEXT NOT NULL,
    advice_given TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS appointments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    requested_at TEXT NOT NULL,
    reason TEXT NOT NULL,
    preferred_date TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
);
"""


@contextmanager
def connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: Path) -> None:
    with connect(db_path) as conn:
        conn.executescript(_SCHEMA)
        conn.commit()


def insert_symptom_log(
    db_path: Path,
    user_id: str,
    timestamp: str,
    symptoms: str,
    predicted_condition: str,
    advice_given: str,
) -> int:
    with connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO symptom_logs (user_id, timestamp, symptoms, predicted_condition, advice_given)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, timestamp, symptoms, predicted_condition, advice_given),
        )
        conn.commit()
        assert cursor.lastrowid is not None
        return cursor.lastrowid


def fetch_symptom_logs(db_path: Path, user_id: str, limit: int = 5) -> list[dict]:
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, user_id, timestamp, symptoms, predicted_condition, advice_given
            FROM symptom_logs
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
        return [dict(row) for row in rows]


def insert_appointment(
    db_path: Path,
    user_id: str,
    requested_at: str,
    reason: str,
    preferred_date: str,
) -> int:
    with connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO appointments (user_id, requested_at, reason, preferred_date, status)
            VALUES (?, ?, ?, ?, 'pending')
            """,
            (user_id, requested_at, reason, preferred_date),
        )
        conn.commit()
        assert cursor.lastrowid is not None
        return cursor.lastrowid

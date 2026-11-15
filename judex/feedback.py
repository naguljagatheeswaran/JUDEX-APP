"""Local, review-first storage for Judex feedback."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class FeedbackRepository:
    """Persist user feedback locally and expose approved records for training."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialise()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialise(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    input_text TEXT NOT NULL,
                    suggested_section INTEGER,
                    selected_section INTEGER NOT NULL,
                    suggested_classifier TEXT,
                    selected_classifier TEXT,
                    classifier_verdict TEXT CHECK (classifier_verdict IN ('correct', 'incorrect')),
                    verdict TEXT NOT NULL CHECK (verdict IN ('correct', 'incorrect')),
                    status TEXT NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending', 'approved', 'rejected')),
                    created_at TEXT NOT NULL
                )
                """
            )
            # Keep existing local capstone databases compatible after the
            # feedback flow was extended from BNS-only to two-stage learning.
            existing = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(feedback)").fetchall()
            }
            for column in (
                "suggested_classifier TEXT",
                "selected_classifier TEXT",
                "classifier_verdict TEXT",
            ):
                name = column.split()[0]
                if name not in existing:
                    connection.execute(f"ALTER TABLE feedback ADD COLUMN {column}")

    def submit(
        self,
        input_text: str,
        suggested_section: int | None,
        selected_section: int,
        verdict: str,
        suggested_classifier: str,
        selected_classifier: str,
        classifier_verdict: str,
    ) -> None:
        if verdict not in {"correct", "incorrect"}:
            raise ValueError("verdict must be 'correct' or 'incorrect'")
        if classifier_verdict not in {"correct", "incorrect"}:
            raise ValueError("classifier_verdict must be 'correct' or 'incorrect'")
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO feedback
                    (input_text, suggested_section, selected_section, suggested_classifier,
                     selected_classifier, classifier_verdict, verdict, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    input_text.strip(), suggested_section, selected_section,
                    suggested_classifier, selected_classifier, classifier_verdict, verdict, now,
                ),
            )

    def pending(self) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return connection.execute(
                "SELECT * FROM feedback WHERE status = 'pending' ORDER BY id DESC"
            ).fetchall()

    def set_status(self, feedback_id: int, status: str) -> None:
        if status not in {"approved", "rejected"}:
            raise ValueError("status must be 'approved' or 'rejected'")
        with self._connect() as connection:
            connection.execute(
                "UPDATE feedback SET status = ? WHERE id = ?", (status, feedback_id)
            )

    def approved_training_examples(self) -> list[tuple[str, int]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT input_text, selected_section
                FROM feedback
                WHERE status = 'approved'
                ORDER BY id
                """
            ).fetchall()
        return [(row["input_text"], row["selected_section"]) for row in rows]

    def approved_classifier_examples(self) -> list[tuple[str, str]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT input_text, selected_classifier
                FROM feedback
                WHERE status = 'approved' AND selected_classifier IS NOT NULL
                ORDER BY id
                """
            ).fetchall()
        return [(row["input_text"], row["selected_classifier"]) for row in rows]

    def count_by_status(self, status: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM feedback WHERE status = ?", (status,)
            ).fetchone()
        return int(row["count"])

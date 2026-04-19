from __future__ import annotations

import logging
import os
import sqlite3
import threading
from typing import Iterable


def _normalize_name(name: str) -> str:
    text = str(name or "").strip()
    if not text:
        raise ValueError("SQLite table name must not be empty.")
    out = []
    for ch in text:
        out.append(ch if ch.isalnum() or ch == "_" else "_")
    normalized = "".join(out)
    if normalized[0].isdigit():
        normalized = f"t_{normalized}"
    return normalized


class SQLiteRunStore:
    def __init__(self, db_path: str):
        self.db_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA temp_store=MEMORY;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._tables: dict[str, list[str]] = {}

    def close(self) -> None:
        with self._lock:
            self._conn.commit()
            self._conn.close()

    def _infer_column_type(self, column: str) -> str:
        key = str(column).strip().lower()
        text_markers = (
            "name",
            "mode",
            "message",
            "level",
            "logger",
            "module",
            "path",
            "argv",
            "pythonpath",
            "cwd",
            "policy",
            "profile",
            "schedule",
            "type",
            "prompt",
            "stage",
            "case",
            "threshold_source",
        )
        return "TEXT" if any(marker in key for marker in text_markers) else "REAL"

    def ensure_table(self, name: str, columns: Iterable[str], column_types: dict[str, str] | None = None) -> str:
        table = _normalize_name(name)
        cols = [str(col) for col in columns]
        if not cols:
            raise ValueError(f"Table '{table}' must declare at least one column.")
        if table in self._tables:
            return table
        type_map = {str(k): str(v) for k, v in (column_types or {}).items()}
        col_defs = []
        for col in cols:
            sql_type = type_map.get(col, self._infer_column_type(col))
            col_defs.append(f'"{col}" {sql_type}')
        create_sql = f'CREATE TABLE IF NOT EXISTS "{table}" ({", ".join(col_defs)});'
        with self._lock:
            self._conn.execute(create_sql)
            self._conn.commit()
        self._tables[table] = cols
        return table

    def columns_for(self, name: str) -> list[str]:
        table = _normalize_name(name)
        cols = self._tables.get(table)
        if cols is None:
            raise KeyError(f"Table '{table}' has not been initialized.")
        return cols

    def insert_rows(self, name: str, rows: Iterable[Iterable]) -> None:
        table = _normalize_name(name)
        cols = self.columns_for(table)
        prepared_rows = [tuple(row) for row in rows]
        if not prepared_rows:
            return
        placeholders = ", ".join(["?"] * len(cols))
        columns_sql = ", ".join(f'"{col}"' for col in cols)
        sql = f'INSERT INTO "{table}" ({columns_sql}) VALUES ({placeholders})'
        with self._lock:
            self._conn.executemany(sql, prepared_rows)
            self._conn.commit()

    def insert_row(self, name: str, row: Iterable) -> None:
        self.insert_rows(name, [tuple(row)])


class SQLiteLogHandler(logging.Handler):
    def __init__(self, store: SQLiteRunStore, table_name: str = "logs", flush_every: int = 20):
        super().__init__()
        self.store = store
        self.table_name = self.store.ensure_table(
            table_name,
            ["created_at", "level", "logger", "message"],
            column_types={
                "created_at": "REAL",
                "level": "TEXT",
                "logger": "TEXT",
                "message": "TEXT",
            },
        )
        self.flush_every = max(1, int(flush_every))
        self._buffer: list[tuple] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._buffer.append(
                (
                    float(record.created),
                    str(record.levelname),
                    str(record.name),
                    str(record.getMessage()),
                )
            )
            if len(self._buffer) >= self.flush_every:
                self.flush()
        except Exception:
            self.handleError(record)

    def flush(self) -> None:
        if not self._buffer:
            return
        rows = list(self._buffer)
        self._buffer.clear()
        self.store.insert_rows(self.table_name, rows)

    def close(self) -> None:
        try:
            self.flush()
        finally:
            super().close()

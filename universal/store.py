import sqlite3
import threading

_local = threading.local()


def _conn() -> sqlite3.Connection:
    if not hasattr(_local, "conn"):
        _local.conn = sqlite3.connect("./knowledge.db", check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _init(_local.conn)
    return _local.conn


def _init(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS docs (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            name    TEXT NOT NULL UNIQUE,
            created_at DATETIME DEFAULT (datetime('now', 'localtime'))
        )
    """)
    conn.commit()


def insert_doc(name: str):
    conn = _conn()
    conn.execute("INSERT OR IGNORE INTO docs (name) VALUES (?)", (name,))
    conn.commit()


def exists_doc(name: str) -> bool:
    conn = _conn()
    row = conn.execute("SELECT 1 FROM docs WHERE name = ?", (name,)).fetchone()
    return row is not None


def list_docs() -> list[dict]:
    conn = _conn()
    rows = conn.execute("SELECT id, name, created_at FROM docs ORDER BY created_at DESC").fetchall()
    return [dict(row) for row in rows]


def delete_doc(name: str) -> bool:
    conn = _conn()
    cur = conn.execute("DELETE FROM docs WHERE name = ?", (name,))
    conn.commit()
    return cur.rowcount > 0

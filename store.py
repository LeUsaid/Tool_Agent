"""App-level tables. LangGraph's checkpointer stores the full conversation
state itself; we only add a small `conversations` index (for --list / titles)
and a `notes` table that one of our custom tools writes to."""
import sqlite3

DB_PATH = "agent.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    title      TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS notes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    text       TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

_conn = None


def get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.executescript(SCHEMA)
        _conn.commit()
    return _conn


def create_conversation() -> int:
    conn = get_conn()
    cur = conn.execute("INSERT INTO conversations (title) VALUES (NULL)")
    conn.commit()
    return cur.lastrowid


def set_title(conv_id: int, title: str) -> None:
    conn = get_conn()
    conn.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conv_id))
    conn.commit()


def list_conversations(limit: int = 20):
    return get_conn().execute(
        "SELECT id, title, created_at FROM conversations ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()


def add_note(text: str) -> int:
    conn = get_conn()
    cur = conn.execute("INSERT INTO notes (text) VALUES (?)", (text,))
    conn.commit()
    return cur.lastrowid


def search_notes(query: str, limit: int = 10):
    return get_conn().execute(
        "SELECT id, text, created_at FROM notes WHERE text LIKE ? ORDER BY id DESC LIMIT ?",
        (f"%{query}%", limit),
    ).fetchall()

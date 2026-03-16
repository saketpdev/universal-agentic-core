import sqlite3
import json
import logging
import os
from typing import Dict, Optional

logger = logging.getLogger("AgenticCore.Memory")

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "sessions.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS threads (
            thread_id TEXT PRIMARY KEY,
            user_id TEXT,
            state_data TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def load_history(thread_id: str) -> Optional[Dict]:
    """Loads the saved Briefcase state. Returns None if new thread."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT state_data FROM threads WHERE thread_id = ?", (thread_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        logger.info(f"Memory: Restored existing Briefcase for thread '{thread_id}'")
        return json.loads(row[0])
    
    logger.info(f"Memory: No existing state for thread '{thread_id}'")
    return None

def save_history(thread_id: str, user_id: str, state_data: Dict):
    """Upserts the current Briefcase state into SQLite."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    state_json = json.dumps(state_data)
    
    cursor.execute('''
        INSERT INTO threads (thread_id, user_id, state_data)
        VALUES (?, ?, ?)
        ON CONFLICT(thread_id) DO UPDATE SET 
            state_data = excluded.state_data,
            updated_at = CURRENT_TIMESTAMP
    ''', (thread_id, user_id, state_json))
    
    conn.commit()
    conn.close()

init_db()
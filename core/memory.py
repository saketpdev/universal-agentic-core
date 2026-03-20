import os
import logging
from typing import Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models.state import SharedBriefcase
from models.db_models import Base, ThreadRecord, StepLogRecord, HumanReviewRecord

logger = logging.getLogger("AgenticCore.Memory")

# Keep your original DB path logic
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "sessions.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

# SQLAlchemy Setup
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class SessionManager:
    def __init__(self):
        self._init_db()

    def _init_db(self):
        """Bootstraps the new SQLAlchemy Schema safely."""
        Base.metadata.create_all(bind=engine)
        logger.info("Memory: SQLAlchemy DB Schema initialized.")

    def get_briefcase(self, thread_id: str) -> Optional[SharedBriefcase]:
        """Wakes up the Briefcase from the database."""
        with SessionLocal() as db:
            record = db.query(ThreadRecord).filter(ThreadRecord.thread_id == thread_id).first()
            if record:
                logger.info(f"Memory: Restored existing Briefcase for thread '{thread_id}'")
                return SharedBriefcase.model_validate_json(record.briefcase_json) # type: ignore
            
            logger.info(f"Memory: No existing state for thread '{thread_id}'")
            return None

    def save_briefcase(self, thread_id: str, user_id: str, briefcase: SharedBriefcase, status: str = "RUNNING"):
        """Persists the Pydantic Briefcase."""
        with SessionLocal() as db:
            record = db.query(ThreadRecord).filter(ThreadRecord.thread_id == thread_id).first()
            
            if record:
                # Update existing (Replicates your ON CONFLICT DO UPDATE)
                record.briefcase_json = briefcase.model_dump_json() # type: ignore
                record.status = status # type: ignore
            else:
                # Insert new
                new_thread = ThreadRecord(
                    thread_id=thread_id,
                    user_id=user_id,
                    status=status,
                    briefcase_json=briefcase.model_dump_json()
                )
                db.add(new_thread)
                
            db.commit()

    def log_raw_output(self, thread_id: str, step_index: int, agent_id: str, raw_output: str, cost_usd: float = 0.0):
        """THE VAULT: Saves the massive token outputs strictly for humans/dashboards."""
        with SessionLocal() as db:
            log_entry = StepLogRecord(
                thread_id=thread_id,
                step_index=step_index,
                agent_id=agent_id,
                raw_output=raw_output,
                cost_usd=cost_usd
            )
            db.add(log_entry)
            db.commit()

    def create_review_ticket(self, thread_id: str, review_type: str, message: str):
        with SessionLocal() as db:
            ticket = HumanReviewRecord(
                thread_id=thread_id,
                review_type=review_type,
                message=message
            )
            # Pause the main thread's status so the user knows it's waiting
            db.query(ThreadRecord).filter(ThreadRecord.thread_id == thread_id).update({"status": "PAUSED_FOR_REVIEW"})

            db.add(ticket)
            db.commit()

# Export the singleton instance
session_manager = SessionManager()
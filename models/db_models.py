from sqlalchemy import Column, String, Text, Integer, Float, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()

class ThreadRecord(Base):
    __tablename__ = 'threads'

    thread_id = Column(String, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False)  # Preserved from your original schema
    status = Column(String, default="PENDING")
    briefcase_json = Column(Text, nullable=False) 
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())

class StepLogRecord(Base):
    __tablename__ = 'step_logs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    thread_id = Column(String, ForeignKey('threads.thread_id'), nullable=False, index=True)
    step_index = Column(Integer, nullable=False)
    agent_id = Column(String, nullable=False)
    raw_output = Column(Text, nullable=False)
    cost_usd = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class HumanReviewRecord(Base):
    __tablename__ = 'human_reviews'

    # 1. Routing Keys
    id = Column(Integer, primary_key=True, autoincrement=True)
    thread_id = Column(String, ForeignKey('threads.thread_id'), nullable=False, index=True)
    
    # 2. Context & State
    status = Column(String, default="PENDING", index=True) # PENDING, IN_REVIEW, APPROVED, REJECTED
    review_type = Column(String, nullable=False, index=True) # FINOPS, SECURITY, LOGIC
    message = Column(Text, nullable=False)
    
    # 3. SOC2 Audit Trail
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    resolved_by = Column(String, nullable=True) # Admin User ID
    resolution_notes = Column(Text, nullable=True)
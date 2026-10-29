"""
SQLAlchemy ORM Entities for Ticket Persistence and Audit Trails.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, Float, Boolean, Text, DateTime
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class TicketEntity(Base):
    __tablename__ = "tickets"

    id = Column(String(64), primary_key=True, index=True)
    ticket_id = Column(String(64), unique=True, index=True)
    source = Column(String(32), default="portal")
    customer_email = Column(String(255), index=True)
    subject = Column(String(512))
    body = Column(Text)
    status = Column(String(32), default="NEW", index=True)  # NEW, TRIAGED, ESCALATED, RESOLVED, QUARANTINED
    
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    triaged_at = Column(DateTime, nullable=True)
    assigned_team = Column(String(64), nullable=True)
    sla_target_minutes = Column(Integer, nullable=True)
    sla_deadline = Column(DateTime, nullable=True)

    # Classification fields
    intent = Column(String(64), nullable=True)
    priority = Column(String(16), nullable=True, index=True)
    urgency_score = Column(Float, nullable=True)
    sentiment = Column(String(32), nullable=True)
    confidence = Column(Float, nullable=True)
    reasoning = Column(Text, nullable=True)

    # Security & Guardrail
    threat_level = Column(String(32), default="CLEAN")
    is_safe = Column(Boolean, default=True)
    escalation_reason = Column(Text, nullable=True)
    latency_ms = Column(Float, default=0.0)

    # JSON Serialized Blobs for Deep Inspection
    guardrail_json = Column(Text, nullable=True)
    customer_json = Column(Text, nullable=True)
    classification_json = Column(Text, nullable=True)
    tool_calls_json = Column(Text, nullable=True)
    kb_citations_json = Column(Text, nullable=True)
    draft_reply_text = Column(Text, nullable=True)
    draft_reply_status = Column(String(32), nullable=True)

class AuditLogEntity(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticket_id = Column(String(64), index=True)
    event_type = Column(String(64), index=True)
    details_json = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))

"""
Pydantic Schemas for Tickets, Triage Pipeline, Tool Results, and Webhooks.
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import uuid
from pydantic import BaseModel, Field
from app.security.guardrails import GuardrailAssessment

class CustomerContext(BaseModel):
    customer_id: Optional[str] = None
    name: Optional[str] = "Unknown Customer"
    email_domain: Optional[str] = None
    tier: str = "Free"  # "Enterprise", "Scale", "Starter", "Free"
    arr: float = 0.0
    sla_tier: str = "P3_24h"
    dedicated_csm: Optional[str] = None
    open_tickets_count: int = 0
    account_status: str = "Active"
    vip_status: bool = False

class KBCitation(BaseModel):
    article_title: str
    filename: str
    relevance_score: float
    snippet: str

class ToolCallRecord(BaseModel):
    tool_name: str
    input_params: Dict[str, Any] = Field(default_factory=dict)
    output_summary: str
    latency_ms: float = 0.0

class ClassificationResult(BaseModel):
    intent: str = "General Inquiry"
    priority: str = "P3"  # "P0", "P1", "P2", "P3"
    urgency_score: float = Field(default=0.5, ge=0.0, le=1.0)
    sentiment: str = "NEUTRAL"  # "POSITIVE", "NEUTRAL", "FRUSTRATED", "ANGRY"
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    reasoning: str = ""
    classifier_type: str = "zero-shot-llm"  # or "fine-tuned"
    token_usage: Dict[str, int] = Field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0})

class DraftReply(BaseModel):
    reply_text: str
    cited_articles: List[str] = Field(default_factory=list)
    tone: str = "Empathetic & Professional"
    status: str = "PENDING_APPROVAL"  # "PENDING_APPROVAL", "APPROVED", "SENT"

class TicketCreate(BaseModel):
    ticket_id: Optional[str] = None
    source: str = "portal"  # "zendesk", "slack", "email", "portal", "api"
    customer_email: str
    subject: str
    body: str
    raw_metadata: Dict[str, Any] = Field(default_factory=dict)

class TicketResponse(BaseModel):
    id: str
    ticket_id: str
    source: str
    customer_email: str
    subject: str
    body: str
    status: str = "NEW"  # "NEW", "TRIAGED", "ESCALATED", "RESOLVED", "QUARANTINED"
    created_at: str
    triaged_at: Optional[str] = None
    assigned_team: Optional[str] = None
    sla_target_minutes: Optional[int] = None
    sla_deadline: Optional[str] = None
    
    # Triage Artifacts
    guardrail: Optional[GuardrailAssessment] = None
    customer: Optional[CustomerContext] = None
    classification: Optional[ClassificationResult] = None
    tool_calls: List[ToolCallRecord] = Field(default_factory=list)
    kb_citations: List[KBCitation] = Field(default_factory=list)
    draft_reply: Optional[DraftReply] = None
    escalation_reason: Optional[str] = None
    latency_ms: float = 0.0

# Webhook Payloads
class ZendeskWebhookPayload(BaseModel):
    ticket: Dict[str, Any]

class SlackWebhookPayload(BaseModel):
    event: Optional[Dict[str, Any]] = None
    command: Optional[str] = None
    text: Optional[str] = None
    user_name: Optional[str] = None
    channel_name: Optional[str] = None
    user_email: Optional[str] = None

class EmailWebhookPayload(BaseModel):
    from_email: str
    to_email: str
    subject: str
    body_text: str
    headers: Dict[str, str] = Field(default_factory=dict)

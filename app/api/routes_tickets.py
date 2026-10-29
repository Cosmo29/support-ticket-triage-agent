"""
Ticket Ingestion, Querying, and Lifecycle Management API Routes.
"""

from typing import List, Optional, Dict, Any
import json
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.core.database import get_db
from app.models.ticket import TicketCreate, TicketResponse
from app.models.entities import TicketEntity, AuditLogEntity
from app.pipeline.triage_agent import triage_agent
from app.connectors.slack import slack_connector
from app.core.logging import logger

router = APIRouter(prefix="/api/tickets", tags=["Tickets"])

@router.post("/intake", response_model=TicketResponse)
async def intake_ticket(
    ticket: TicketCreate,
    background_tasks: BackgroundTasks,
    use_fine_tuned: bool = Query(False, description="Use fine-tuned classifier instead of zero-shot"),
    db: Session = Depends(get_db)
):
    """
    Submits a ticket to the triage agent pipeline.
    Executes guardrails, tool calling, classification, SLA computation, and routing.
    """
    response = triage_agent.process_ticket(ticket, db=db, use_fine_tuned=use_fine_tuned)
    
    # If ticket was escalated or quarantined, dispatch Slack alert in background
    if response.status in ("ESCALATED", "QUARANTINED"):
        background_tasks.add_task(slack_connector.post_escalation_alert, response)
        
    return response

@router.get("/", response_model=List[Dict[str, Any]])
def list_tickets(
    status: Optional[str] = Query(None, description="Filter by status (NEW, TRIAGED, ESCALATED, RESOLVED, QUARANTINED)"),
    priority: Optional[str] = Query(None, description="Filter by priority (P0, P1, P2, P3)"),
    limit: int = Query(50, le=200),
    db: Session = Depends(get_db)
):
    """
    Lists tickets ordered by newest first with optional status/priority filtering.
    """
    query = db.query(TicketEntity)
    if status:
        query = query.filter(TicketEntity.status == status.upper())
    if priority:
        query = query.filter(TicketEntity.priority == priority.upper())
        
    entities = query.order_by(desc(TicketEntity.created_at)).limit(limit).all()
    
    results = []
    for e in entities:
        results.append({
            "id": e.id,
            "ticket_id": e.ticket_id,
            "source": e.source,
            "customer_email": e.customer_email,
            "subject": e.subject,
            "body": e.body,
            "status": e.status,
            "created_at": e.created_at.isoformat() if e.created_at else None,
            "assigned_team": e.assigned_team,
            "intent": e.intent,
            "priority": e.priority,
            "confidence": e.confidence,
            "threat_level": e.threat_level,
            "sla_target_minutes": e.sla_target_minutes,
            "escalation_reason": e.escalation_reason,
            "latency_ms": e.latency_ms
        })
    return results

@router.get("/{ticket_id}", response_model=Dict[str, Any])
def get_ticket_detail(ticket_id: str, db: Session = Depends(get_db)):
    """
    Retrieves full details and triage trace for a specific ticket.
    """
    entity = db.query(TicketEntity).filter(
        (TicketEntity.ticket_id == ticket_id) | (TicketEntity.id == ticket_id)
    ).first()
    
    if not entity:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found")
        
    return {
        "id": entity.id,
        "ticket_id": entity.ticket_id,
        "source": entity.source,
        "customer_email": entity.customer_email,
        "subject": entity.subject,
        "body": entity.body,
        "status": entity.status,
        "created_at": entity.created_at.isoformat() if entity.created_at else None,
        "triaged_at": entity.triaged_at.isoformat() if entity.triaged_at else None,
        "assigned_team": entity.assigned_team,
        "sla_target_minutes": entity.sla_target_minutes,
        "sla_deadline": entity.sla_deadline.isoformat() if entity.sla_deadline else None,
        "intent": entity.intent,
        "priority": entity.priority,
        "urgency_score": entity.urgency_score,
        "sentiment": entity.sentiment,
        "confidence": entity.confidence,
        "reasoning": entity.reasoning,
        "threat_level": entity.threat_level,
        "is_safe": entity.is_safe,
        "escalation_reason": entity.escalation_reason,
        "latency_ms": entity.latency_ms,
        "guardrail": json.loads(entity.guardrail_json) if entity.guardrail_json else None,
        "customer": json.loads(entity.customer_json) if entity.customer_json else None,
        "classification": json.loads(entity.classification_json) if entity.classification_json else None,
        "tool_calls": json.loads(entity.tool_calls_json) if entity.tool_calls_json else [],
        "kb_citations": json.loads(entity.kb_citations_json) if entity.kb_citations_json else [],
        "draft_reply": {
            "reply_text": entity.draft_reply_text,
            "status": entity.draft_reply_status
        } if entity.draft_reply_text else None
    }

@router.post("/{ticket_id}/reply")
def approve_and_send_reply(ticket_id: str, db: Session = Depends(get_db)):
    """
    Approves AI draft reply and marks ticket as RESOLVED.
    """
    entity = db.query(TicketEntity).filter(TicketEntity.ticket_id == ticket_id).first()
    if not entity:
        raise HTTPException(status_code=404, detail="Ticket not found")
        
    entity.status = "RESOLVED"
    entity.draft_reply_status = "SENT"
    db.commit()
    return {"status": "success", "ticket_id": ticket_id, "new_status": "RESOLVED"}

@router.post("/seed-demo")
def seed_demo_tickets(count: int = Query(5, le=25), db: Session = Depends(get_db)):
    """
    Seeds a batch of sample realistic tickets through the triage pipeline for immediate testing.
    """
    import os
    test_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../evals/test_dataset.json"))
    with open(test_file, "r", encoding="utf-8") as f:
        samples = json.load(f)[:count]

    created = []
    for s in samples:
        t = TicketCreate(
            ticket_id=s["id"],
            source="portal",
            customer_email="user@acmefinancial.com" if "outage" in s["subject"].lower() else "client@example.com",
            subject=s["subject"],
            body=s["body"]
        )
        res = triage_agent.process_ticket(t, db=db)
        created.append(res.ticket_id)
        
    return {"status": "success", "seeded_count": len(created), "ticket_ids": created}

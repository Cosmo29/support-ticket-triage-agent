"""
Webhook Ingestion Endpoints for Zendesk, Slack, and Email.
"""

from typing import Dict, Any
from fastapi import APIRouter, Request, BackgroundTasks, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.connectors.zendesk import zendesk_connector
from app.connectors.slack import slack_connector
from app.connectors.email_parser import parse_inbound_email
from app.pipeline.triage_agent import triage_agent
from app.core.logging import logger

router = APIRouter(prefix="/webhooks", tags=["Webhooks"])

@router.post("/zendesk")
async def zendesk_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Receives Zendesk ticket webhook trigger, runs agent triage,
    and updates Zendesk sandbox in the background.
    """
    payload = await request.json()
    ticket_input = zendesk_connector.parse_webhook_payload(payload)
    
    # Process through Agent
    response = triage_agent.process_ticket(ticket_input, db=db)
    
    # Sync back to Zendesk Sandbox
    zendesk_id = ticket_input.raw_metadata.get("zendesk_id")
    if zendesk_id:
        tags = [
            f"intent_{response.classification.intent.lower().replace(' ', '_')}" if response.classification else "intent_general",
            f"priority_{response.classification.priority.lower()}" if response.classification else "priority_p3",
            f"status_{response.status.lower()}"
        ]
        internal_note = (
            f"[AI Triage Summary]\n"
            f"Assigned Team: {response.assigned_team}\n"
            f"Confidence: {response.classification.confidence if response.classification else 'N/A'}\n"
            f"Reasoning: {response.classification.reasoning if response.classification else 'N/A'}\n"
            f"Escalation: {response.escalation_reason or 'None'}"
        )
        background_tasks.add_task(
            zendesk_connector.update_zendesk_ticket,
            zendesk_id=str(zendesk_id),
            priority=response.classification.priority if response.classification else "normal",
            tags=tags,
            internal_note=internal_note
        )

    return {
        "status": "received",
        "ticket_id": response.ticket_id,
        "assigned_team": response.assigned_team,
        "priority": response.classification.priority if response.classification else "P3"
    }

@router.post("/slack")
async def slack_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Receives Slack slash command (e.g., /triage) or interactive event webhook.
    """
    content_type = request.headers.get("content-type", "")
    if "application/x-www-form-urlencoded" in content_type:
        form = await request.form()
        payload = dict(form)
    else:
        payload = await request.json()
        
    ticket_input = slack_connector.parse_slack_payload(payload)
    response = triage_agent.process_ticket(ticket_input, db=db)
    
    # Post alert back if escalated
    if response.status in ("ESCALATED", "QUARANTINED"):
        background_tasks.add_task(slack_connector.post_escalation_alert, response)
        
    return {
        "response_type": "in_channel",
        "text": f"✅ Ticket *{response.ticket_id}* triaged to *{response.assigned_team}* ({response.classification.priority if response.classification else 'P3'}). Status: {response.status}"
    }

@router.post("/email")
async def email_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Receives parsed inbound email from SendGrid / Mailgun webhook.
    """
    content_type = request.headers.get("content-type", "")
    if "application/x-www-form-urlencoded" in content_type or "multipart/form-data" in content_type:
        form = await request.form()
        data = dict(form)
    else:
        data = await request.json()
        
    ticket_input = parse_inbound_email(data)
    response = triage_agent.process_ticket(ticket_input, db=db)
    
    return {
        "status": "processed",
        "ticket_id": response.ticket_id,
        "assigned_team": response.assigned_team,
        "status": response.status
    }

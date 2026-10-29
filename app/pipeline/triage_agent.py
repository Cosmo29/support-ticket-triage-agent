"""
Agent Orchestrator: End-to-End Support Ticket Triage Pipeline.
Executes the sequential agentic workflow:
1. Prompt injection defense & sanitization
2. Tool call: Customer CRM profile enrichment
3. Multi-dimensional classification (Intent, Priority, Urgency, Sentiment, Confidence)
4. Tool call: Knowledge base retrieval (grounding runbooks)
5. SLA target and deadline computation
6. Escalation logic (P0, VIP, low confidence < 0.75, or security containment)
7. Tool call: Grounded draft reply generation
8. Structured logging, metrics recording, and SQLite persistence
"""

from __future__ import annotations
import time
import json
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session

from app.models.ticket import (
    TicketCreate,
    TicketResponse,
    CustomerContext,
    ClassificationResult,
    GuardrailAssessment,
    ToolCallRecord,
    KBCitation,
    DraftReply
)
from app.models.entities import TicketEntity, AuditLogEntity
from app.security.guardrails import inspect_for_prompt_injection
from app.tools.customer_lookup import customer_lookup
from app.tools.kb_search import kb_search
from app.tools.escalate import evaluate_escalation
from app.tools.draft_reply import draft_reply
from app.pipeline.classifiers import TfidfCalibratedBaseline, ZeroShotLLMClassifier
from app.core.logging import logger
from app.config import settings

class TriageAgent:
    def __init__(self):
        self.tfidf_baseline = TfidfCalibratedBaseline()
        self.fine_tuned_classifier = self.tfidf_baseline  # Backward compatibility alias
        self.zero_shot_classifier = ZeroShotLLMClassifier()

    def process_ticket(
        self,
        ticket_input: TicketCreate,
        db: Optional[Session] = None,
        use_baseline: bool = False,
        use_fine_tuned: bool = False
    ) -> TicketResponse:
        """
        Executes end-to-end triage on an incoming support ticket.
        """
        start_overall = time.time()
        ticket_id = ticket_input.ticket_id or f"TICK-{uuid.uuid4().hex[:6].upper()}"
        internal_id = str(uuid.uuid4())
        tool_calls: List[ToolCallRecord] = []
        now_utc = datetime.now(timezone.utc)

        logger.info(
            f"Starting triage for ticket {ticket_id}",
            extra={"ticket_id": ticket_id, "action": "TRIAGE_START", "source": ticket_input.source}
        )

        # -------------------------------------------------------------
        # Step 1: Prompt Injection & Guardrail Assessment
        # -------------------------------------------------------------
        combined_text = f"{ticket_input.subject}\n{ticket_input.body}"
        guardrail: GuardrailAssessment = inspect_for_prompt_injection(combined_text)
        
        # -------------------------------------------------------------
        # Step 2: Tool Call - Customer CRM Lookup
        # -------------------------------------------------------------
        customer, cust_tool_record = customer_lookup(ticket_input.customer_email)
        tool_calls.append(cust_tool_record)

        # If malicious prompt injection detected: QUARANTINE IMMEDIATELY
        if not guardrail.is_safe and guardrail.threat_level == "MALICIOUS":
            latency_ms = round((time.time() - start_overall) * 1000, 2)
            classification = ClassificationResult(
                intent="Security Threat",
                priority="P0",
                urgency_score=1.0,
                sentiment="ANGRY",
                confidence=0.99,
                reasoning=f"Quarantined due to adversarial injection: {', '.join(guardrail.detected_patterns)}",
                classifier_type="guardrail-shield"
            )
            
            response = TicketResponse(
                id=internal_id,
                ticket_id=ticket_id,
                source=ticket_input.source,
                customer_email=ticket_input.customer_email,
                subject=ticket_input.subject,
                body=ticket_input.body,
                status="QUARANTINED",
                created_at=now_utc.isoformat(),
                triaged_at=now_utc.isoformat(),
                assigned_team="Security Operations & Trust",
                sla_target_minutes=15,
                sla_deadline=(now_utc + timedelta(minutes=15)).isoformat(),
                guardrail=guardrail,
                customer=customer,
                classification=classification,
                tool_calls=tool_calls,
                kb_citations=[],
                draft_reply=None,
                escalation_reason=f"Adversarial input detected: {', '.join(guardrail.detected_patterns)}",
                latency_ms=latency_ms
            )
            self._save_ticket(response, db)
            return response

        # -------------------------------------------------------------
        # Step 3: Multi-Dimensional Classification
        # -------------------------------------------------------------
        use_base = use_baseline or use_fine_tuned
        classifier = self.tfidf_baseline if use_base else self.zero_shot_classifier
        classification = classifier.classify(ticket_input.subject, ticket_input.body)

        # -------------------------------------------------------------
        # Step 4: Tool Call - Knowledge Base Retrieval
        # -------------------------------------------------------------
        kb_query = f"{classification.intent} {ticket_input.subject} {ticket_input.body[:150]}"
        citations, kb_tool_record = kb_search(kb_query, top_k=2)
        tool_calls.append(kb_tool_record)

        # -------------------------------------------------------------
        # Step 5: SLA Calculation
        # -------------------------------------------------------------
        sla_target_minutes = self._calculate_sla(classification.priority, customer.tier)
        sla_deadline = now_utc + timedelta(minutes=sla_target_minutes)

        # -------------------------------------------------------------
        # Step 6: Tool Call - Escalation Evaluation
        # -------------------------------------------------------------
        is_escalated, assigned_team, escalation_reason, esc_tool_record = evaluate_escalation(
            ticket_id=ticket_id,
            guardrail=guardrail,
            customer=customer,
            classification=classification
        )
        tool_calls.append(esc_tool_record)

        # -------------------------------------------------------------
        # Step 7: Tool Call - Draft Grounded Reply
        # -------------------------------------------------------------
        generated_draft, draft_tool_record = draft_reply(
            ticket_id=ticket_id,
            subject=ticket_input.subject,
            body=ticket_input.body,
            customer=customer,
            classification=classification,
            citations=citations
        )
        tool_calls.append(draft_tool_record)

        # Status determination
        if is_escalated:
            status = "ESCALATED"
        else:
            status = "TRIAGED"

        total_latency_ms = round((time.time() - start_overall) * 1000, 2)

        response = TicketResponse(
            id=internal_id,
            ticket_id=ticket_id,
            source=ticket_input.source,
            customer_email=ticket_input.customer_email,
            subject=ticket_input.subject,
            body=ticket_input.body,
            status=status,
            created_at=now_utc.isoformat(),
            triaged_at=now_utc.isoformat(),
            assigned_team=assigned_team,
            sla_target_minutes=sla_target_minutes,
            sla_deadline=sla_deadline.isoformat(),
            guardrail=guardrail,
            customer=customer,
            classification=classification,
            tool_calls=tool_calls,
            kb_citations=citations,
            draft_reply=generated_draft,
            escalation_reason=escalation_reason,
            latency_ms=total_latency_ms
        )

        logger.info(
            f"Triage complete for {ticket_id}: Status={status}, Team={assigned_team}, Intent={classification.intent}, Priority={classification.priority}",
            extra={
                "ticket_id": ticket_id,
                "intent": classification.intent,
                "priority": classification.priority,
                "confidence": classification.confidence,
                "latency_ms": total_latency_ms,
                "action": "TRIAGE_COMPLETE"
            }
        )

        self._save_ticket(response, db)
        return response

    def _calculate_sla(self, priority: str, tier: str) -> int:
        """Computes target response time in minutes based on priority and tier."""
        if priority == "P0":
            return 15 if tier == "Enterprise" else 30
        elif priority == "P1":
            return 60 if tier == "Enterprise" else 120
        elif priority == "P2":
            return 240  # 4 hours
        else:
            return 1440  # 24 hours

    def _save_ticket(self, ticket: TicketResponse, db: Optional[Session]):
        """Persists ticket and audit log into SQLite if DB session is provided."""
        if db is None:
            return
        try:
            entity = TicketEntity(
                id=ticket.id,
                ticket_id=ticket.ticket_id,
                source=ticket.source,
                customer_email=ticket.customer_email,
                subject=ticket.subject,
                body=ticket.body,
                status=ticket.status,
                created_at=datetime.fromisoformat(ticket.created_at),
                triaged_at=datetime.fromisoformat(ticket.triaged_at) if ticket.triaged_at else None,
                assigned_team=ticket.assigned_team,
                sla_target_minutes=ticket.sla_target_minutes,
                sla_deadline=datetime.fromisoformat(ticket.sla_deadline) if ticket.sla_deadline else None,
                intent=ticket.classification.intent if ticket.classification else None,
                priority=ticket.classification.priority if ticket.classification else None,
                urgency_score=ticket.classification.urgency_score if ticket.classification else None,
                sentiment=ticket.classification.sentiment if ticket.classification else None,
                confidence=ticket.classification.confidence if ticket.classification else None,
                reasoning=ticket.classification.reasoning if ticket.classification else None,
                threat_level=ticket.guardrail.threat_level if ticket.guardrail else "CLEAN",
                is_safe=ticket.guardrail.is_safe if ticket.guardrail else True,
                escalation_reason=ticket.escalation_reason,
                latency_ms=ticket.latency_ms,
                guardrail_json=ticket.guardrail.model_dump_json() if ticket.guardrail else None,
                customer_json=ticket.customer.model_dump_json() if ticket.customer else None,
                classification_json=ticket.classification.model_dump_json() if ticket.classification else None,
                tool_calls_json=json.dumps([t.model_dump() for t in ticket.tool_calls]),
                kb_citations_json=json.dumps([c.model_dump() for c in ticket.kb_citations]),
                draft_reply_text=ticket.draft_reply.reply_text if ticket.draft_reply else None,
                draft_reply_status=ticket.draft_reply.status if ticket.draft_reply else None
            )
            db.merge(entity)
            
            # Record audit log
            audit = AuditLogEntity(
                ticket_id=ticket.ticket_id,
                event_type=f"TRIAGE_{ticket.status}",
                details_json=json.dumps({
                    "assigned_team": ticket.assigned_team,
                    "confidence": ticket.classification.confidence if ticket.classification else 0.0,
                    "escalation_reason": ticket.escalation_reason,
                    "latency_ms": ticket.latency_ms
                }),
                timestamp=datetime.now(timezone.utc)
            )
            db.add(audit)
            db.commit()
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to persist ticket {ticket.ticket_id} to DB: {e}", exc_info=True)

# Global Agent Singleton
triage_agent = TriageAgent()

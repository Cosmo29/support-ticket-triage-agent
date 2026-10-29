"""
Human Escalation Tool.
Handles escalation when confidence is low, P0 outages occur, or security threats are detected.
"""

import time
from typing import Optional, Tuple
from app.models.ticket import (
    CustomerContext,
    ClassificationResult,
    GuardrailAssessment,
    ToolCallRecord
)
from app.core.logging import logger
from app.config import settings

def evaluate_escalation(
    ticket_id: str,
    guardrail: GuardrailAssessment,
    customer: CustomerContext,
    classification: ClassificationResult
) -> Tuple[bool, Optional[str], Optional[str], ToolCallRecord]:
    """
    Evaluates whether a ticket requires human escalation or containment.
    Returns: (is_escalated, assigned_team, escalation_reason, tool_record)
    """
    start_time = time.time()
    is_escalated = False
    assigned_team = "General Support"
    reason = None
    
    # 1. Security Threat Quarantine
    if not guardrail.is_safe or guardrail.threat_level in ("MALICIOUS", "SUSPICIOUS"):
        is_escalated = True
        assigned_team = "Security Operations & Trust"
        reason = f"Security Alert: {guardrail.threat_level} threat detected ({', '.join(guardrail.detected_patterns or ['Suspicious text'])})"
        
    # 2. P0 Blocker / Outage
    elif classification.priority == "P0" and settings.AUTO_ESCALATE_P0:
        is_escalated = True
        assigned_team = "SRE & Incident Response"
        reason = "P0 Critical Outage detected - Urgent incident response required (< 15m SLA)"
        
    # 3. Enterprise VIP Escalation
    elif customer.vip_status and classification.priority in ("P0", "P1"):
        is_escalated = True
        assigned_team = "Enterprise VIP Support"
        csm_info = f" (Dedicated CSM: {customer.dedicated_csm})" if customer.dedicated_csm else ""
        reason = f"Enterprise Tier client ({customer.name}, ${customer.arr:,.0f} ARR) high priority request{csm_info}"
        
    # 4. Low AI Confidence (< threshold)
    elif classification.confidence < settings.CONFIDENCE_THRESHOLD:
        is_escalated = True
        assigned_team = _map_team_from_intent(classification.intent)
        reason = f"AI confidence score ({classification.confidence:.2f}) below operational threshold ({settings.CONFIDENCE_THRESHOLD:.2f})"
        
    # Standard non-escalated routing
    else:
        assigned_team = _map_team_from_intent(classification.intent)

    latency_ms = round((time.time() - start_time) * 1000, 2)
    record = ToolCallRecord(
        tool_name="escalate_evaluator",
        input_params={"ticket_id": ticket_id, "confidence": classification.confidence, "priority": classification.priority},
        output_summary=f"Escalated: {is_escalated} -> {assigned_team} ({reason or 'Standard Routing'})",
        latency_ms=latency_ms
    )
    
    return is_escalated, assigned_team, reason, record

def _map_team_from_intent(intent: str) -> str:
    mapping = {
        # Standardized Public Dataset Taxonomy
        "Billing & Payments": "Finance & Billing Ops",
        "Refunds & Disputes": "Finance & Billing Ops",
        "Account & Authentication": "Identity & Security Engineering",
        "Technical & System Issues": "SRE & Core Infrastructure",
        "Technical & System Issue": "SRE & Core Infrastructure",
        "General Service Inquiry": "Tier-1 Customer Support",
        "General Product Inquiry": "Tier-1 Customer Support",
        # Legacy / Edge Taxonomy
        "Billing & Refunds": "Finance & Billing Ops",
        "SSO & Authentication": "Identity & Security Engineering",
        "API & Rate Limits": "Developer Relations & Platform Eng",
        "Cloud Outage & Downtime": "SRE & Core Infrastructure",
        "GDPR & Privacy": "Legal & Privacy Compliance",
        "General Inquiry": "Tier-1 Customer Support"
    }
    return mapping.get(intent, "Tier-1 Customer Support")

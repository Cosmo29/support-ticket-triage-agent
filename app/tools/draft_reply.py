"""
AI Draft Reply Generation Tool.
Drafts grounded, empathetic, and actionable support responses grounded in KB citations.
"""

import time
from typing import List, Tuple
from app.models.ticket import (
    CustomerContext,
    ClassificationResult,
    KBCitation,
    DraftReply,
    ToolCallRecord
)
from app.core.logging import logger

def draft_reply(
    ticket_id: str,
    subject: str,
    body: str,
    customer: CustomerContext,
    classification: ClassificationResult,
    citations: List[KBCitation]
) -> Tuple[DraftReply, ToolCallRecord]:
    """
    Generates a draft response grounded in verified Knowledge Base articles.
    """
    start_time = time.time()
    
    greeting = f"Hi {customer.name.split()[0] if customer.name != 'Guest User' else 'there'},"
    closing = "Best regards,\nCustomer Success & Support Engineering"
    
    # Adapt tone to customer sentiment and tier
    empathy = ""
    if classification.sentiment in ("FRUSTRATED", "ANGRY"):
        empathy = "Thank you for reaching out, and I apologize for the inconvenience this issue has caused you. We are treating this with high priority.\n\n"
    elif customer.vip_status:
        empathy = f"Thank you for contacting Enterprise Support. As an Enterprise client, your dedicated CSM ({customer.dedicated_csm or 'our team'}) has also been notified.\n\n"
        
    cited_titles = [c.article_title for c in citations]
    
    # Grounded body based on citations and intent
    if citations:
        primary_kb = citations[0]
        remediation_lines = []
        for line in primary_kb.snippet.splitlines()[:5]:
            if line.strip() and not line.startswith("#"):
                remediation_lines.append(f"• {line.strip()}")
                
        kb_text = "\n".join(remediation_lines) if remediation_lines else primary_kb.snippet
        
        reply_content = (
            f"{greeting}\n\n"
            f"{empathy}"
            f"Regarding your inquiry concerning **{subject}**:\n\n"
            f"Based on our technical documentation ({primary_kb.article_title}), here are the recommended steps:\n\n"
            f"{kb_text}\n\n"
            f"If you continue to encounter errors, please reply with any correlation IDs or error logs so we can investigate immediately.\n\n"
            f"{closing}"
        )
    else:
        # Fallback when no KB matches
        reply_content = (
            f"{greeting}\n\n"
            f"{empathy}"
            f"We have received your ticket regarding **{subject}** and our engineering team is actively investigating.\n\n"
            f"We will update you as soon as we have a diagnostic trace or resolution.\n\n"
            f"{closing}"
        )

    latency_ms = round((time.time() - start_time) * 1000, 2)
    draft = DraftReply(
        reply_text=reply_content,
        cited_articles=cited_titles,
        tone="Empathetic & Grounded" if classification.sentiment in ("FRUSTRATED", "ANGRY") else "Technical & Direct",
        status="PENDING_APPROVAL"
    )
    
    record = ToolCallRecord(
        tool_name="draft_reply",
        input_params={"ticket_id": ticket_id, "cited_count": len(citations)},
        output_summary=f"Drafted grounded response citing {len(citations)} KB sources",
        latency_ms=latency_ms
    )
    
    return draft, record

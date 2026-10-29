"""
Inbound Email Webhook Parser.
Transforms raw email HTTP post payloads (e.g., SendGrid Inbound Parse or Mailgun) into TicketCreate.
"""

import re
from typing import Dict, Any
from app.models.ticket import TicketCreate

def parse_inbound_email(form_data: Dict[str, Any]) -> TicketCreate:
    """
    Parses multipart email form data from SendGrid / Mailgun webhook.
    """
    # Extract sender
    from_field = form_data.get("from") or form_data.get("sender") or form_data.get("from_email", "unknown@sender.com")
    
    # Extract email address from string like "John Doe <john@company.com>"
    email_match = re.search(r"[\w\.-]+@[\w\.-]+", from_field)
    customer_email = email_match.group(0) if email_match else from_field.strip()

    subject = form_data.get("subject", "No Email Subject")
    body = form_data.get("text") or form_data.get("body_text") or form_data.get("body-plain") or ""

    # Strip quoted email reply chains (lines starting with '>')
    cleaned_lines = [line for line in body.splitlines() if not line.strip().startswith(">")]
    clean_body = "\n".join(cleaned_lines).strip()

    return TicketCreate(
        ticket_id=None,  # Will be generated
        source="email",
        customer_email=customer_email,
        subject=subject,
        body=clean_body or body,
        raw_metadata={
            "to": form_data.get("to"),
            "spam_score": form_data.get("spam_score"),
            "SPF": form_data.get("SPF")
        }
    )

"""
Customer Lookup Tool.
Queries CRM data to enrich ticket context with customer tier, ARR, SLA, and VIP status.
"""

import os
import json
import time
from typing import Optional, Tuple
from app.models.ticket import CustomerContext, ToolCallRecord
from app.core.logging import logger

CUSTOMERS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/seed_customers.json"))

def customer_lookup(email: str, customer_id: Optional[str] = None) -> Tuple[CustomerContext, ToolCallRecord]:
    """
    Enriches ticket with customer CRM profile based on email or customer_id.
    """
    start_time = time.time()
    context = CustomerContext(
        customer_id=customer_id or f"GUEST-{hash(email) % 10000:04d}",
        name="Guest User",
        email_domain=email.split("@")[-1] if "@" in email else "unknown",
        tier="Free",
        arr=0.0,
        sla_tier="P3_24h",
        dedicated_csm=None,
        open_tickets_count=0,
        account_status="Active",
        vip_status=False
    )
    
    if os.path.exists(CUSTOMERS_FILE):
        try:
            with open(CUSTOMERS_FILE, "r", encoding="utf-8") as f:
                customers = json.load(f)
                
            email_domain = email.split("@")[-1].lower() if "@" in email else ""
            clean_email = email.lower().strip()
            
            # Exact match by customer_id or contact_email
            for c in customers:
                if (customer_id and c.get("customer_id") == customer_id) or \
                   (c.get("contact_email", "").lower() == clean_email):
                    context = CustomerContext(**c)
                    break
            else:
                # Match by domain for enterprise/scale clients
                if email_domain and email_domain not in ("gmail.com", "yahoo.com", "outlook.com", "hotmail.com"):
                    for c in customers:
                        if c.get("email_domain", "").lower() == email_domain:
                            context = CustomerContext(**c)
                            break
        except Exception as e:
            logger.error(f"Error reading customer database: {e}")

    latency_ms = round((time.time() - start_time) * 1000, 2)
    record = ToolCallRecord(
        tool_name="customer_lookup",
        input_params={"email": email, "customer_id": customer_id},
        output_summary=f"Found: {context.name} (Tier: {context.tier}, ARR: ${context.arr:,.0f}, VIP: {context.vip_status})",
        latency_ms=latency_ms
    )
    
    return context, record

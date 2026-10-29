"""
Zendesk Connector: Webhook Intake & Outbound API Integration.
Supports both live Zendesk Sandbox API communication and local resilient simulation.
"""

from typing import Dict, Any, Optional
import httpx
from app.models.ticket import TicketCreate
from app.config import settings
from app.core.logging import logger

class ZendeskConnector:
    def __init__(self):
        self.subdomain = settings.ZENDESK_SUBDOMAIN
        self.email = settings.ZENDESK_EMAIL
        self.token = settings.ZENDESK_API_TOKEN
        self.base_url = f"https://{self.subdomain}.zendesk.com/api/v2"

    def parse_webhook_payload(self, payload: Dict[str, Any]) -> TicketCreate:
        """
        Parses standard Zendesk webhook JSON payload into normalized TicketCreate format.
        """
        ticket_data = payload.get("ticket", payload)
        
        ticket_id = f"ZD-{ticket_data.get('id', 'NEW')}"
        requester = ticket_data.get("requester", {})
        if isinstance(requester, dict):
            customer_email = requester.get("email", "unknown@zendesk-user.com")
        else:
            customer_email = ticket_data.get("requester_email", "unknown@zendesk-user.com")

        subject = ticket_data.get("subject", "No Subject Provided")
        description = ticket_data.get("description") or ticket_data.get("body") or ticket_data.get("comment", {}).get("body", "")

        return TicketCreate(
            ticket_id=ticket_id,
            source="zendesk",
            customer_email=customer_email,
            subject=subject,
            body=description,
            raw_metadata={
                "zendesk_id": ticket_data.get("id"),
                "status": ticket_data.get("status"),
                "priority": ticket_data.get("priority"),
                "tags": ticket_data.get("tags", [])
            }
        )

    async def update_zendesk_ticket(
        self,
        zendesk_id: str,
        priority: str,
        tags: list[str],
        internal_note: str,
        public_reply: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Pushes triage results back to Zendesk sandbox ticket via REST API.
        """
        # If running in sandbox/mock mode without live credentials:
        if "sandbox" in self.subdomain or not self.token or self.token == "sandbox_token_abc123":
            logger.info(
                f"[SANDBOX MOCK] Pushed update to Zendesk Ticket #{zendesk_id}: Priority={priority}, Tags={tags}",
                extra={"zendesk_id": zendesk_id, "action": "ZENDESK_DISPATCH_MOCK"}
            )
            return {"status": "success", "mock": True, "updated_id": zendesk_id}

        url = f"{self.base_url}/tickets/{zendesk_id}.json"
        auth = (f"{self.email}/token", self.token)
        
        ticket_update: Dict[str, Any] = {
            "priority": priority.lower(),
            "tags": tags,
            "comment": {
                "body": internal_note,
                "public": False
            }
        }
        
        if public_reply:
            ticket_update["comment"] = {
                "body": public_reply,
                "public": True
            }

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.put(url, json={"ticket": ticket_update}, auth=auth)
                if res.status_code in (200, 201):
                    logger.info(f"Successfully synced ticket {zendesk_id} to Zendesk.")
                    return res.json()
                else:
                    logger.warning(f"Zendesk API returned status {res.status_code}: {res.text}")
                    return {"status": "error", "code": res.status_code, "text": res.text}
        except Exception as e:
            logger.error(f"Zendesk API network exception: {e}")
            return {"status": "error", "message": str(e)}

zendesk_connector = ZendeskConnector()

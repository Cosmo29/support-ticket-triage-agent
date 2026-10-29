"""
Slack Connector: Inbound Webhook Parser & Block Kit Escalation Dispatcher.
"""

from typing import Dict, Any, Optional
import httpx
from app.models.ticket import TicketCreate, TicketResponse
from app.config import settings
from app.core.logging import logger

class SlackConnector:
    def __init__(self):
        self.webhook_url = settings.SLACK_WEBHOOK_URL
        self.escalation_channel = settings.SLACK_ESCALATION_CHANNEL

    def parse_slack_payload(self, form_data: Dict[str, Any]) -> TicketCreate:
        """
        Parses Slack slash command (e.g., /triage) or interactive payload into TicketCreate.
        """
        user_name = form_data.get("user_name", "slack_user")
        user_email = form_data.get("user_email") or f"{user_name}@slack-team.internal"
        text = form_data.get("text", "")
        
        # Split text into subject and body if separator exists
        if " - " in text:
            parts = text.split(" - ", 1)
            subject, body = parts[0].strip(), parts[1].strip()
        else:
            subject = text[:60] if text else "Slack Inbound Support Request"
            body = text or "No details provided via Slack"

        return TicketCreate(
            ticket_id=f"SLACK-{form_data.get('trigger_id', 'REQ')[:6].upper()}",
            source="slack",
            customer_email=user_email,
            subject=subject,
            body=body,
            raw_metadata=form_data
        )

    async def post_escalation_alert(self, ticket: TicketResponse) -> bool:
        """
        Posts a formatted Slack Block Kit alert to the support escalation channel.
        """
        # Block Kit payload
        priority_emoji = "🚨" if ticket.classification and ticket.classification.priority == "P0" else "⚠️"
        threat_notice = f"🔒 Threat: *{ticket.guardrail.threat_level}*" if ticket.guardrail and not ticket.guardrail.is_safe else ""

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{priority_emoji} Escalation Alert: {ticket.ticket_id}",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Customer:* {ticket.customer.name if ticket.customer else 'Unknown'}"},
                    {"type": "mrkdwn", "text": f"*Tier:* {ticket.customer.tier if ticket.customer else 'Free'} (ARR: ${ticket.customer.arr:,.0f})" if ticket.customer else "*Tier:* Free"},
                    {"type": "mrkdwn", "text": f"*Assigned Team:* {ticket.assigned_team}"},
                    {"type": "mrkdwn", "text": f"*SLA Deadline:* {ticket.sla_target_minutes} min ({ticket.classification.priority if ticket.classification else 'P3'})"}
                ]
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Reason:* {ticket.escalation_reason or 'Manual review required'}\n*Subject:* {ticket.subject}\n{threat_notice}"
                }
            }
        ]

        payload = {
            "channel": self.escalation_channel,
            "text": f"Escalation Alert: {ticket.ticket_id} - {ticket.subject}",
            "blocks": blocks
        }

        # Sandbox / Mock fallback if live webhook url is dummy
        if "MOCK" in self.webhook_url or not self.webhook_url.startswith("https://hooks.slack.com"):
            logger.info(
                f"[SLACK MOCK NOTIFICATION] Dispatched alert for {ticket.ticket_id} to {self.escalation_channel}",
                extra={"ticket_id": ticket.ticket_id, "action": "SLACK_DISPATCH_MOCK"}
            )
            return True

        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                res = await client.post(self.webhook_url, json=payload)
                return res.status_code == 200
        except Exception as e:
            logger.error(f"Error posting Slack escalation: {e}")
            return False

slack_connector = SlackConnector()

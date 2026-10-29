"""
Unit Tests for Zendesk, Slack, and Email Webhook Connectors.
"""

from app.connectors.zendesk import zendesk_connector
from app.connectors.slack import slack_connector
from app.connectors.email_parser import parse_inbound_email

def test_zendesk_webhook_parser():
    payload = {
        "ticket": {
            "id": 14920,
            "subject": "SSO Certificate expired on Okta",
            "description": "Engineers cannot log in through SAML.",
            "requester": {"email": "devs@hyperscale.io"},
            "priority": "urgent"
        }
    }
    ticket = zendesk_connector.parse_webhook_payload(payload)
    assert ticket.ticket_id == "ZD-14920"
    assert ticket.source == "zendesk"
    assert ticket.customer_email == "devs@hyperscale.io"
    assert "SSO Certificate" in ticket.subject

def test_slack_payload_parser():
    form_data = {
        "user_name": "jane.doe",
        "user_email": "jane@acmefinancial.com",
        "text": "Rate Limit 429 - Bulk sync pipeline blocked by API throttling",
        "trigger_id": "992144"
    }
    ticket = slack_connector.parse_slack_payload(form_data)
    assert ticket.source == "slack"
    assert ticket.customer_email == "jane@acmefinancial.com"
    assert "Rate Limit 429" in ticket.subject

def test_email_webhook_parser():
    form_data = {
        "from": "John Client <founder@indiedev.co>",
        "subject": "Question about refund policy",
        "text": "Can we get a refund within 14 days?\n> On Mon, previous reply..."
    }
    ticket = parse_inbound_email(form_data)
    assert ticket.source == "email"
    assert ticket.customer_email == "founder@indiedev.co"
    assert "Question about refund" in ticket.subject
    assert "> On Mon" not in ticket.body

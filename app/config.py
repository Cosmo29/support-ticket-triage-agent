"""
Application Configuration and Environment Settings.
Uses standard Pydantic BaseModel and os.environ with fallback defaults.
"""

import os
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv

# Pre-load local .env file if it exists
load_dotenv()

class Settings(BaseModel):
    # Server & Runtime
    PORT: int = int(os.getenv("PORT", "8000"))
    HOST: str = os.getenv("HOST", "0.0.0.0")
    STREAMLIT_PORT: int = int(os.getenv("STREAMLIT_PORT", "8501"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    
    # LLM & Classifier Settings
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-1.5-flash")
    
    # Triage Agent Thresholds
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))
    ENABLE_PROMPT_INJECTION_SHIELD: bool = os.getenv("ENABLE_PROMPT_INJECTION_SHIELD", "true").lower() in ("true", "1", "yes")
    AUTO_ESCALATE_P0: bool = os.getenv("AUTO_ESCALATE_P0", "true").lower() in ("true", "1", "yes")
    
    # Connectors
    ZENDESK_SUBDOMAIN: str = os.getenv("ZENDESK_SUBDOMAIN", "sandbox-company")
    ZENDESK_EMAIL: str = os.getenv("ZENDESK_EMAIL", "agent@example.com")
    ZENDESK_API_TOKEN: str = os.getenv("ZENDESK_API_TOKEN", "sandbox_token_abc123")
    SLACK_WEBHOOK_URL: str = os.getenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/services/MOCK/TOKEN/WEBHOOK")
    SLACK_ESCALATION_CHANNEL: str = os.getenv("SLACK_ESCALATION_CHANNEL", "#support-escalations")
    
    # Database & Storage
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./data/tickets.db")
    KB_DIRECTORY: str = os.getenv("KB_DIRECTORY", "./data/knowledge_base")
    CUSTOMERS_FILE: str = os.getenv("CUSTOMERS_FILE", "./data/seed_customers.json")

settings = Settings()

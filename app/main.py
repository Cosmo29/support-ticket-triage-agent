"""
FastAPI Application Entrypoint for AI Support Ticket Triage Agent.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.core.database import init_db
from app.core.logging import logger
from app.tools.kb_search import kb_engine
from app.pipeline.triage_agent import triage_agent
from app.api.routes_tickets import router as tickets_router
from app.api.routes_webhooks import router as webhooks_router
from app.api.routes_metrics import router as metrics_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle startup and shutdown management."""
    logger.info("Initializing Support Ticket Triage Agent backend...")
    init_db()
    kb_engine._index_articles()
    triage_agent.fine_tuned_classifier._train()
    logger.info("Startup complete. All agent pipelines, guardrails, and tools ready.")
    yield
    logger.info("Shutting down Triage Agent service.")

app = FastAPI(
    title="AI Support Ticket Triage Agent API",
    description="Enterprise-grade autonomous support ticket triage agent with multi-dimensional classification, prompt injection protection, tool calling, and Zendesk/Slack webhooks.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for dashboard and local tools
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(tickets_router)
app.include_router(webhooks_router)
app.include_router(metrics_router)

@app.get("/health", tags=["Health"])
def health_check():
    """Service health and component status check."""
    return {
        "status": "healthy",
        "service": "Support Ticket Triage Agent",
        "environment": settings.ENVIRONMENT,
        "kb_articles_indexed": len(kb_engine.documents),
        "fine_tuned_model_ready": triage_agent.fine_tuned_classifier.is_trained,
        "guardrail_shield_active": settings.ENABLE_PROMPT_INJECTION_SHIELD
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)

"""
Observability & Triage Metrics API Routes.
"""

from typing import Dict, Any, List
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.core.database import get_db
from app.models.entities import TicketEntity, AuditLogEntity

router = APIRouter(prefix="/api/metrics", tags=["Metrics"])

@router.get("/summary")
def get_metrics_summary(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Returns aggregated KPIs, SLA status, confidence, and distribution metrics.
    """
    total = db.query(TicketEntity).count()
    if total == 0:
        return {
            "total_tickets": 0,
            "escalated_count": 0,
            "escalation_rate": 0.0,
            "quarantined_count": 0,
            "resolved_count": 0,
            "avg_latency_ms": 0.0,
            "avg_confidence": 0.0,
            "intents_distribution": {},
            "priorities_distribution": {},
            "teams_distribution": {}
        }

    escalated = db.query(TicketEntity).filter(TicketEntity.status == "ESCALATED").count()
    quarantined = db.query(TicketEntity).filter(TicketEntity.status == "QUARANTINED").count()
    resolved = db.query(TicketEntity).filter(TicketEntity.status == "RESOLVED").count()
    
    avg_latency = db.query(func.avg(TicketEntity.latency_ms)).scalar() or 0.0
    avg_confidence = db.query(func.avg(TicketEntity.confidence)).scalar() or 0.0

    # Intent distribution
    intents = db.query(TicketEntity.intent, func.count(TicketEntity.id)).group_by(TicketEntity.intent).all()
    intent_dist = {i[0] or "Unknown": i[1] for i in intents}

    # Priority distribution
    priorities = db.query(TicketEntity.priority, func.count(TicketEntity.id)).group_by(TicketEntity.priority).all()
    prio_dist = {p[0] or "Unknown": p[1] for p in priorities}

    # Team assignment distribution
    teams = db.query(TicketEntity.assigned_team, func.count(TicketEntity.id)).group_by(TicketEntity.assigned_team).all()
    team_dist = {t[0] or "Unassigned": t[1] for t in teams}

    return {
        "total_tickets": total,
        "escalated_count": escalated,
        "escalation_rate": round((escalated / total) * 100, 1),
        "quarantined_count": quarantined,
        "resolved_count": resolved,
        "avg_latency_ms": round(float(avg_latency), 2),
        "avg_confidence": round(float(avg_confidence), 3),
        "intents_distribution": intent_dist,
        "priorities_distribution": prio_dist,
        "teams_distribution": team_dist
    }

@router.get("/audit-logs")
def get_audit_logs(limit: int = 50, db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    """
    Returns recent audit trace records for agent operations.
    """
    logs = db.query(AuditLogEntity).order_by(AuditLogEntity.timestamp.desc()).limit(limit).all()
    return [
        {
            "id": l.id,
            "ticket_id": l.ticket_id,
            "event_type": l.event_type,
            "details": l.details_json,
            "timestamp": l.timestamp.isoformat() if l.timestamp else None
        }
        for l in logs
    ]

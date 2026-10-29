"""
Structured JSON Logger & Observability Engine.
Produces structured log lines with trace context, ticket metadata, latencies, and security flags.
"""

import sys
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects for log aggregators."""
    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Inject standard contextual attributes if passed via extra
        for attr in ("trace_id", "ticket_id", "action", "intent", "priority", 
                     "confidence", "latency_ms", "threat_level", "model", "tokens"):
            if hasattr(record, attr):
                log_data[attr] = getattr(record, attr)
                
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_data)

def setup_logger(name: str = "triage_agent", level: str = "INFO", json_output: bool = True) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        if json_output:
            handler.setFormatter(JSONFormatter())
        else:
            handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
        
    return logger

# Global default logger instance
logger = setup_logger()

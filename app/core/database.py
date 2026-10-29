"""
Database Session Management and Schema Initialization.
Uses SQLite for zero-configuration, self-contained embedded persistence.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from app.models.entities import Base
from app.core.logging import logger

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/tickets.db"))
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

SYNC_DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    SYNC_DATABASE_URL,
    connect_args={"check_same_thread": False},
    pool_pre_ping=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initializes database tables if they do not exist."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info(f"Database initialized successfully at {DB_PATH}")
    except Exception as e:
        logger.error(f"Error initializing database: {e}", exc_info=True)

def get_db():
    """FastAPI Dependency for database session."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()

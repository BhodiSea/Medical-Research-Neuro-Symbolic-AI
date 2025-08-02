"""
Database connection management for Medical Research AI API
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator
import logging

from ..core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Create database engine
engine = create_engine(
    settings.DATABASE_URL,
    pool_size=settings.DATABASE_POOL_SIZE,
    max_overflow=settings.DATABASE_MAX_OVERFLOW,
    pool_timeout=settings.DATABASE_POOL_TIMEOUT,
    echo=settings.DEBUG  # Log SQL queries in debug mode
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base model class
Base = declarative_base()


class Database:
    """Database connection manager"""
    
    def __init__(self):
        self.engine = engine
        self.SessionLocal = SessionLocal
    
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session"""
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()
    
    def close(self):
        """Close database connections"""
        self.engine.dispose()


# Global database instance
_database = None


def get_database() -> Database:
    """Get database instance"""
    global _database
    if _database is None:
        _database = Database()
    return _database


def get_db_session() -> Generator[Session, None, None]:
    """FastAPI dependency for database sessions"""
    db = get_database()
    yield from db.get_session()


def create_tables():
    """Create all database tables"""
    from .models import Base
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created") 
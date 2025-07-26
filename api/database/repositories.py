"""
Repository pattern for database operations
"""

from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from fastapi import Depends
import logging

from .models import User, MedicalQuery, ApplicationReview, StudyPlan, SchoolMatch, AuditLog
from .connection import get_db_session

logger = logging.getLogger(__name__)


class BaseRepository:
    """Base repository with common operations"""
    
    def __init__(self, model, db_session: Session):
        self.model = model
        self.db = db_session
    
    def get_by_id(self, id: int):
        """Get record by ID"""
        return self.db.query(self.model).filter(self.model.id == id).first()
    
    def get_all(self, limit: int = 100, offset: int = 0):
        """Get all records with pagination"""
        return self.db.query(self.model).offset(offset).limit(limit).all()
    
    def create(self, obj):
        """Create new record"""
        self.db.add(obj)
        self.db.commit()
        self.db.refresh(obj)
        return obj
    
    def update(self, obj):
        """Update existing record"""
        self.db.commit()
        self.db.refresh(obj)
        return obj
    
    def delete(self, obj):
        """Delete record"""
        self.db.delete(obj)
        self.db.commit()


class UserRepository(BaseRepository):
    """User repository"""
    
    def __init__(self, db_session: Session):
        super().__init__(User, db_session)
    
    def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        return self.db.query(User).filter(User.email == email).first()
    
    def get_active_users(self, limit: int = 100, offset: int = 0) -> List[User]:
        """Get active users"""
        return self.db.query(User).filter(User.is_active == True).offset(offset).limit(limit).all()


class MedicalQueryRepository(BaseRepository):
    """Medical query repository"""
    
    def __init__(self, db_session: Session):
        super().__init__(MedicalQuery, db_session)
    
    def get_user_queries(self, user_id: int, limit: int = 20, offset: int = 0) -> List[MedicalQuery]:
        """Get queries for a specific user"""
        return (
            self.db.query(MedicalQuery)
            .filter(MedicalQuery.user_id == user_id)
            .order_by(MedicalQuery.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )
    
    def get_by_query_type(self, query_type: str, limit: int = 100, offset: int = 0) -> List[MedicalQuery]:
        """Get queries by type"""
        return (
            self.db.query(MedicalQuery)
            .filter(MedicalQuery.query_type == query_type)
            .order_by(MedicalQuery.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )


class ApplicationReviewRepository(BaseRepository):
    """Application review repository"""
    
    def __init__(self, db_session: Session):
        super().__init__(ApplicationReview, db_session)
    
    def get_user_reviews(self, user_id: int, limit: int = 20, offset: int = 0) -> List[ApplicationReview]:
        """Get reviews for a specific user"""
        return (
            self.db.query(ApplicationReview)
            .filter(ApplicationReview.user_id == user_id)
            .order_by(ApplicationReview.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )


# Repository dependency functions for FastAPI
def get_user_repository(db: Session = Depends(get_db_session)) -> UserRepository:
    """Get user repository"""
    return UserRepository(db)


def get_medical_query_repository(db: Session = Depends(get_db_session)) -> MedicalQueryRepository:
    """Get medical query repository"""
    return MedicalQueryRepository(db)


def get_application_review_repository(db: Session = Depends(get_db_session)) -> ApplicationReviewRepository:
    """Get application review repository"""
    return ApplicationReviewRepository(db) 
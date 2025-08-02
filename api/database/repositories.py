"""
Repository pattern for database operations
"""

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from typing import List, Optional, Dict, Any
from fastapi import Depends
import logging
from passlib.context import CryptContext
from datetime import datetime

from .models import User, MedicalQuery, ApplicationReview, StudyPlan, SchoolMatch, AuditLog
from .connection import get_db_session

logger = logging.getLogger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


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
    """User repository with authentication capabilities"""
    
    def __init__(self, db_session: Session):
        super().__init__(User, db_session)
    
    def create_user(self, user_data: Dict[str, Any]) -> User:
        """Create new user with password hashing"""
        try:
            # Hash the password
            hashed_password = pwd_context.hash(user_data["password"])
            
            # Create user object
            user = User(
                email=user_data["email"],
                hashed_password=hashed_password,
                first_name=user_data["first_name"],
                last_name=user_data["last_name"],
                role=user_data.get("role", "premed"),
                is_active=user_data.get("is_active", True),
                is_verified=user_data.get("is_verified", False),
                academic_background=user_data.get("academic_background"),
                career_goals=user_data.get("career_goals")
            )
            
            # Save to database
            self.db.add(user)
            self.db.commit()
            self.db.refresh(user)
            
            logger.info(f"Created new user: {user.email}")
            return user
            
        except IntegrityError as e:
            self.db.rollback()
            logger.error(f"Failed to create user {user_data.get('email')}: {str(e)}")
            raise ValueError("User with this email already exists")
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating user: {str(e)}")
            raise
    
    def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        return self.db.query(User).filter(User.email == email).first()
    
    def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """Authenticate user with email and password"""
        user = self.get_by_email(email)
        if not user:
            return None
        
        if not pwd_context.verify(password, user.hashed_password):
            return None
        
        # Update last login timestamp
        user.last_login = datetime.utcnow()
        self.db.commit()
        
        logger.info(f"User authenticated: {email}")
        return user
    
    def update_password(self, user_id: int, new_password: str) -> bool:
        """Update user password"""
        try:
            user = self.get_by_id(user_id)
            if not user:
                return False
            
            user.hashed_password = pwd_context.hash(new_password)
            self.db.commit()
            
            logger.info(f"Password updated for user: {user.email}")
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error updating password for user {user_id}: {str(e)}")
            return False
    
    def deactivate_user(self, user_id: int) -> bool:
        """Deactivate user account"""
        try:
            user = self.get_by_id(user_id)
            if not user:
                return False
            
            user.is_active = False
            self.db.commit()
            
            logger.info(f"User deactivated: {user.email}")
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error deactivating user {user_id}: {str(e)}")
            return False
    
    def verify_user(self, user_id: int) -> bool:
        """Mark user as verified"""
        try:
            user = self.get_by_id(user_id)
            if not user:
                return False
            
            user.is_verified = True
            self.db.commit()
            
            logger.info(f"User verified: {user.email}")
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error verifying user {user_id}: {str(e)}")
            return False
    
    def get_active_users(self, limit: int = 100, offset: int = 0) -> List[User]:
        """Get active users"""
        return self.db.query(User).filter(User.is_active == True).offset(offset).limit(limit).all()
    
    def get_users_by_role(self, role: str, limit: int = 100, offset: int = 0) -> List[User]:
        """Get users by role"""
        return (
            self.db.query(User)
            .filter(User.role == role, User.is_active == True)
            .offset(offset)
            .limit(limit)
            .all()
        )


class MedicalQueryRepository(BaseRepository):
    """Medical query repository with enhanced functionality"""
    
    def __init__(self, db_session: Session):
        super().__init__(MedicalQuery, db_session)
    
    def save_query(self, query_data: Dict[str, Any]) -> MedicalQuery:
        """Save medical query with metadata"""
        try:
            query = MedicalQuery(
                user_id=query_data["user_id"],
                query_text=query_data["query_text"],
                query_type=query_data.get("query_type", "general"),
                urgency=query_data.get("urgency", "normal"),
                response_data=query_data.get("response_data"),
                confidence_score=query_data.get("confidence_score"),
                ethical_compliance=query_data.get("ethical_compliance"),
                processing_time_ms=query_data.get("processing_time_ms"),
                contains_personal_data=query_data.get("contains_personal_data", False),
                follow_up_to=query_data.get("follow_up_to")
            )
            
            self.db.add(query)
            self.db.commit()
            self.db.refresh(query)
            
            logger.info(f"Saved medical query {query.id} for user {query.user_id}")
            return query
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error saving medical query: {str(e)}")
            raise
    
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
    
    def get_queries_by_urgency(self, urgency: str, limit: int = 100, offset: int = 0) -> List[MedicalQuery]:
        """Get queries by urgency level"""
        return (
            self.db.query(MedicalQuery)
            .filter(MedicalQuery.urgency == urgency)
            .order_by(MedicalQuery.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )
    
    def get_follow_up_queries(self, original_query_id: int) -> List[MedicalQuery]:
        """Get follow-up queries for an original query"""
        return (
            self.db.query(MedicalQuery)
            .filter(MedicalQuery.follow_up_to == original_query_id)
            .order_by(MedicalQuery.created_at.asc())
            .all()
        )
    
    def get_query_analytics(self) -> Dict[str, Any]:
        """Get query analytics and statistics"""
        try:
            from sqlalchemy import func
            
            # Total queries
            total_queries = self.db.query(MedicalQuery).count()
            
            # Queries by type
            queries_by_type = (
                self.db.query(
                    MedicalQuery.query_type,
                    func.count(MedicalQuery.id).label('count')
                )
                .group_by(MedicalQuery.query_type)
                .all()
            )
            
            # Queries by urgency
            queries_by_urgency = (
                self.db.query(
                    MedicalQuery.urgency,
                    func.count(MedicalQuery.id).label('count')
                )
                .group_by(MedicalQuery.urgency)
                .all()
            )
            
            # Average confidence score
            avg_confidence = (
                self.db.query(func.avg(MedicalQuery.confidence_score))
                .filter(MedicalQuery.confidence_score.isnot(None))
                .scalar() or 0.0
            )
            
            # Average processing time
            avg_processing_time = (
                self.db.query(func.avg(MedicalQuery.processing_time_ms))
                .filter(MedicalQuery.processing_time_ms.isnot(None))
                .scalar() or 0.0
            )
            
            # Ethical compliance rate
            ethical_compliant = (
                self.db.query(MedicalQuery)
                .filter(MedicalQuery.ethical_compliance == True)
                .count()
            )
            ethical_compliance_rate = (ethical_compliant / total_queries * 100) if total_queries > 0 else 0.0
            
            return {
                "total_queries": total_queries,
                "queries_by_type": {row.query_type: row.count for row in queries_by_type},
                "queries_by_urgency": {row.urgency: row.count for row in queries_by_urgency},
                "average_confidence_score": round(avg_confidence, 2),
                "average_processing_time_ms": round(avg_processing_time, 2),
                "ethical_compliance_rate": round(ethical_compliance_rate, 2)
            }
            
        except Exception as e:
            logger.error(f"Error generating query analytics: {str(e)}")
            return {}


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


class AuditLogRepository(BaseRepository):
    """Audit log repository for tracking system activities"""
    
    def __init__(self, db_session: Session):
        super().__init__(AuditLog, db_session)
    
    def log_activity(self, activity_data: Dict[str, Any]) -> AuditLog:
        """Log system activity for audit trail"""
        try:
            audit_log = AuditLog(
                user_id=activity_data.get("user_id"),
                action=activity_data["action"],
                resource_type=activity_data.get("resource_type"),
                resource_id=activity_data.get("resource_id"),
                details=activity_data.get("details"),
                ip_address=activity_data.get("ip_address"),
                user_agent=activity_data.get("user_agent"),
                success=activity_data.get("success", True)
            )
            
            self.db.add(audit_log)
            self.db.commit()
            self.db.refresh(audit_log)
            
            return audit_log
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error logging audit activity: {str(e)}")
            raise
    
    def get_user_activities(self, user_id: int, limit: int = 50, offset: int = 0) -> List[AuditLog]:
        """Get audit log entries for a specific user"""
        return (
            self.db.query(AuditLog)
            .filter(AuditLog.user_id == user_id)
            .order_by(AuditLog.timestamp.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )
    
    def get_activities_by_action(self, action: str, limit: int = 100, offset: int = 0) -> List[AuditLog]:
        """Get audit log entries by action type"""
        return (
            self.db.query(AuditLog)
            .filter(AuditLog.action == action)
            .order_by(AuditLog.timestamp.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )
    
    def get_failed_activities(self, limit: int = 100, offset: int = 0) -> List[AuditLog]:
        """Get failed activities for security monitoring"""
        return (
            self.db.query(AuditLog)
            .filter(AuditLog.success == False)
            .order_by(AuditLog.timestamp.desc())
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


def get_audit_log_repository(db: Session = Depends(get_db_session)) -> AuditLogRepository:
    """Get audit log repository"""
    return AuditLogRepository(db) 
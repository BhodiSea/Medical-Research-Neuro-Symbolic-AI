"""
Database models for Medical Research AI API
"""

from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, Float, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional, Dict, Any

Base = declarative_base()


class User(Base):
    """User model"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    role = Column(String(50), nullable=False, default="premed")
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    last_login = Column(DateTime(timezone=True), nullable=True)
    
    # Profile information
    academic_background = Column(JSON, nullable=True)
    career_goals = Column(Text, nullable=True)
    
    # Relationships
    medical_queries = relationship("MedicalQuery", back_populates="user")
    application_reviews = relationship("ApplicationReview", back_populates="user")
    
    def __repr__(self):
        return f"<User(email='{self.email}', role='{self.role}')>"


class MedicalQuery(Base):
    """Medical query model"""
    __tablename__ = "medical_queries"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    # Query data
    query_text = Column(Text, nullable=False)
    query_type = Column(String(50), nullable=False, default="general")
    urgency = Column(String(20), nullable=False, default="normal")
    
    # Response data
    response_data = Column(JSON, nullable=True)
    confidence_score = Column(Float, nullable=True)
    ethical_compliance = Column(Boolean, nullable=True)
    processing_time_ms = Column(Integer, nullable=True)
    
    # Metadata
    contains_personal_data = Column(Boolean, default=False, nullable=False)
    follow_up_to = Column(Integer, ForeignKey("medical_queries.id"), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="medical_queries")
    follow_ups = relationship("MedicalQuery", remote_side=[id])
    
    def __repr__(self):
        return f"<MedicalQuery(id={self.id}, user_id={self.user_id}, type='{self.query_type}')>"


class ApplicationReview(Base):
    """Application review model"""
    __tablename__ = "application_reviews"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    # Review data
    component = Column(String(100), nullable=False)
    content = Column(Text, nullable=False)
    target_schools = Column(JSON, nullable=True)
    
    # Results
    overall_score = Column(Float, nullable=True)
    strengths = Column(JSON, nullable=True)
    weaknesses = Column(JSON, nullable=True)
    improvement_suggestions = Column(JSON, nullable=True)
    detailed_feedback = Column(JSON, nullable=True)
    competitiveness_assessment = Column(JSON, nullable=True)
    
    # Metadata
    confidence_score = Column(Float, nullable=True)
    estimated_review_time = Column(Integer, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="application_reviews")
    
    def __repr__(self):
        return f"<ApplicationReview(id={self.id}, user_id={self.user_id}, component='{self.component}')>"


class StudyPlan(Base):
    """Study plan model"""
    __tablename__ = "study_plans"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    # Plan data
    target_exam = Column(String(100), nullable=False)
    target_date = Column(DateTime, nullable=False)
    current_level = Column(String(100), nullable=False)
    available_hours_per_week = Column(Integer, nullable=False)
    
    # Plan details
    total_study_hours = Column(Integer, nullable=True)
    weekly_schedule = Column(JSON, nullable=True)
    milestones = Column(JSON, nullable=True)
    resource_recommendations = Column(JSON, nullable=True)
    
    # Metadata
    success_probability = Column(Float, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User")
    
    def __repr__(self):
        return f"<StudyPlan(id={self.id}, user_id={self.user_id}, exam='{self.target_exam}')>"


class SchoolMatch(Base):
    """School matching result model"""
    __tablename__ = "school_matches"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    # Input data
    academic_stats = Column(JSON, nullable=False)
    preferences = Column(JSON, nullable=True)
    
    # Results
    safety_schools = Column(JSON, nullable=True)
    target_schools = Column(JSON, nullable=True)
    reach_schools = Column(JSON, nullable=True)
    overall_strategy = Column(JSON, nullable=True)
    
    # Metadata
    success_predictions = Column(JSON, nullable=True)
    estimated_costs = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User")
    
    def __repr__(self):
        return f"<SchoolMatch(id={self.id}, user_id={self.user_id})>"


class AuditLog(Base):
    """Audit log model for tracking important actions"""
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    
    # Action data
    action = Column(String(100), nullable=False, index=True)
    resource_type = Column(String(100), nullable=True)
    resource_id = Column(String(100), nullable=True)
    
    # Details
    details = Column(JSON, nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    
    # Result
    success = Column(Boolean, nullable=False)
    error_message = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User")
    
    def __repr__(self):
        return f"<AuditLog(id={self.id}, action='{self.action}', success={self.success})>" 
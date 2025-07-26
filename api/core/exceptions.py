"""
Custom exceptions for PremedPro AI API
"""

from fastapi import HTTPException, status
from typing import Dict, Any, Optional


class PremedProException(HTTPException):
    """Base exception for PremedPro AI"""
    
    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: str,
        headers: Optional[Dict[str, Any]] = None
    ):
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.error_code = error_code


class AuthenticationException(PremedProException):
    """Authentication related exceptions"""
    
    def __init__(self, detail: str = "Authentication failed"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            error_code="AUTHENTICATION_FAILED",
            headers={"WWW-Authenticate": "Bearer"}
        )


class AuthorizationException(PremedProException):
    """Authorization related exceptions"""
    
    def __init__(self, detail: str = "Insufficient permissions"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
            error_code="AUTHORIZATION_FAILED"
        )


class ValidationException(PremedProException):
    """Data validation exceptions"""
    
    def __init__(self, detail: str = "Validation failed"):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail,
            error_code="VALIDATION_ERROR"
        )


class MedicalAIException(PremedProException):
    """Medical AI service exceptions"""
    
    def __init__(self, detail: str = "Medical AI service error"):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
            error_code="MEDICAL_AI_ERROR"
        )


class DatabaseException(PremedProException):
    """Database related exceptions"""
    
    def __init__(self, detail: str = "Database error"):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
            error_code="DATABASE_ERROR"
        )


class RateLimitException(PremedProException):
    """Rate limiting exceptions"""
    
    def __init__(self, detail: str = "Rate limit exceeded"):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail,
            error_code="RATE_LIMIT_EXCEEDED",
            headers={"Retry-After": "60"}
        )


class FileUploadException(PremedProException):
    """File upload related exceptions"""
    
    def __init__(self, detail: str = "File upload error"):
        super().__init__(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=detail,
            error_code="FILE_UPLOAD_ERROR"
        )


class EthicsViolationException(PremedProException):
    """Ethics compliance violations"""
    
    def __init__(self, detail: str = "Ethics policy violation"):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
            error_code="ETHICS_VIOLATION"
        ) 
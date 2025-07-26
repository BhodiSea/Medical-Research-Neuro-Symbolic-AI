"""
Authentication and authorization for PremedPro AI API
"""

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, List
import jwt
import logging
from datetime import datetime, timedelta

from .config import get_settings
from .exceptions import AuthenticationException, AuthorizationException

logger = logging.getLogger(__name__)
security = HTTPBearer()
settings = get_settings()


class User:
    """User model placeholder"""
    def __init__(self, id: int, email: str, role: str, is_active: bool = True):
        self.id = id
        self.email = email
        self.role = role
        self.is_active = is_active


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """Get current authenticated user"""
    
    # This is a placeholder implementation
    # In production, this would validate JWT tokens and query the database
    
    try:
        # Decode JWT token
        token = credentials.credentials
        
        # For now, create a mock user
        # In production, decode the JWT and get user from database
        mock_user = User(
            id=1,
            email="test@example.com",
            role="premed",
            is_active=True
        )
        
        return mock_user
        
    except Exception as e:
        logger.error(f"Authentication failed: {str(e)}")
        raise AuthenticationException("Invalid authentication credentials")


async def get_optional_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[User]:
    """Get current user if authenticated, otherwise return None"""
    
    if not credentials:
        return None
    
    try:
        return await get_current_user(credentials)
    except Exception:
        return None


def require_permissions(required_permissions: List[str]):
    """Decorator to require specific permissions"""
    
    def permission_checker(current_user: User = Depends(get_current_user)):
        # This is a placeholder - implement actual permission checking
        if not current_user.is_active:
            raise AuthorizationException("User account is disabled")
        
        # For now, all active users have all permissions
        # In production, check user permissions against required_permissions
        
        return current_user
    
    return permission_checker


def require_role(required_role: str):
    """Decorator to require specific role"""
    
    def role_checker(current_user: User = Depends(get_current_user)):
        if current_user.role != required_role and current_user.role != "admin":
            raise AuthorizationException(f"Role '{required_role}' required")
        
        return current_user
    
    return role_checker


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> dict:
    """Verify and decode JWT token"""
    
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise AuthenticationException("Token has expired")
    except jwt.JWTError:
        raise AuthenticationException("Invalid token") 
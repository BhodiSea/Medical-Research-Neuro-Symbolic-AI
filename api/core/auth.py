"""
Authentication and authorization for Medical Research AI API
"""

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, List
import jwt
import logging
from datetime import datetime, timedelta, timezone
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from .config import get_settings
from .exceptions import AuthenticationException, AuthorizationException
from ..database.models import User
from ..database.connection import get_db_session

logger = logging.getLogger(__name__)
security = HTTPBearer()
settings = get_settings()

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db_session)
) -> User:
    """Get current authenticated user from JWT token"""
    
    try:
        # Decode JWT token
        token = credentials.credentials
        payload = verify_token(token)
        
        # Extract user email from token
        email: str = payload.get("sub")
        if email is None:
            raise AuthenticationException("Token missing subject")
        
        # Get user from database
        user = db.query(User).filter(User.email == email).first()
        if user is None:
            raise AuthenticationException("User not found")
        
        # Check if user is active
        if not user.is_active:
            raise AuthenticationException("User account is disabled")
        
        # Update last login timestamp
        user.last_login = datetime.now(timezone.utc)
        db.commit()
        
        logger.debug(f"User authenticated: {email}")
        return user
        
    except AuthenticationException:
        raise
    except Exception as e:
        logger.error(f"Authentication failed: {str(e)}")
        raise AuthenticationException("Invalid authentication credentials")


async def get_optional_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db_session)
) -> Optional[User]:
    """Get current user if authenticated, otherwise return None"""
    
    if not credentials:
        return None
    
    try:
        return await get_current_user(credentials, db)
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
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "access"
    })
    
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> dict:
    """Verify and decode JWT token"""
    
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        
        # Check token type
        if payload.get("type") != "access":
            raise AuthenticationException("Invalid token type")
        
        return payload
        
    except jwt.ExpiredSignatureError:
        raise AuthenticationException("Token has expired")
    except jwt.InvalidTokenError:
        raise AuthenticationException("Invalid token")
    except Exception as e:
        logger.error(f"Token verification error: {str(e)}")
        raise AuthenticationException("Token verification failed")


def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)


def authenticate_user_credentials(email: str, password: str, db: Session) -> Optional[User]:
    """Authenticate user with email and password"""
    
    user = db.query(User).filter(User.email == email).first()
    if not user:
        return None
    
    if not verify_password(password, user.hashed_password):
        return None
    
    if not user.is_active:
        return None
    
    # Update last login
    user.last_login = datetime.now(timezone.utc)
    db.commit()
    
    logger.info(f"User authenticated: {email}")
    return user 
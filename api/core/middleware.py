"""
Middleware for PremedPro AI API
"""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import time
import uuid
import logging
from typing import Dict, Any
import asyncio
from collections import defaultdict, deque
from datetime import datetime, timedelta

from .config import get_settings
from .exceptions import RateLimitException

logger = logging.getLogger(__name__)
settings = get_settings()


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses"""
    
    async def dispatch(self, request: Request, call_next):
        """Add security headers"""
        
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        # Content Security Policy (adjust based on your needs)
        if settings.is_production:
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "font-src 'self'; "
                "connect-src 'self'; "
                "frame-ancestors 'none'"
            )
        
        # HSTS header for production
        if settings.is_production:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests with timing and context"""
    
    async def dispatch(self, request: Request, call_next):
        """Log request details and timing"""
        
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Start timing
        start_time = time.time()
        
        # Log request
        logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "query_params": str(request.query_params),
                "client_ip": request.client.host if request.client else "unknown",
                "user_agent": request.headers.get("user-agent", "unknown")
            }
        )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            # Log response
            logger.info(
                f"Request completed: {request.method} {request.url.path} - "
                f"Status: {response.status_code} - Time: {process_time:.3f}s",
                extra={
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "process_time": process_time
                }
            )
            
            return response
            
        except Exception as e:
            # Calculate processing time for failed requests
            process_time = time.time() - start_time
            
            # Log error
            logger.error(
                f"Request failed: {request.method} {request.url.path} - "
                f"Error: {str(e)} - Time: {process_time:.3f}s",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                    "process_time": process_time
                },
                exc_info=True
            )
            
            raise


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware using in-memory storage"""
    
    def __init__(self, app, requests_per_minute: int = 100):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.window_size = 60  # seconds
        self.client_requests: Dict[str, deque] = defaultdict(deque)
        self.cleanup_interval = 300  # Clean up every 5 minutes
        self.last_cleanup = time.time()
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting"""
        
        # Try to get user ID from request state (set by auth middleware)
        if hasattr(request.state, "user_id"):
            return f"user:{request.state.user_id}"
        
        # Fall back to IP address
        if request.client:
            return f"ip:{request.client.host}"
        
        return "unknown"
    
    def _cleanup_old_requests(self):
        """Clean up old request records"""
        
        current_time = time.time()
        cutoff_time = current_time - self.window_size
        
        # Clean up old requests
        for client_id in list(self.client_requests.keys()):
            requests = self.client_requests[client_id]
            
            # Remove old requests
            while requests and requests[0] < cutoff_time:
                requests.popleft()
            
            # Remove empty deques
            if not requests:
                del self.client_requests[client_id]
        
        self.last_cleanup = current_time
    
    def _is_rate_limited(self, client_id: str) -> bool:
        """Check if client is rate limited"""
        
        current_time = time.time()
        cutoff_time = current_time - self.window_size
        
        # Get client's requests
        requests = self.client_requests[client_id]
        
        # Remove old requests
        while requests and requests[0] < cutoff_time:
            requests.popleft()
        
        # Check rate limit
        if len(requests) >= self.requests_per_minute:
            return True
        
        # Add current request
        requests.append(current_time)
        return False
    
    async def dispatch(self, request: Request, call_next):
        """Apply rate limiting"""
        
        # Skip rate limiting for health checks
        if request.url.path.startswith("/health"):
            return await call_next(request)
        
        # Clean up old requests periodically
        current_time = time.time()
        if current_time - self.last_cleanup > self.cleanup_interval:
            self._cleanup_old_requests()
        
        # Check rate limit
        client_id = self._get_client_id(request)
        
        if self._is_rate_limited(client_id):
            logger.warning(
                f"Rate limit exceeded for client: {client_id}",
                extra={
                    "client_id": client_id,
                    "path": request.url.path,
                    "method": request.method
                }
            )
            
            return JSONResponse(
                status_code=429,
                content={
                    "error": True,
                    "error_code": "RATE_LIMIT_EXCEEDED",
                    "message": f"Rate limit exceeded. Maximum {self.requests_per_minute} requests per minute.",
                    "retry_after": 60
                },
                headers={"Retry-After": "60"}
            )
        
        return await call_next(request)


# Rate limiting function for specific endpoints
async def rate_limit(request: Request, limit: int = 20, window: int = 60):
    """Apply rate limiting to specific endpoints"""
    
    # This is a placeholder that can be enhanced with Redis for production
    # For now, we'll use the middleware for global rate limiting
    pass


class CORSPolicyMiddleware(BaseHTTPMiddleware):
    """Enhanced CORS policy middleware"""
    
    async def dispatch(self, request: Request, call_next):
        """Handle CORS with enhanced security"""
        
        response = await call_next(request)
        
        # Get origin from request
        origin = request.headers.get("origin")
        
        if origin and origin in settings.CORS_ORIGINS:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
        elif not settings.is_production:
            # Allow all origins in development
            response.headers["Access-Control-Allow-Origin"] = "*"
        
        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Global error handling middleware"""
    
    async def dispatch(self, request: Request, call_next):
        """Handle unhandled exceptions"""
        
        try:
            return await call_next(request)
        except Exception as e:
            request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
            
            logger.error(
                f"Unhandled exception in middleware: {str(e)}",
                extra={"request_id": request_id},
                exc_info=True
            )
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": True,
                    "error_code": "INTERNAL_SERVER_ERROR",
                    "message": "An internal server error occurred",
                    "request_id": request_id
                }
            ) 
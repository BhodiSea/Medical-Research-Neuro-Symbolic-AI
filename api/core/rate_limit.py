"""
Rate limiting utilities for PremedPro AI API
"""

from fastapi import Request
from typing import Dict, Any
import time
import asyncio
from collections import defaultdict, deque

from .exceptions import RateLimitException

# Simple in-memory rate limiting
# In production, use Redis for distributed rate limiting
_rate_limit_cache: Dict[str, deque] = defaultdict(deque)


async def rate_limit(
    request: Request,
    limit: int = 20,
    window: int = 60,
    key_func=None
) -> None:
    """
    Apply rate limiting to an endpoint
    
    Args:
        request: FastAPI request object
        limit: Number of requests allowed
        window: Time window in seconds
        key_func: Function to generate cache key (defaults to client IP)
    """
    
    # Generate cache key
    if key_func:
        cache_key = key_func(request)
    else:
        # Default to client IP
        client_ip = request.client.host if request.client else "unknown"
        endpoint = request.url.path
        cache_key = f"{client_ip}:{endpoint}"
    
    current_time = time.time()
    cutoff_time = current_time - window
    
    # Get request times for this key
    request_times = _rate_limit_cache[cache_key]
    
    # Remove old requests
    while request_times and request_times[0] < cutoff_time:
        request_times.popleft()
    
    # Check if limit exceeded
    if len(request_times) >= limit:
        raise RateLimitException(
            f"Rate limit exceeded: {limit} requests per {window} seconds"
        )
    
    # Add current request
    request_times.append(current_time)


def get_rate_limit_status(request: Request, limit: int = 20, window: int = 60) -> Dict[str, Any]:
    """Get current rate limit status for a client"""
    
    client_ip = request.client.host if request.client else "unknown"
    endpoint = request.url.path
    cache_key = f"{client_ip}:{endpoint}"
    
    current_time = time.time()
    cutoff_time = current_time - window
    
    # Get request times for this key
    request_times = _rate_limit_cache[cache_key]
    
    # Remove old requests
    while request_times and request_times[0] < cutoff_time:
        request_times.popleft()
    
    remaining = max(0, limit - len(request_times))
    reset_time = int(current_time + window)
    
    return {
        "limit": limit,
        "remaining": remaining,
        "reset": reset_time,
        "window": window
    } 
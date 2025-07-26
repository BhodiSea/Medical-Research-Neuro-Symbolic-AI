"""
Health check endpoints for PremedPro AI API
"""

from fastapi import APIRouter, Request, Depends
from datetime import datetime
import time
import psutil
import logging
from typing import Dict, Any

from ..models.request_response import HealthCheckResponse
from ..core.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Store startup time for uptime calculation
_startup_time = time.time()


@router.get(
    "/",
    response_model=HealthCheckResponse,
    summary="Basic health check",
    description="Check if the API is running and responsive"
)
async def health_check() -> HealthCheckResponse:
    """Basic health check endpoint"""
    
    settings = get_settings()
    current_time = time.time()
    uptime_seconds = int(current_time - _startup_time)
    
    return HealthCheckResponse(
        status="healthy",
        version=settings.APP_VERSION,
        timestamp=datetime.utcnow(),
        services={
            "api": "healthy",
            "database": "unknown",  # Will be updated when DB is connected
            "medical_ai": "unknown"  # Will be updated when medical AI is connected
        },
        uptime_seconds=uptime_seconds
    )


@router.get(
    "/detailed",
    summary="Detailed health check",
    description="Detailed health check with system metrics and service status"
)
async def detailed_health_check(request: Request) -> Dict[str, Any]:
    """Detailed health check with system information"""
    
    settings = get_settings()
    current_time = time.time()
    uptime_seconds = int(current_time - _startup_time)
    
    # System metrics
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
    except Exception as e:
        logger.warning(f"Could not get system metrics: {e}")
        cpu_percent = 0
        memory = None
        disk = None
    
    # Check database connection
    database_status = "unknown"
    try:
        # This will be implemented when database is connected
        database_status = "not_configured"
    except Exception as e:
        database_status = "error"
        logger.error(f"Database health check failed: {e}")
    
    # Check medical AI service
    medical_ai_status = "unknown"
    try:
        medical_agent = getattr(request.app.state, 'medical_agent', None)
        if medical_agent:
            medical_ai_status = "healthy"
        else:
            medical_ai_status = "not_initialized"
    except Exception as e:
        medical_ai_status = "error"
        logger.error(f"Medical AI health check failed: {e}")
    
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": uptime_seconds,
        "uptime_human": format_uptime(uptime_seconds),
        "services": {
            "api": "healthy",
            "database": database_status,
            "medical_ai": medical_ai_status
        },
        "system_metrics": {
            "cpu_percent": cpu_percent if cpu_percent else "unavailable",
            "memory_percent": memory.percent if memory else "unavailable",
            "memory_available_gb": round(memory.available / (1024**3), 2) if memory else "unavailable",
            "disk_percent": round((disk.used / disk.total) * 100, 2) if disk else "unavailable",
            "disk_free_gb": round(disk.free / (1024**3), 2) if disk else "unavailable"
        },
        "configuration": {
            "debug_mode": settings.DEBUG,
            "cors_origins": len(settings.CORS_ORIGINS),
            "rate_limiting": True,
            "authentication": True,
            "medical_features": {
                "ethics_checking": settings.ENABLE_ETHICS_CHECKING,
                "bias_detection": settings.ENABLE_BIAS_DETECTION,
                "audit_logging": settings.ENABLE_AUDIT_LOGGING,
                "caching": settings.ENABLE_CACHING
            }
        }
    }


@router.get(
    "/readiness",
    summary="Readiness probe",
    description="Check if the service is ready to accept traffic"
)
async def readiness_check(request: Request) -> Dict[str, Any]:
    """Readiness probe for Kubernetes deployment"""
    
    ready = True
    services = {}
    
    # Check database readiness
    try:
        # Will implement database check when DB is connected
        services["database"] = "not_configured"
    except Exception as e:
        services["database"] = "not_ready"
        ready = False
        logger.error(f"Database not ready: {e}")
    
    # Check medical AI readiness
    try:
        medical_agent = getattr(request.app.state, 'medical_agent', None)
        if medical_agent:
            services["medical_ai"] = "ready"
        else:
            services["medical_ai"] = "not_ready"
            # Don't mark as not ready for now since medical AI is optional
    except Exception as e:
        services["medical_ai"] = "error"
        logger.error(f"Medical AI not ready: {e}")
    
    status_code = 200 if ready else 503
    
    return {
        "ready": ready,
        "services": services,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get(
    "/liveness",
    summary="Liveness probe",
    description="Check if the service is alive and should be restarted if not"
)
async def liveness_check() -> Dict[str, Any]:
    """Liveness probe for Kubernetes deployment"""
    
    # Basic liveness check - if we can respond, we're alive
    return {
        "alive": True,
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": int(time.time() - _startup_time)
    }


def format_uptime(seconds: int) -> str:
    """Format uptime in human-readable format"""
    
    days = seconds // (24 * 3600)
    seconds = seconds % (24 * 3600)
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 or not parts:
        parts.append(f"{seconds}s")
    
    return " ".join(parts) 
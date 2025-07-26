"""
PremedPro AI - Main FastAPI Application
Production-ready API for medical education and application assistance
"""

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import http_exception_handler
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any
import os

# Import route modules
from .routes import medical, user, application, health
from .core.config import Settings, get_settings
from .core.exceptions import PremedProException
from .core.logging_config import setup_logging
from .core.middleware import (
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
    RequestLoggingMiddleware
)

# Import database
from .database.connection import get_database
from .database.models import Base

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Global settings
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting PremedPro AI API...")
    
    # Initialize database
    try:
        database = get_database()
        # Create tables if they don't exist
        Base.metadata.create_all(bind=database.engine)
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise
    
    # Initialize medical AI components
    try:
        import sys
        import os
        # Add parent directory to Python path to import core modules
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        from core.medical_agents.premedpro_agent import create_premedpro_agent
        
        config = {
            "safety_mode": "high",
            "reasoning_mode": "adaptive",
            "max_query_length": 5000
        }
        app.state.medical_agent = create_premedpro_agent(config)
        logger.info("Medical AI agent initialized successfully")
    except Exception as e:
        logger.error(f"Medical AI initialization failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # Continue without medical agent for now
        app.state.medical_agent = None
    
    logger.info("PremedPro AI API startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down PremedPro AI API...")
    
    # Cleanup database connections
    try:
        database = get_database()
        database.close()
        logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Database cleanup error: {e}")
    
    logger.info("PremedPro AI API shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="PremedPro AI API",
    description="AI-powered medical education and application assistance platform",
    version="0.1.0",
    docs_url="/docs" if settings.ENVIRONMENT == "development" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT == "development" else None,
    openapi_url="/openapi.json" if settings.ENVIRONMENT == "development" else None,
    lifespan=lifespan
)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["*"],
)

# Custom middleware
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(RequestLoggingMiddleware)


# Exception handlers
@app.exception_handler(PremedProException)
async def premedpro_exception_handler(request: Request, exc: PremedProException):
    """Handle custom PremedPro exceptions"""
    logger.error(f"PremedPro exception: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "error_code": exc.error_code,
            "message": exc.detail,
            "request_id": getattr(request.state, "request_id", "unknown")
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.error(f"Unhandled exception [{request_id}]: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "error_code": "INTERNAL_SERVER_ERROR",
            "message": "An internal server error occurred",
            "request_id": request_id
        }
    )


# Include routers with API versioning
app.include_router(
    health.router,
    prefix="/health",
    tags=["Health Check"]
)

app.include_router(
    user.router,
    prefix="/api/v1/user",
    tags=["User Management"]
)

app.include_router(
    medical.router,
    prefix="/api/v1/medical",
    tags=["Medical AI"]
)

app.include_router(
    application.router,
    prefix="/api/v1/application",
    tags=["Application Review"]
)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "PremedPro AI API",
        "version": "0.1.0",
        "description": "AI-powered medical education and application assistance platform",
        "status": "operational",
        "documentation": "/docs" if settings.ENVIRONMENT == "development" else "Contact support for API documentation"
    }


# API information endpoint
@app.get("/api/v1/info")
async def api_info():
    """API information and capabilities"""
    return {
        "api_version": "1.0",
        "capabilities": [
            "medical_query_processing",
            "application_review",
            "study_planning",
            "school_matching",
            "interview_preparation"
        ],
        "features": {
            "medical_ai": app.state.medical_agent is not None,
            "ethics_engine": True,
            "knowledge_graph": True,
            "hybrid_reasoning": True
        },
        "limits": {
            "max_query_length": 5000,
            "rate_limit": "100 requests per minute",
            "max_file_size": "10MB"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.ENVIRONMENT == "development",
        log_level="info"
    ) 
"""
Configuration management for PremedPro AI API
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional
import os
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True
    )
    
    # Basic application settings
    APP_NAME: str = "PremedPro AI API"
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    
    # Security settings
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # CORS settings
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    ALLOWED_HOSTS: List[str] = ["localhost", "127.0.0.1"]
    
    # Database settings
    DATABASE_URL: str = "sqlite:///./premedpro_ai.db"
    DATABASE_POOL_SIZE: int = 5
    DATABASE_MAX_OVERFLOW: int = 10
    DATABASE_POOL_TIMEOUT: int = 30
    
    # Redis settings (for caching and rate limiting)
    REDIS_URL: Optional[str] = None
    REDIS_PASSWORD: Optional[str] = None
    REDIS_DB: int = 0
    
    # Rate limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60  # seconds
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    LOG_FILE: Optional[str] = None
    
    # Medical AI settings
    MEDICAL_AI_MODEL_PATH: Optional[str] = None
    MEDICAL_AI_CONFIDENCE_THRESHOLD: float = 0.7
    MEDICAL_AI_MAX_QUERY_LENGTH: int = 5000
    
    # OpenAI settings (if using external LLM)
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4"
    OPENAI_MAX_TOKENS: int = 2000
    
    # File upload settings
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_FILE_TYPES: List[str] = [".pdf", ".docx", ".txt"]
    UPLOAD_FOLDER: str = "./uploads"
    
    # Medical knowledge settings
    MEDICAL_KB_UPDATE_INTERVAL: int = 24  # hours
    MEDICAL_KB_SOURCE_URLS: List[str] = []
    
    # Ethics and compliance
    ENABLE_ETHICS_CHECKING: bool = True
    ENABLE_BIAS_DETECTION: bool = True
    REQUIRE_MEDICAL_DISCLAIMERS: bool = True
    ENABLE_AUDIT_LOGGING: bool = True
    
    # Performance settings
    ENABLE_CACHING: bool = True
    CACHE_TTL: int = 300  # seconds
    ENABLE_COMPRESSION: bool = True
    
    # External services
    SENDGRID_API_KEY: Optional[str] = None
    SLACK_WEBHOOK_URL: Optional[str] = None
    
    # Feature flags
    ENABLE_APPLICATION_REVIEW: bool = True
    ENABLE_STUDY_PLANNING: bool = True
    ENABLE_SCHOOL_MATCHING: bool = True
    ENABLE_INTERVIEW_PREP: bool = True
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.ENVIRONMENT.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.ENVIRONMENT.lower() == "development"
    
    @property
    def database_url_async(self) -> str:
        """Get async database URL"""
        if self.DATABASE_URL.startswith("sqlite"):
            return self.DATABASE_URL.replace("sqlite://", "sqlite+aiosqlite://")
        elif self.DATABASE_URL.startswith("postgresql"):
            return self.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
        return self.DATABASE_URL


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings"""
    return Settings()


# Environment-specific settings
def get_database_url() -> str:
    """Get database URL based on environment"""
    settings = get_settings()
    
    if settings.is_production:
        # Production should use PostgreSQL
        if not settings.DATABASE_URL.startswith("postgresql"):
            raise ValueError("Production environment requires PostgreSQL database")
    
    return settings.DATABASE_URL


def get_cors_origins() -> List[str]:
    """Get CORS origins based on environment"""
    settings = get_settings()
    
    if settings.is_production:
        # Production should have specific allowed origins
        return [origin for origin in settings.CORS_ORIGINS if not origin.startswith("http://localhost")]
    
    return settings.CORS_ORIGINS 
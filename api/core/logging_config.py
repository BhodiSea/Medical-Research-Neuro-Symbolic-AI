"""
Logging configuration for Medical Research AI API
"""

import logging
import logging.config
import sys
from datetime import datetime
from typing import Dict, Any
import json

from .config import get_settings


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id
        
        if hasattr(record, "user_id"):
            log_entry["user_id"] = record.user_id
        
        if hasattr(record, "endpoint"):
            log_entry["endpoint"] = record.endpoint
        
        return json.dumps(log_entry)


def setup_logging() -> None:
    """Setup logging configuration"""
    
    settings = get_settings()
    
    # Configure logging based on environment
    if settings.LOG_FORMAT.lower() == "json":
        formatter_class = "api.core.logging_config.JSONFormatter"
        format_string = None
    else:
        formatter_class = "logging.Formatter"
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    config: Dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "class": formatter_class,
            },
            "detailed": {
                "class": formatter_class,
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": settings.LOG_LEVEL,
                "formatter": "default",
                "stream": sys.stdout
            }
        },
        "loggers": {
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False
            },
            "uvicorn.access": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False
            },
            "fastapi": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False
            },
            "api": {
                "level": settings.LOG_LEVEL,
                "handlers": ["console"],
                "propagate": False
            },
            "core": {
                "level": settings.LOG_LEVEL,
                "handlers": ["console"],
                "propagate": False
            }
        },
        "root": {
            "level": settings.LOG_LEVEL,
            "handlers": ["console"]
        }
    }
    
    # Add file handler if log file is specified
    if settings.LOG_FILE:
        config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": settings.LOG_LEVEL,
            "formatter": "detailed",
            "filename": settings.LOG_FILE,
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5
        }
        
        # Add file handler to all loggers
        for logger_config in config["loggers"].values():
            if "handlers" in logger_config:
                logger_config["handlers"].append("file")
        
        config["root"]["handlers"].append("file")
    
    # Add format string for non-JSON formatters
    if format_string:
        config["formatters"]["default"]["format"] = format_string
        config["formatters"]["detailed"]["format"] = (
            "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s"
        )
    
    logging.config.dictConfig(config)
    
    # Set log level for third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


class RequestContextFilter(logging.Filter):
    """Add request context to log records"""
    
    def __init__(self, request_id: str = None, user_id: str = None, endpoint: str = None):
        super().__init__()
        self.request_id = request_id
        self.user_id = user_id
        self.endpoint = endpoint
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add context fields to log record"""
        
        if self.request_id:
            record.request_id = self.request_id
        
        if self.user_id:
            record.user_id = self.user_id
        
        if self.endpoint:
            record.endpoint = self.endpoint
        
        return True


def get_logger_with_context(
    name: str,
    request_id: str = None,
    user_id: str = None,
    endpoint: str = None
) -> logging.Logger:
    """Get logger with request context"""
    
    logger = logging.getLogger(name)
    
    # Add context filter
    context_filter = RequestContextFilter(
        request_id=request_id,
        user_id=user_id,
        endpoint=endpoint
    )
    
    logger.addFilter(context_filter)
    return logger 
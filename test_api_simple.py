#!/usr/bin/env python3
"""
Simple test API to debug startup issues
"""

from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime

# Create simple FastAPI app
app = FastAPI(title="Medical Research AI API Test", version="0.1.0")

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    message: str

@app.get("/")
async def root():
    return {"message": "Medical Research AI API is running!", "status": "ok"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        message="Simple API test is working"
    )

@app.get("/test")
async def test_endpoint():
    return {
        "test": "successful",
        "endpoints": ["GET /", "GET /health", "GET /test"],
        "next_steps": "Integrate with medical agent"
    }

if __name__ == "__main__":
    import uvicorn
    print("🧪 Starting Simple API Test...")
    print("📍 Available at: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000) 
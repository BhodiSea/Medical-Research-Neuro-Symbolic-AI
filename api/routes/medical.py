"""
Medical AI endpoints for PremedPro AI API
"""

from fastapi import APIRouter, HTTPException, Depends, Request, status, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Any, Optional
import logging
import asyncio
from datetime import datetime

from ..core.auth import get_current_user, require_permissions
from ..core.rate_limit import rate_limit
from ..models.request_response import (
    MedicalQueryRequest,
    MedicalQueryResponse,
    ErrorResponse
)
from ..database.models import User, MedicalQuery
from ..database.repositories import MedicalQueryRepository, get_medical_query_repository

logger = logging.getLogger(__name__)
router = APIRouter()


# Medical query endpoint
@router.post(
    "/query",
    response_model=MedicalQueryResponse,
    summary="Process medical query",
    description="Process a medical education query using our hybrid AI system"
)
async def process_medical_query(
    request: MedicalQueryRequest,
    current_user: User = Depends(get_current_user),
    api_request: Request = None
) -> MedicalQueryResponse:
    """
    Process medical query using our PremedPro AI agent
    
    Features:
    - Hybrid neuro-symbolic reasoning
    - Medical ethics compliance checking
    - Confidence scoring and uncertainty quantification
    - Source attribution and medical disclaimers
    """
    
    # Rate limiting
    await rate_limit(api_request, limit=20, window=60)  # 20 requests per minute
    
    try:
        # Validate query
        if len(request.query.strip()) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query cannot be empty"
            )
        
        if len(request.query) > 5000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query too long. Maximum 5000 characters allowed."
            )
        
        # Get medical agent from app state
        medical_agent = api_request.app.state.medical_agent
        if not medical_agent:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Medical AI service is currently unavailable"
            )
        
        # Process query with medical agent
        logger.info(f"Processing medical query for user {current_user.id}: {request.query[:100]}...")
        
        # Prepare context for the medical agent
        context = {
            "user_id": current_user.id,
            "user_role": current_user.role,
            "query_type": request.query_type,
            "urgency": request.urgency,
            "context": request.context,
            "has_personal_data": request.contains_personal_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Call the medical agent (our existing PremedPro agent)
        agent_response = await medical_agent.process_query(
            query=request.query,
            context=context
        )
        
        # Extract response components
        medical_response = agent_response.get("response", {})
        confidence_score = agent_response.get("confidence", 0.0)
        reasoning_path = agent_response.get("reasoning_path", [])
        ethical_compliance = agent_response.get("ethical_compliance", True)
        sources = agent_response.get("sources", [])
        
        # Generate medical disclaimer if required
        medical_disclaimer = None
        if confidence_score > 0.5:  # Add disclaimer for medical advice
            medical_disclaimer = (
                "This AI-generated response is for educational purposes only and "
                "should not be considered as professional medical advice, diagnosis, "
                "or treatment. Always consult with qualified healthcare professionals "
                "for medical concerns."
            )
        
        # Store query in database
        db_query = MedicalQuery(
            user_id=current_user.id,
            query_text=request.query,
            response_data=agent_response,
            confidence_score=confidence_score,
            ethical_compliance=ethical_compliance,
            processing_time_ms=agent_response.get("processing_time", 0),
            query_type=request.query_type
        )
        
        # TODO: Implement database storage
        # For now, create a mock saved query
        class MockQuery:
            def __init__(self):
                self.id = 1
        saved_query = MockQuery()
        
        # Prepare response
        response = MedicalQueryResponse(
            query_id=str(saved_query.id),
            query=request.query,
            response=medical_response,
            confidence_score=confidence_score,
            ethical_compliance=ethical_compliance,
            reasoning_path=reasoning_path,
            sources=sources,
            medical_disclaimer=medical_disclaimer,
            processing_time_ms=agent_response.get("processing_time", 0),
            timestamp=datetime.utcnow()
        )
        
        logger.info(f"Successfully processed medical query {saved_query.id} with confidence {confidence_score}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing medical query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your medical query"
        )


# Medical query history
@router.get(
    "/history",
    response_model=List[MedicalQueryResponse],
    summary="Get user's medical query history"
)
async def get_medical_history(
    current_user: User = Depends(get_current_user),
    limit: int = 20,
    offset: int = 0
) -> List[MedicalQueryResponse]:
    """Get user's medical query history"""
    
    try:
        # TODO: Implement database retrieval
        queries = []  # Placeholder empty list
        
        responses = []
        for query in queries:
            response_data = query.response_data or {}
            responses.append(MedicalQueryResponse(
                query_id=str(query.id),
                query=query.query_text,
                response=response_data.get("response", {}),
                confidence_score=query.confidence_score,
                ethical_compliance=query.ethical_compliance,
                reasoning_path=response_data.get("reasoning_path", []),
                sources=response_data.get("sources", []),
                medical_disclaimer=response_data.get("medical_disclaimer"),
                processing_time_ms=query.processing_time_ms,
                timestamp=query.created_at
            ))
        
        return responses
        
    except Exception as e:
        logger.error(f"Error retrieving medical history: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving medical query history"
        )


# Medical knowledge validation endpoint
@router.post(
    "/validate",
    summary="Validate medical information",
    description="Validate medical facts and claims using our knowledge base"
)
async def validate_medical_information(
    request_data: Dict[str, str],
    current_user: User = Depends(get_current_user),
    api_request: Request = None
):
    """Validate medical information against our knowledge base"""
    
    await rate_limit(api_request, limit=30, window=60)
    
    try:
        # Get medical agent
        medical_agent = api_request.app.state.medical_agent
        if not medical_agent:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Medical validation service is currently unavailable"
            )
        
        # Extract claim from request data
        claim = request_data.get("claim", "")
        
        # Validate the claim
        validation_result = await medical_agent.validate_medical_claim(
            claim=claim,
            user_context={"user_id": current_user.id}
        )
        
        return {
            "claim": claim,
            "is_valid": validation_result.get("is_valid", False),
            "confidence": validation_result.get("confidence", 0.0),
            "sources": validation_result.get("sources", []),
            "corrections": validation_result.get("corrections", []),
            "warnings": validation_result.get("warnings", [])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating medical information: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error validating medical information"
        )


# Medical image analysis (for future expansion)
@router.post(
    "/analyze-image",
    summary="Analyze medical educational images",
    description="Analyze medical diagrams and educational images"
)
async def analyze_medical_image(
    file: UploadFile = File(...),
    description: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    api_request: Request = None
):
    """Analyze medical educational images (future feature)"""
    
    await rate_limit(api_request, limit=10, window=60)
    
    # For now, return a placeholder response
    return {
        "message": "Medical image analysis is not yet implemented",
        "filename": file.filename,
        "content_type": file.content_type,
        "description": description,
        "status": "feature_not_available"
    }


# Medical glossary lookup
@router.get(
    "/glossary/{term}",
    summary="Look up medical term definition",
    description="Get definition and explanation of medical terms"
)
async def lookup_medical_term(
    term: str,
    current_user: User = Depends(get_current_user),
    api_request: Request = None
):
    """Look up medical term in glossary"""
    
    await rate_limit(api_request, limit=50, window=60)
    
    try:
        # Get medical agent
        medical_agent = api_request.app.state.medical_agent
        if not medical_agent:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Medical glossary service is currently unavailable"
            )
        
        # Look up the term
        definition_result = await medical_agent.lookup_term(
            term=term,
            context={"educational": True}
        )
        
        return {
            "term": term,
            "definition": definition_result.get("definition", "Term not found"),
            "synonyms": definition_result.get("synonyms", []),
            "related_terms": definition_result.get("related_terms", []),
            "sources": definition_result.get("sources", []),
            "confidence": definition_result.get("confidence", 0.0)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error looking up medical term: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error looking up medical term"
        ) 
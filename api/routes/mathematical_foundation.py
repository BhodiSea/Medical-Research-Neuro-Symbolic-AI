"""
Mathematical Foundation API Routes
Provides endpoints for Julia quantum models and AutoDock visualization integration
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import logging
import yaml
from pathlib import Path

# Import mathematical foundation components
try:
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent / "math_foundation"))
    from python_wrapper import create_math_foundation
    from autodock_integration import AutoDockIntegration
    MATH_FOUNDATION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Mathematical foundation not available: {e}")
    MATH_FOUNDATION_AVAILABLE = False

# Import medical agent
try:
    sys.path.append(str(Path(__file__).parent.parent.parent / "core" / "medical_agents"))
    from mathematical_foundation_agent import create_mathematical_foundation_agent
    AGENT_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Mathematical foundation agent not available: {e}")
    AGENT_AVAILABLE = False

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mathematical-foundation", tags=["mathematical-foundation"])

# Load configuration
def load_math_foundation_config() -> Dict[str, Any]:
    """Load mathematical foundation configuration"""
    config_path = Path(__file__).parent.parent.parent / "config" / "mathematical_foundation.yaml"
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Failed to load mathematical foundation config: {e}")
        return {}

# Pydantic models
class QuantumAnalysisRequest(BaseModel):
    query: str = Field(..., description="Medical query for quantum analysis")
    confidence: float = Field(0.5, ge=0.0, le=1.0, description="Base confidence level")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")

class MolecularAnalysisRequest(BaseModel):
    query: str = Field(..., description="Medical query for molecular analysis")
    receptor_file: Optional[str] = Field(None, description="Path to receptor PDB file")
    ligand_smiles: Optional[str] = Field(None, description="SMILES string for ligand")
    binding_site: Optional[Dict[str, float]] = Field(None, description="Binding site coordinates")

class ThermodynamicAnalysisRequest(BaseModel):
    query: str = Field(..., description="Medical query for thermodynamic analysis")
    temperature: float = Field(1.0, description="Temperature for entropy calculation")
    information_content: List[float] = Field(default_factory=list, description="Information content values")

class MathematicalFoundationStatus(BaseModel):
    julia_available: bool
    autodock_available: bool
    agent_available: bool
    capabilities: Dict[str, bool]
    configuration_loaded: bool

# Global instances
math_foundation_agent = None
math_config = None

def get_math_foundation_agent():
    """Get or create mathematical foundation agent"""
    global math_foundation_agent, math_config
    
    if math_foundation_agent is None and AGENT_AVAILABLE:
        math_config = load_math_foundation_config()
        math_foundation_agent = create_mathematical_foundation_agent(math_config)
    
    return math_foundation_agent

@router.get("/status", response_model=MathematicalFoundationStatus)
async def get_mathematical_foundation_status():
    """Get mathematical foundation system status"""
    agent = get_math_foundation_agent()
    
    capabilities = {}
    if agent:
        capabilities = agent.capabilities
    
    return MathematicalFoundationStatus(
        julia_available=MATH_FOUNDATION_AVAILABLE,
        autodock_available=MATH_FOUNDATION_AVAILABLE,
        agent_available=AGENT_AVAILABLE,
        capabilities=capabilities,
        configuration_loaded=math_config is not None
    )

@router.post("/quantum-analysis")
async def perform_quantum_analysis(request: QuantumAnalysisRequest):
    """Perform quantum uncertainty quantification analysis"""
    if not MATH_FOUNDATION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Mathematical foundation not available")
    
    try:
        agent = get_math_foundation_agent()
        if not agent:
            raise HTTPException(status_code=503, detail="Mathematical foundation agent not available")
        
        # Create context for analysis
        context = {
            "confidence": request.confidence,
            **request.context
        }
        
        # Perform quantum analysis
        result = agent._apply_quantum_analysis(request.query, context)
        
        return {
            "status": "success",
            "query": request.query,
            "quantum_analysis": result,
            "timestamp": "2024-01-01T00:00:00Z"  # Mock timestamp
        }
        
    except Exception as e:
        logger.error(f"Quantum analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Quantum analysis failed: {str(e)}")

@router.post("/molecular-analysis")
async def perform_molecular_analysis(request: MolecularAnalysisRequest):
    """Perform molecular docking analysis"""
    if not MATH_FOUNDATION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Mathematical foundation not available")
    
    try:
        agent = get_math_foundation_agent()
        if not agent:
            raise HTTPException(status_code=503, detail="Mathematical foundation agent not available")
        
        # Create context for analysis
        context = {
            "receptor_file": request.receptor_file,
            "ligand_smiles": request.ligand_smiles,
            "binding_site": request.binding_site
        }
        
        # Perform molecular analysis
        result = agent._apply_molecular_analysis(request.query, context)
        
        return {
            "status": "success",
            "query": request.query,
            "molecular_analysis": result,
            "timestamp": "2024-01-01T00:00:00Z"  # Mock timestamp
        }
        
    except Exception as e:
        logger.error(f"Molecular analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Molecular analysis failed: {str(e)}")

@router.post("/thermodynamic-analysis")
async def perform_thermodynamic_analysis(request: ThermodynamicAnalysisRequest):
    """Perform thermodynamic entropy analysis"""
    if not MATH_FOUNDATION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Mathematical foundation not available")
    
    try:
        agent = get_math_foundation_agent()
        if not agent:
            raise HTTPException(status_code=503, detail="Mathematical foundation agent not available")
        
        # Create context for analysis
        context = {
            "temperature": request.temperature,
            "information_content": request.information_content
        }
        
        # Perform thermodynamic analysis
        result = agent._apply_thermodynamic_analysis(request.query, context)
        
        return {
            "status": "success",
            "query": request.query,
            "thermodynamic_analysis": result,
            "timestamp": "2024-01-01T00:00:00Z"  # Mock timestamp
        }
        
    except Exception as e:
        logger.error(f"Thermodynamic analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Thermodynamic analysis failed: {str(e)}")

@router.post("/comprehensive-analysis")
async def perform_comprehensive_analysis(request: QuantumAnalysisRequest):
    """Perform comprehensive mathematical foundation analysis"""
    if not MATH_FOUNDATION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Mathematical foundation not available")
    
    try:
        agent = get_math_foundation_agent()
        if not agent:
            raise HTTPException(status_code=503, detail="Mathematical foundation agent not available")
        
        # Create context for analysis
        context = {
            "confidence": request.confidence,
            **request.context
        }
        
        # Perform comprehensive analysis
        result = agent.process_medical_query(request.query, context)
        
        return {
            "status": "success",
            "query": request.query,
            "comprehensive_analysis": result,
            "timestamp": "2024-01-01T00:00:00Z"  # Mock timestamp
        }
        
    except Exception as e:
        logger.error(f"Comprehensive analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Comprehensive analysis failed: {str(e)}")

@router.get("/capabilities")
async def get_capabilities():
    """Get available mathematical foundation capabilities"""
    agent = get_math_foundation_agent()
    
    if not agent:
        return {
            "capabilities": {},
            "available": False,
            "message": "Mathematical foundation agent not available"
        }
    
    return {
        "capabilities": agent.capabilities,
        "available": agent.initialized,
        "agent_status": agent.get_agent_status()
    }

@router.get("/configuration")
async def get_configuration():
    """Get mathematical foundation configuration"""
    config = load_math_foundation_config()
    
    return {
        "configuration": config,
        "loaded": bool(config)
    } 
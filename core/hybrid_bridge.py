"""
Hybrid Bridge for Medical Research AI
Fuses symbolic and neural reasoning for comprehensive medical research analysis
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import asyncio
from dataclasses import dataclass
from enum import Enum

# Import custom components
from .symbolic.custom_logic import MedicalLogicEngine
from .neural.custom_neural import MedicalNeuralReasoner, HybridNeuralSymbolic

logger = logging.getLogger(__name__)

class ReasoningMode(Enum):
    """Different modes of hybrid reasoning"""
    SYMBOLIC_FIRST = "symbolic_first"
    NEURAL_FIRST = "neural_first"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"

@dataclass
class ReasoningResult:
    """Structured result from hybrid reasoning"""
    query: str
    final_answer: Dict[str, Any]
    confidence: float
    reasoning_path: List[str]
    symbolic_contribution: float
    neural_contribution: float
    ethical_compliance: bool
    uncertainty_bounds: Optional[Tuple[float, float]]
    interpretability_score: float

class HybridReasoningEngine:
    """
    Main hybrid reasoning engine that orchestrates symbolic and neural components
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.symbolic_engine = None
        self.neural_reasoner = None
        self.hybrid_bridge = None
        
        # Reasoning strategy
        self.default_mode = ReasoningMode(config.get("reasoning_mode", "adaptive"))
        
        # Performance metrics
        self.reasoning_history = []
        self.performance_metrics = {
            "symbolic_success_rate": 0.0,
            "neural_success_rate": 0.0,
            "hybrid_success_rate": 0.0,
            "average_confidence": 0.0
        }
    
    def initialize(self) -> None:
        """Initialize all reasoning components"""
        try:
            # Initialize symbolic engine
            from .symbolic.custom_logic import create_medical_logic_engine
            self.symbolic_engine = create_medical_logic_engine(
                self.config.get("symbolic_config_path")
            )
            
            # Initialize neural reasoner
            from .neural.custom_neural import create_medical_neural_reasoner
            self.neural_reasoner = create_medical_neural_reasoner(
                self.config.get("neural_config")
            )
            
            # Create hybrid bridge
            self.hybrid_bridge = HybridNeuralSymbolic(
                self.neural_reasoner, 
                self.symbolic_engine
            )
            
            logger.info("Hybrid reasoning engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize hybrid reasoning engine: {e}")
            raise
    
    async def reason(self, query: str, context: Dict[str, Any], mode: Optional[ReasoningMode] = None) -> ReasoningResult:
        """
        Main reasoning method that coordinates symbolic and neural components
        """
        reasoning_mode = mode or self.default_mode
        
        try:
            # Preprocess query and context
            processed_query, processed_context = self._preprocess_input(query, context)
            
            # Choose reasoning strategy based on mode
            if reasoning_mode == ReasoningMode.SYMBOLIC_FIRST:
                result = await self._symbolic_first_reasoning(processed_query, processed_context)
            elif reasoning_mode == ReasoningMode.NEURAL_FIRST:
                result = await self._neural_first_reasoning(processed_query, processed_context)
            elif reasoning_mode == ReasoningMode.PARALLEL:
                result = await self._parallel_reasoning(processed_query, processed_context)
            else:  # ADAPTIVE
                result = await self._adaptive_reasoning(processed_query, processed_context)
            
            # Post-process and validate result
            validated_result = self._validate_and_postprocess(result, processed_context)
            
            # Update performance metrics
            self._update_metrics(validated_result)
            
            return validated_result
            
        except Exception as e:
            logger.error(f"Error in hybrid reasoning: {e}")
            return self._create_error_result(query, str(e))
    
    def _preprocess_input(self, query: str, context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Preprocess query and context for reasoning"""
        # Clean and normalize query
        processed_query = query.strip().lower()
        
        # Enhance context with metadata
        processed_context = context.copy()
        processed_context.update({
            "query_length": len(query),
            "has_medical_terms": self._detect_medical_terms(query),
            "urgency_level": self._assess_urgency(query),
            "privacy_sensitivity": self._assess_privacy_sensitivity(query, context)
        })
        
        return processed_query, processed_context
    
    async def _symbolic_first_reasoning(self, query: str, context: Dict[str, Any]) -> ReasoningResult:
        """Symbolic-first reasoning strategy"""
        reasoning_path = ["symbolic_primary"]
        
        # Try symbolic reasoning first
        symbolic_result = self.symbolic_engine.process_medical_query(query, context)
        
        # Check if symbolic reasoning is sufficient
        if symbolic_result.get("confidence", 0) > 0.8:
            return self._create_result_from_symbolic(query, symbolic_result, reasoning_path)
        
        # Enhance with neural reasoning
        reasoning_path.append("neural_enhancement")
        neural_result = self.neural_reasoner.process_medical_input(query, context)
        
        # Fuse results
        fused_result = self.hybrid_bridge.fuse_reasoning(query, context)
        
        return self._create_hybrid_result(query, fused_result, reasoning_path)
    
    async def _neural_first_reasoning(self, query: str, context: Dict[str, Any]) -> ReasoningResult:
        """Neural-first reasoning strategy"""
        reasoning_path = ["neural_primary"]
        
        # Try neural reasoning first
        neural_result = self.neural_reasoner.process_medical_input(query, context)
        
        # Always validate with symbolic reasoning for safety
        reasoning_path.append("symbolic_validation")
        symbolic_result = self.symbolic_engine.process_medical_query(query, context)
        
        # Check for conflicts or safety issues
        if symbolic_result.get("status") == "blocked":
            return self._create_result_from_symbolic(query, symbolic_result, reasoning_path)
        
        # Fuse results
        fused_result = self.hybrid_bridge.fuse_reasoning(query, context)
        
        return self._create_hybrid_result(query, fused_result, reasoning_path)
    
    async def _parallel_reasoning(self, query: str, context: Dict[str, Any]) -> ReasoningResult:
        """Parallel reasoning strategy"""
        reasoning_path = ["parallel_execution"]
        
        # Execute both reasoning modes in parallel
        symbolic_task = asyncio.create_task(
            self._run_symbolic_async(query, context)
        )
        neural_task = asyncio.create_task(
            self._run_neural_async(query, context)
        )
        
        # Wait for both to complete
        symbolic_result, neural_result = await asyncio.gather(
            symbolic_task, neural_task, return_exceptions=True
        )
        
        # Handle exceptions
        if isinstance(symbolic_result, Exception):
            logger.error(f"Symbolic reasoning failed: {symbolic_result}")
            symbolic_result = {"error": str(symbolic_result), "confidence": 0.0}
        
        if isinstance(neural_result, Exception):
            logger.error(f"Neural reasoning failed: {neural_result}")
            neural_result = {"error": str(neural_result), "model_confidence": 0.0}
        
        # Fuse results
        fused_result = self.hybrid_bridge.fuse_reasoning(query, context)
        
        return self._create_hybrid_result(query, fused_result, reasoning_path)
    
    async def _adaptive_reasoning(self, query: str, context: Dict[str, Any]) -> ReasoningResult:
        """Adaptive reasoning strategy based on query characteristics"""
        reasoning_path = ["adaptive_selection"]
        
        # Analyze query to choose best strategy
        strategy = self._select_adaptive_strategy(query, context)
        reasoning_path.append(f"selected_{strategy.value}")
        
        if strategy == ReasoningMode.SYMBOLIC_FIRST:
            return await self._symbolic_first_reasoning(query, context)
        elif strategy == ReasoningMode.NEURAL_FIRST:
            return await self._neural_first_reasoning(query, context)
        else:
            return await self._parallel_reasoning(query, context)
    
    def _select_adaptive_strategy(self, query: str, context: Dict[str, Any]) -> ReasoningMode:
        """Select the best reasoning strategy based on query analysis"""
        # Rule-based selection (can be enhanced with ML)
        
        # High privacy sensitivity → Symbolic first
        if context.get("privacy_sensitivity", 0) > 0.8:
            return ReasoningMode.SYMBOLIC_FIRST
        
        # Medical diagnosis keywords → Symbolic first (safety)
        diagnosis_keywords = ["diagnose", "symptoms", "treatment", "medication"]
        if any(keyword in query.lower() for keyword in diagnosis_keywords):
            return ReasoningMode.SYMBOLIC_FIRST
        
        # Research/analytical queries → Neural first
        research_keywords = ["analyze", "research", "compare", "investigate", "examine"]
        if any(keyword in query.lower() for keyword in research_keywords):
            return ReasoningMode.NEURAL_FIRST
        
        # Complex queries → Parallel
        if len(query.split()) > 20 or context.get("complexity_score", 0) > 0.7:
            return ReasoningMode.PARALLEL
        
        # Default to symbolic first for safety
        return ReasoningMode.SYMBOLIC_FIRST
    
    async def _run_symbolic_async(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run symbolic reasoning asynchronously"""
        # Wrap synchronous call in async
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.symbolic_engine.process_medical_query, 
            query, 
            context
        )
    
    async def _run_neural_async(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run neural reasoning asynchronously"""
        # Wrap synchronous call in async
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.neural_reasoner.process_medical_input,
            query,
            context
        )
    
    def _create_result_from_symbolic(self, query: str, symbolic_result: Dict[str, Any], reasoning_path: List[str]) -> ReasoningResult:
        """Create ReasoningResult from symbolic reasoning only"""
        return ReasoningResult(
            query=query,
            final_answer=symbolic_result,
            confidence=symbolic_result.get("confidence", 0.0),
            reasoning_path=reasoning_path,
            symbolic_contribution=1.0,
            neural_contribution=0.0,
            ethical_compliance=symbolic_result.get("ethical_assessment", {}).get("medically_appropriate", True),
            uncertainty_bounds=None,
            interpretability_score=0.9  # Symbolic reasoning is highly interpretable
        )
    
    def _create_hybrid_result(self, query: str, fused_result: Dict[str, Any], reasoning_path: List[str]) -> ReasoningResult:
        """Create ReasoningResult from hybrid reasoning"""
        neural_result = fused_result.get("neural_reasoning", {})
        symbolic_result = fused_result.get("symbolic_reasoning", {})
        
        # Calculate contribution weights
        neural_conf = neural_result.get("model_confidence", 0.0)
        symbolic_conf = symbolic_result.get("confidence", 0.0)
        total_conf = neural_conf + symbolic_conf
        
        if total_conf > 0:
            neural_contrib = neural_conf / total_conf
            symbolic_contrib = symbolic_conf / total_conf
        else:
            neural_contrib = 0.5
            symbolic_contrib = 0.5
        
        # Extract uncertainty bounds from neural component
        confidence_intervals = neural_result.get("confidence_intervals", {})
        uncertainty_bounds = None
        if confidence_intervals.get("95_percent"):
            bounds = confidence_intervals["95_percent"]
            uncertainty_bounds = (
                min(bounds.get("lower", [])) if bounds.get("lower") else None,
                max(bounds.get("upper", [])) if bounds.get("upper") else None
            )
        
        return ReasoningResult(
            query=query,
            final_answer=fused_result.get("fused_result", {}),
            confidence=fused_result.get("fused_confidence", 0.0),
            reasoning_path=reasoning_path,
            symbolic_contribution=symbolic_contrib,
            neural_contribution=neural_contrib,
            ethical_compliance=symbolic_result.get("ethical_assessment", {}).get("medically_appropriate", True),
            uncertainty_bounds=uncertainty_bounds,
            interpretability_score=0.7 * symbolic_contrib + 0.3 * neural_contrib
        )
    
    def _create_error_result(self, query: str, error_message: str) -> ReasoningResult:
        """Create error result"""
        return ReasoningResult(
            query=query,
            final_answer={"error": error_message, "status": "failed"},
            confidence=0.0,
            reasoning_path=["error"],
            symbolic_contribution=0.0,
            neural_contribution=0.0,
            ethical_compliance=False,
            uncertainty_bounds=None,
            interpretability_score=0.0
        )
    
    def _validate_and_postprocess(self, result: ReasoningResult, context: Dict[str, Any]) -> ReasoningResult:
        """Validate and post-process reasoning result"""
        # Add result to history
        self.reasoning_history.append({
            "query": result.query,
            "confidence": result.confidence,
            "ethical_compliance": result.ethical_compliance,
            "reasoning_path": result.reasoning_path
        })
        
        # Limit history size
        if len(self.reasoning_history) > 1000:
            self.reasoning_history = self.reasoning_history[-1000:]
        
        return result
    
    def _update_metrics(self, result: ReasoningResult) -> None:
        """Update performance metrics"""
        if len(self.reasoning_history) > 0:
            recent_results = self.reasoning_history[-100:]  # Last 100 results
            
            # Calculate success rates
            total_results = len(recent_results)
            symbolic_successes = sum(1 for r in recent_results if r.get("confidence", 0) > 0.7)
            
            self.performance_metrics.update({
                "average_confidence": sum(r.get("confidence", 0) for r in recent_results) / total_results,
                "ethical_compliance_rate": sum(1 for r in recent_results if r.get("ethical_compliance", False)) / total_results
            })
    
    def _detect_medical_terms(self, query: str) -> bool:
        """Detect if query contains medical terminology"""
        medical_terms = [
            "symptom", "diagnosis", "treatment", "medication", "anatomy", 
            "physiology", "pathology", "disease", "syndrome", "therapy"
        ]
        return any(term in query.lower() for term in medical_terms)
    
    def _assess_urgency(self, query: str) -> float:
        """Assess urgency level of query (0.0 to 1.0)"""
        urgent_keywords = ["emergency", "urgent", "immediate", "critical", "severe"]
        urgency_score = sum(1 for keyword in urgent_keywords if keyword in query.lower())
        return min(urgency_score / len(urgent_keywords), 1.0)
    
    def _assess_privacy_sensitivity(self, query: str, context: Dict[str, Any]) -> float:
        """Assess privacy sensitivity (0.0 to 1.0)"""
        personal_indicators = ["my", "i have", "patient", "personal", "private"]
        sensitivity_score = sum(1 for indicator in personal_indicators if indicator in query.lower())
        
        # Factor in context
        if context.get("has_personal_data", False):
            sensitivity_score += 2
        
        return min(sensitivity_score / 5.0, 1.0)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "symbolic_engine_status": self.symbolic_engine.get_system_status() if self.symbolic_engine else "not_initialized",
            "neural_reasoner_status": "initialized" if self.neural_reasoner else "not_initialized",
            "hybrid_bridge_status": "initialized" if self.hybrid_bridge else "not_initialized",
            "reasoning_history_size": len(self.reasoning_history),
            "performance_metrics": self.performance_metrics,
            "default_reasoning_mode": self.default_mode.value
        }

# Factory function for creating hybrid reasoning engine
def create_hybrid_reasoning_engine(config: Optional[Dict[str, Any]] = None) -> HybridReasoningEngine:
    """Factory function to create and initialize hybrid reasoning engine"""
    if config is None:
        config = {
            "reasoning_mode": "adaptive",
            "symbolic_config_path": None,
            "neural_config": {
                "input_dim": 512,
                "output_dim": 256,
                "medical_vocab_size": 10000,
                "embedding_dim": 512
            }
        }
    
    engine = HybridReasoningEngine(config)
    engine.initialize()
    return engine 
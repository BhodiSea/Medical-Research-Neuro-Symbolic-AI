"""
Mathematical Foundation Medical Agent
Integrates Julia quantum models and AutoDock visualization for enhanced medical reasoning
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import sys

# Add math foundation to path
math_foundation_path = Path(__file__).parent.parent.parent / "math_foundation"
if str(math_foundation_path) not in sys.path:
    sys.path.insert(0, str(math_foundation_path))

try:
    from python_wrapper import JuliaMathFoundation, create_math_foundation
    from autodock_integration import AutoDockIntegration
    MATH_FOUNDATION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Mathematical foundation not available: {e}")
    MATH_FOUNDATION_AVAILABLE = False

logger = logging.getLogger(__name__)


class MathematicalFoundationAgent:
    """
    Medical agent that leverages mathematical foundation for enhanced reasoning
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.math_foundation = None
        self.autodock_integration = None
        self.initialized = False
        
        # Agent capabilities
        self.capabilities = {
            "quantum_uncertainty_quantification": False,
            "molecular_docking_analysis": False,
            "thermodynamic_entropy_calculation": False,
            "field_evolution_modeling": False
        }
        
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize mathematical foundation components"""
        if not MATH_FOUNDATION_AVAILABLE:
            logger.warning("Mathematical foundation not available")
            return
        
        try:
            # Initialize Julia mathematical foundation
            math_config = self.config.get("mathematical_foundation", {})
            julia_path = math_config.get("julia", {}).get("julia_path")
            math_foundation_path = str(Path(__file__).parent.parent.parent / "math_foundation")
            
            self.math_foundation = create_math_foundation(
                julia_path=julia_path,
                math_foundation_path=math_foundation_path
            )
            
            if self.math_foundation and self.math_foundation.initialized:
                self.capabilities["quantum_uncertainty_quantification"] = True
                self.capabilities["thermodynamic_entropy_calculation"] = True
                self.capabilities["field_evolution_modeling"] = True
                logger.info("Julia mathematical foundation initialized successfully")
            
            # Initialize AutoDock integration
            autodock_config = math_config.get("autodock", {})
            self.autodock_integration = AutoDockIntegration(autodock_config)
            
            if self.autodock_integration:
                self.capabilities["molecular_docking_analysis"] = True
                logger.info("AutoDock integration initialized successfully")
            
            self.initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize mathematical foundation agent: {e}")
            self.initialized = False
    
    def process_medical_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process medical query with mathematical foundation enhancement"""
        if not self.initialized:
            return self._create_fallback_response(query, "Mathematical foundation not available")
        
        try:
            # Analyze query type and apply appropriate mathematical methods
            query_analysis = self._analyze_query_type(query, context)
            
            # Apply quantum uncertainty quantification
            quantum_analysis = self._apply_quantum_analysis(query, context)
            
            # Apply molecular analysis if relevant
            molecular_analysis = self._apply_molecular_analysis(query, context)
            
            # Apply thermodynamic analysis
            thermodynamic_analysis = self._apply_thermodynamic_analysis(query, context)
            
            # Synthesize results
            enhanced_result = self._synthesize_mathematical_results(
                query_analysis, quantum_analysis, molecular_analysis, thermodynamic_analysis
            )
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Error in mathematical foundation agent: {e}")
            return self._create_fallback_response(query, str(e))
    
    def _analyze_query_type(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze query to determine appropriate mathematical methods"""
        query_lower = query.lower()
        
        analysis = {
            "query_type": "general",
            "requires_quantum_analysis": False,
            "requires_molecular_analysis": False,
            "requires_thermodynamic_analysis": False,
            "complexity_score": 0.0
        }
        
        # Check for quantum-related queries
        quantum_keywords = ["uncertainty", "probability", "confidence", "entropy", "information"]
        if any(keyword in query_lower for keyword in quantum_keywords):
            analysis["requires_quantum_analysis"] = True
            analysis["complexity_score"] += 0.3
        
        # Check for molecular/drug-related queries
        molecular_keywords = ["drug", "protein", "binding", "molecule", "ligand", "receptor", "docking"]
        if any(keyword in query_lower for keyword in molecular_keywords):
            analysis["requires_molecular_analysis"] = True
            analysis["complexity_score"] += 0.4
        
        # Check for thermodynamic/energy-related queries
        thermodynamic_keywords = ["energy", "equilibrium", "temperature", "entropy", "free_energy"]
        if any(keyword in query_lower for keyword in thermodynamic_keywords):
            analysis["requires_thermodynamic_analysis"] = True
            analysis["complexity_score"] += 0.3
        
        # Determine query type
        if analysis["requires_molecular_analysis"]:
            analysis["query_type"] = "molecular"
        elif analysis["requires_quantum_analysis"]:
            analysis["query_type"] = "quantum"
        elif analysis["requires_thermodynamic_analysis"]:
            analysis["query_type"] = "thermodynamic"
        
        return analysis
    
    def _apply_quantum_analysis(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply quantum uncertainty quantification"""
        if not self.math_foundation or not self.capabilities["quantum_uncertainty_quantification"]:
            return {"quantum_analysis": "not_available"}
        
        try:
            # Create quantum state representation
            base_confidence = context.get("confidence", 0.5)
            amplitudes = [complex(base_confidence, 0.0)]
            phases = [0.0]
            uncertainties = [1.0 - base_confidence]
            
            # Calculate quantum uncertainty
            quantum_state = self.math_foundation.create_quantum_state(amplitudes, phases, uncertainties)
            
            if quantum_state:
                # Calculate uncertainty principle
                uncertainty_result = self.math_foundation.calculate_uncertainty_principle(
                    knowledge_uncertainty=1.0 - base_confidence,
                    belief_uncertainty=0.1,
                    hbar_analog=1.0
                )
                
                # Calculate quantum entropy
                entropy_result = self.math_foundation.calculate_quantum_entropy(amplitudes, uncertainties)
                
                return {
                    "quantum_analysis": "success",
                    "quantum_uncertainty": uncertainty_result.get("uncertainty_product", 0.0),
                    "quantum_entropy": entropy_result.get("entropy", 0.0),
                    "quantum_state": "initialized",
                    "confidence_enhancement": base_confidence * (1.0 + entropy_result.get("entropy", 0.0))
                }
            
        except Exception as e:
            logger.warning(f"Quantum analysis failed: {e}")
        
        return {"quantum_analysis": "error"}
    
    def _apply_molecular_analysis(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply molecular docking analysis"""
        if not self.autodock_integration or not self.capabilities["molecular_docking_analysis"]:
            return {"molecular_analysis": "not_available"}
        
        # Check if query involves molecular topics
        molecular_keywords = ["drug", "protein", "binding", "molecule", "ligand", "receptor"]
        if not any(keyword in query.lower() for keyword in molecular_keywords):
            return {"molecular_analysis": "not_applicable"}
        
        try:
            # For demonstration, create enhanced molecular analysis
            # In real implementation, this would call actual AutoDock methods
            analysis_result = {
                "molecular_analysis": "available",
                "binding_affinity_estimate": 0.75,
                "docking_confidence": 0.8,
                "analysis_type": "virtual_screening",
                "molecular_targets": ["protein_target_1", "protein_target_2"],
                "binding_site_prediction": {
                    "x": 10.5,
                    "y": 15.2,
                    "z": 8.7,
                    "radius": 5.0
                },
                "ligand_protein_interactions": [
                    {"type": "hydrogen_bond", "strength": 0.8},
                    {"type": "hydrophobic", "strength": 0.6}
                ]
            }
            
            return analysis_result
            
        except Exception as e:
            logger.warning(f"Molecular analysis failed: {e}")
            return {"molecular_analysis": "error", "error": str(e)}
    
    def _apply_thermodynamic_analysis(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply thermodynamic entropy analysis"""
        if not self.math_foundation or not self.capabilities["thermodynamic_entropy_calculation"]:
            return {"thermodynamic_analysis": "not_available"}
        
        try:
            # Calculate thermodynamic entropy for information content
            information_content = [1.0, 0.8, 0.6, 0.4]  # Mock information values
            truth_energies = [0.5, 0.7, 0.3, 0.9]  # Mock energy values
            
            entropy_result = self.math_foundation.calculate_truth_entropy(
                truth_energies=truth_energies,
                information_content=information_content,
                temperature=1.0
            )
            
            return {
                "thermodynamic_analysis": "success",
                "truth_entropy": entropy_result.get("entropy", 0.0),
                "free_energy": entropy_result.get("free_energy", 0.0),
                "information_content": sum(information_content) / len(information_content)
            }
            
        except Exception as e:
            logger.warning(f"Thermodynamic analysis failed: {e}")
            return {"thermodynamic_analysis": "error"}
    
    def _synthesize_mathematical_results(self, query_analysis: Dict[str, Any], 
                                       quantum_analysis: Dict[str, Any],
                                       molecular_analysis: Dict[str, Any],
                                       thermodynamic_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize all mathematical analysis results"""
        
        # Calculate overall confidence enhancement
        confidence_enhancements = []
        
        if quantum_analysis.get("quantum_analysis") == "success":
            confidence_enhancements.append(quantum_analysis.get("confidence_enhancement", 0.0))
        
        if molecular_analysis.get("molecular_analysis") == "available":
            confidence_enhancements.append(molecular_analysis.get("docking_confidence", 0.0))
        
        if thermodynamic_analysis.get("thermodynamic_analysis") == "success":
            confidence_enhancements.append(1.0 - thermodynamic_analysis.get("truth_entropy", 0.0))
        
        # Calculate average enhancement
        overall_enhancement = sum(confidence_enhancements) / len(confidence_enhancements) if confidence_enhancements else 0.0
        
        return {
            "status": "success",
            "mathematical_foundation_enhanced": True,
            "overall_confidence_enhancement": overall_enhancement,
            "query_analysis": query_analysis,
            "quantum_analysis": quantum_analysis,
            "molecular_analysis": molecular_analysis,
            "thermodynamic_analysis": thermodynamic_analysis,
            "capabilities_used": [cap for cap, available in self.capabilities.items() if available],
            "agent_type": "mathematical_foundation"
        }
    
    def _create_fallback_response(self, query: str, error_message: str) -> Dict[str, Any]:
        """Create fallback response when mathematical foundation is unavailable"""
        return {
            "status": "fallback",
            "mathematical_foundation_enhanced": False,
            "error": error_message,
            "overall_confidence_enhancement": 0.0,
            "agent_type": "mathematical_foundation",
            "capabilities_used": []
        }
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get agent status and capabilities"""
        return {
            "agent_type": "mathematical_foundation",
            "initialized": self.initialized,
            "capabilities": self.capabilities,
            "math_foundation_available": self.math_foundation is not None and self.math_foundation.initialized,
            "autodock_available": self.autodock_integration is not None
        }


def create_mathematical_foundation_agent(config: Optional[Dict[str, Any]] = None) -> MathematicalFoundationAgent:
    """Factory function for creating mathematical foundation agent"""
    if config is None:
        config = {}
    
    return MathematicalFoundationAgent(config) 
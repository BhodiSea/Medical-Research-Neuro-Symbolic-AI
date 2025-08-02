"""
Medical SymbolicAI Wrapper
Specialized integration for medical reasoning using SymbolicAI framework
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import asyncio

# Import the base SymbolicAI integration
from .symbolicai_integration import SymbolicAIIntegration, SYMBOLICAI_AVAILABLE

logger = logging.getLogger(__name__)

class MedicalSymbolicAIWrapper:
    """Medical-specific wrapper for SymbolicAI integration"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.symbolic_ai = SymbolicAIIntegration(config)
        self.medical_reasoning_templates = self._load_medical_templates()
        self.safety_constraints = self._load_safety_constraints()
        
        logger.info(f"Medical SymbolicAI Wrapper initialized. Available: {SYMBOLICAI_AVAILABLE}")
    
    def _load_medical_templates(self) -> Dict[str, str]:
        """Load medical reasoning templates for different query types"""
        return {
            "anatomy_query": """
            Provide educational anatomical information about: {query}
            
            Guidelines:
            - Focus on educational content only
            - Reference standard medical texts
            - Include anatomical relationships
            - Avoid diagnostic implications
            - Use precise medical terminology
            
            Context: {context}
            """,
            
            "research_analysis": """
            Analyze the medical research query: {query}
            
            Approach:
            - Review relevant peer-reviewed literature
            - Identify key research findings
            - Assess evidence quality
            - Highlight limitations and gaps
            - Suggest future research directions
            
            Medical Context: {context}
            Research Focus: {research_focus}
            """,
            
            "pathophysiology_explanation": """
            Explain the pathophysiological mechanisms for: {query}
            
            Structure:
            - Normal physiological processes
            - Disease/disorder mechanisms
            - Molecular and cellular basis
            - System-level effects
            - Clinical correlations (educational only)
            
            Context: {context}
            Complexity Level: {complexity_level}
            """,
            
            "drug_mechanism": """
            Provide educational information about drug mechanisms: {query}
            
            Content Areas:
            - Mechanism of action
            - Pharmacokinetics
            - Therapeutic targets
            - Known side effects (general)
            - Research developments
            
            Important: This is educational information only. Dosage and treatment decisions require professional consultation.
            
            Context: {context}
            """,
            
            "clinical_safety_check": """
            Evaluate the safety implications of this medical query: {query}
            
            Assessment Areas:
            - Potential for misinterpretation
            - Risk of self-medication
            - Need for professional consultation
            - Educational vs. clinical advice boundary
            - Vulnerable population considerations
            
            Context: {context}
            Safety Priority: High
            """
        }
    
    def _load_safety_constraints(self) -> Dict[str, Any]:
        """Load medical safety constraints for SymbolicAI processing"""
        return {
            "prohibited_outputs": [
                "specific medical diagnoses",
                "treatment recommendations",
                "medication dosages", 
                "emergency medical advice",
                "surgical procedures",
                "self-treatment protocols"
            ],
            "required_disclaimers": [
                "Educational purposes only",
                "Consult healthcare professionals for medical decisions",
                "Individual cases may vary",
                "Not a substitute for professional medical advice"
            ],
            "confidence_limits": {
                "personal_medical_queries": 0.3,  # Low confidence for personal queries
                "diagnostic_questions": 0.1,      # Very low confidence
                "educational_content": 0.8,       # Higher confidence for educational
                "research_analysis": 0.9          # Highest for research
            }
        }
    
    async def process_medical_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process medical queries with safety-first symbolic reasoning"""
        try:
            # Step 1: Safety assessment
            safety_result = await self._assess_query_safety(query, context)
            if safety_result["status"] == "blocked":
                return safety_result
            
            # Step 2: Query classification
            query_type = self._classify_query_type(query, context)
            
            # Step 3: Select appropriate template
            template = self._select_template(query_type, context)
            
            # Step 4: SymbolicAI processing
            symbolic_result = await self._process_with_symbolic_ai(query, template, context)
            
            # Step 5: Apply medical constraints
            constrained_result = self._apply_medical_constraints(symbolic_result, query_type, context)
            
            # Step 6: Generate final response
            return self._generate_medical_response(constrained_result, safety_result, context)
            
        except Exception as e:
            logger.error(f"Error in medical SymbolicAI processing: {e}")
            return self._generate_error_response(query, str(e))
    
    async def _assess_query_safety(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess safety implications of the medical query"""
        safety_issues = []
        confidence_limit = 1.0
        
        # Check for dangerous patterns
        dangerous_patterns = [
            ("emergency symptoms", ["chest pain", "difficulty breathing", "severe bleeding"]),
            ("self_diagnosis", ["do i have", "am i sick", "is this cancer"]),
            ("medication_dosage", ["how much", "dosage", "pills"]),
            ("treatment_advice", ["what should i do", "how to treat", "cure"])
        ]
        
        query_lower = query.lower()
        for issue_type, patterns in dangerous_patterns:
            if any(pattern in query_lower for pattern in patterns):
                safety_issues.append(issue_type)
                confidence_limit = min(confidence_limit, self.safety_constraints["confidence_limits"].get(issue_type, 0.5))
        
        # Determine if query should be blocked
        if "emergency symptoms" in safety_issues:
            return {
                "status": "blocked",
                "reason": "potential_medical_emergency",
                "message": "This appears to be a medical emergency. Please seek immediate medical attention.",
                "safety_issues": safety_issues
            }
        
        return {
            "status": "proceed_with_caution",
            "safety_issues": safety_issues,
            "confidence_limit": confidence_limit,
            "requires_disclaimers": len(safety_issues) > 0
        }
    
    def _classify_query_type(self, query: str, context: Dict[str, Any]) -> str:
        """Classify the type of medical query"""
        query_lower = query.lower()
        
        # Classification logic
        if any(kw in query_lower for kw in ["anatomy", "structure", "organ", "system"]):
            return "anatomy_query"
        elif any(kw in query_lower for kw in ["research", "study", "clinical trial", "evidence"]):
            return "research_analysis"
        elif any(kw in query_lower for kw in ["mechanism", "pathophysiology", "how does", "why does"]):
            return "pathophysiology_explanation"
        elif any(kw in query_lower for kw in ["drug", "medication", "pharmaceutical", "treatment"]):
            return "drug_mechanism"
        else:
            return "general_medical"
    
    def _select_template(self, query_type: str, context: Dict[str, Any]) -> str:
        """Select appropriate template for query processing"""
        template = self.medical_reasoning_templates.get(query_type, self.medical_reasoning_templates["anatomy_query"])
        
        # Add safety template if needed
        if context.get("high_risk", False):
            safety_template = self.medical_reasoning_templates["clinical_safety_check"]
            template = f"{safety_template}\n\nThen proceed with:\n{template}"
        
        return template
    
    async def _process_with_symbolic_ai(self, query: str, template: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process query using SymbolicAI with medical template"""
        if not SYMBOLICAI_AVAILABLE:
            return self._mock_symbolic_processing(query, template, context)
        
        try:
            # Format template with query and context
            formatted_prompt = template.format(
                query=query,
                context=context.get("medical_context", "General medical query"),
                research_focus=context.get("research_focus", "General research"),
                complexity_level=context.get("complexity_level", "Intermediate")
            )
            
            # Process with SymbolicAI
            result = self.symbolic_ai.process_medical_query_symbolic(
                formatted_prompt, context
            )
            
            return {
                "symbolic_result": result,
                "template_used": template,
                "processing_successful": True
            }
            
        except Exception as e:
            logger.error(f"SymbolicAI processing error: {e}")
            return self._mock_symbolic_processing(query, template, context)
    
    def _apply_medical_constraints(self, symbolic_result: Dict[str, Any], query_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply medical safety constraints to SymbolicAI results"""
        result = symbolic_result.copy()
        
        # Apply confidence limits based on query type
        if query_type == "research_analysis":
            max_confidence = self.safety_constraints["confidence_limits"]["research_analysis"]
        elif query_type == "anatomy_query":
            max_confidence = self.safety_constraints["confidence_limits"]["educational_content"]
        else:
            max_confidence = self.safety_constraints["confidence_limits"]["educational_content"]
        
        # Check for personal medical queries
        if context.get("is_personal_query", False):
            max_confidence = min(max_confidence, self.safety_constraints["confidence_limits"]["personal_medical_queries"])
        
        # Update confidence
        if "confidence" in result:
            result["confidence"] = min(result["confidence"], max_confidence)
        
        # Add required disclaimers
        result["disclaimers"] = self.safety_constraints["required_disclaimers"]
        
        # Flag prohibited content
        prohibited_flags = []
        if symbolic_result.get("symbolic_result", {}).get("symbolic_result", ""):
            content = str(symbolic_result["symbolic_result"]["symbolic_result"]).lower()
            for prohibited in self.safety_constraints["prohibited_outputs"]:
                if any(word in content for word in prohibited.split()):
                    prohibited_flags.append(prohibited)
        
        result["content_flags"] = prohibited_flags
        result["needs_review"] = len(prohibited_flags) > 0
        
        return result
    
    def _generate_medical_response(self, constrained_result: Dict[str, Any], safety_result: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final medical response with all safety measures"""
        # Base response structure
        response = {
            "query_processed": True,
            "safety_assessment": safety_result,
            "symbolic_reasoning": constrained_result,
            "medical_disclaimers": constrained_result.get("disclaimers", []),
            "confidence": constrained_result.get("confidence", 0.5),
            "requires_professional_consultation": safety_result.get("requires_disclaimers", False)
        }
        
        # Add content if safe
        if not constrained_result.get("needs_review", False):
            response["educational_content"] = constrained_result.get("symbolic_result", {}).get("symbolic_result", "")
            response["reasoning_path"] = constrained_result.get("symbolic_result", {}).get("reasoning_path", [])
        else:
            response["message"] = "Query requires additional safety review. Please consult healthcare professionals."
            response["confidence"] = 0.1
        
        # Add safety warnings if needed
        if safety_result.get("safety_issues"):
            response["safety_warnings"] = [
                f"Safety concern detected: {issue}" for issue in safety_result["safety_issues"]
            ]
        
        return response
    
    def _generate_error_response(self, query: str, error_message: str) -> Dict[str, Any]:
        """Generate error response for failed processing"""
        return {
            "query": query,
            "status": "processing_error",
            "error": error_message,
            "message": "Unable to process medical query safely. Please consult healthcare professionals.",
            "confidence": 0.0,
            "requires_professional_consultation": True
        }
    
    def _mock_symbolic_processing(self, query: str, template: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Mock processing when SymbolicAI is not available"""
        return {
            "symbolic_result": {
                "query": query,
                "symbolic_result": f"Mock medical symbolic reasoning for: {query}",
                "confidence": 0.5,
                "reasoning_path": ["Mock medical reasoning step 1", "Mock medical reasoning step 2"],
                "medical_context": context
            },
            "template_used": template,
            "processing_successful": False,
            "mock_mode": True
        }
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get status of SymbolicAI medical integration"""
        return {
            "symbolicai_available": SYMBOLICAI_AVAILABLE,
            "medical_templates_loaded": len(self.medical_reasoning_templates),
            "safety_constraints_loaded": len(self.safety_constraints),
            "integration_ready": SYMBOLICAI_AVAILABLE and len(self.medical_reasoning_templates) > 0
        }

# Factory function for creating medical SymbolicAI wrapper
def create_medical_symbolic_ai(config: Optional[Dict[str, Any]] = None) -> MedicalSymbolicAIWrapper:
    """Factory function to create medical SymbolicAI wrapper"""
    return MedicalSymbolicAIWrapper(config)
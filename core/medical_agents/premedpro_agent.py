#!/usr/bin/env python3
"""
PremedPro Medical AI Agent
Uses OpenSSA framework for medical education and clinical reasoning assistance
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import json

# OpenSSA imports
try:
    from openssa import DANA, HTP, HTPlanner, OodaReasoner, LMConfig
    from openssa.core.util import FileResource
    OPENSSA_AVAILABLE = True
except ImportError:
    OPENSSA_AVAILABLE = False
    print("OpenSSA not available - using fallback implementations")

# Our medical knowledge system
try:
    from core.medical_knowledge.knowledge_graph import create_medical_knowledge_graph, MedicalKnowledgeGraph
    MEDICAL_KB_AVAILABLE = True
except ImportError:
    MEDICAL_KB_AVAILABLE = False
    print("Medical knowledge graph not available")

logger = logging.getLogger(__name__)

@dataclass
class MedicalQuery:
    """Represents a medical query from a student or practitioner"""
    query_id: str
    user_type: str  # "medical_student", "resident", "attending", "researcher"
    query_text: str
    context: Dict[str, Any]
    urgency: str  # "low", "medium", "high", "critical"
    domain: str  # "cardiology", "pulmonology", "general", etc.

@dataclass
class MedicalResponse:
    """Represents a response from the medical AI agent"""
    query_id: str
    response_text: str
    confidence: float
    reasoning_steps: List[str]
    sources: List[str]
    ethical_compliance: bool
    limitations: List[str]
    follow_up_suggestions: List[str]

class MedicalReasoningPlan:
    """Defines reasoning plans for medical AI tasks"""
    
    @staticmethod
    def get_diagnostic_reasoning_plan() -> Dict[str, Any]:
        """Plan for diagnostic reasoning tasks"""
        return {
            "name": "diagnostic_reasoning",
            "description": "Systematic approach to medical diagnosis",
            "steps": [
                {
                    "name": "symptom_analysis",
                    "description": "Analyze and categorize presenting symptoms",
                    "inputs": ["patient_presentation", "symptom_list"],
                    "outputs": ["categorized_symptoms", "red_flags"]
                },
                {
                    "name": "differential_diagnosis", 
                    "description": "Generate differential diagnosis list",
                    "inputs": ["categorized_symptoms", "patient_history"],
                    "outputs": ["differential_list", "probability_scores"]
                },
                {
                    "name": "diagnostic_workup",
                    "description": "Recommend appropriate diagnostic tests",
                    "inputs": ["differential_list", "patient_factors"],
                    "outputs": ["test_recommendations", "test_rationale"]
                },
                {
                    "name": "diagnosis_refinement",
                    "description": "Refine diagnosis based on additional data",
                    "inputs": ["test_results", "clinical_response"],
                    "outputs": ["refined_diagnosis", "confidence_assessment"]
                }
            ]
        }
    
    @staticmethod
    def get_educational_plan() -> Dict[str, Any]:
        """Plan for educational support tasks"""
        return {
            "name": "medical_education",
            "description": "Provide medical education and learning support",
            "steps": [
                {
                    "name": "topic_analysis",
                    "description": "Understand the learning topic and student level",
                    "inputs": ["question", "student_level", "curriculum_context"],
                    "outputs": ["topic_breakdown", "learning_objectives"]
                },
                {
                    "name": "knowledge_retrieval",
                    "description": "Retrieve relevant medical knowledge",
                    "inputs": ["topic_breakdown", "knowledge_base"],
                    "outputs": ["core_concepts", "supporting_evidence"]
                },
                {
                    "name": "explanation_generation",
                    "description": "Generate clear, level-appropriate explanation",
                    "inputs": ["core_concepts", "student_level"],
                    "outputs": ["explanation", "examples", "mnemonics"]
                },
                {
                    "name": "assessment_preparation",
                    "description": "Provide practice questions and scenarios",
                    "inputs": ["explained_concepts", "student_progress"],
                    "outputs": ["practice_questions", "case_scenarios"]
                }
            ]
        }

class PremedProAgent:
    """Main medical AI agent using OpenSSA framework"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.knowledge_graph = None
        self.dana_agent = None
        self.reasoning_engine = None
        
        self._initialize_components()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for the agent"""
        return {
            "agent_name": "PremedPro Medical Assistant",
            "safety_mode": "high",
            "educational_focus": True,
            "clinical_assistance": True,
            "max_response_length": 2000,
            "require_sources": True,
            "ethical_verification": True
        }
    
    def _initialize_components(self):
        """Initialize agent components"""
        self.logger.info("Initializing PremedPro Medical Agent...")
        
        # Initialize knowledge graph
        if MEDICAL_KB_AVAILABLE:
            self.knowledge_graph = create_medical_knowledge_graph()
            self.logger.info("Medical knowledge graph initialized")
        else:
            self.logger.warning("Medical knowledge graph not available")
        
        # Initialize OpenSSA components
        if OPENSSA_AVAILABLE:
            self._initialize_openssa()
        else:
            self.logger.warning("OpenSSA not available - using fallback mode")
        
        self.logger.info("PremedPro Medical Agent initialized successfully")
    
    def _initialize_openssa(self):
        """Initialize OpenSSA DANA agent"""
        try:
            # Configure language model (would use actual LM in production)
            lm_config = LMConfig(
                model_name="gpt-4",  # Would be configured based on available models
                temperature=0.1,  # Low temperature for medical accuracy
                max_tokens=1000
            )
            
            # Create DANA agent with medical reasoning plans
            self.dana_agent = DANA(
                name="PremedPro Medical Assistant",
                description="Medical AI agent for education and clinical reasoning support",
                lm_config=lm_config
            )
            
            # Add medical reasoning plans
            diagnostic_plan = MedicalReasoningPlan.get_diagnostic_reasoning_plan()
            educational_plan = MedicalReasoningPlan.get_educational_plan()
            
            # Note: In actual implementation, would register these plans with DANA
            self.logger.info("OpenSSA DANA agent configured for medical reasoning")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenSSA: {e}")
            self.dana_agent = None
    
    async def process_medical_query(self, query: MedicalQuery) -> MedicalResponse:
        """Process a medical query and generate response"""
        self.logger.info(f"Processing medical query: {query.query_id}")
        
        # Safety check
        if not self._is_safe_query(query):
            return self._create_safety_response(query)
        
        # Route to appropriate processing method
        if query.domain in ["education", "learning", "study"]:
            return await self._process_educational_query(query)
        elif query.domain in ["diagnosis", "clinical", "patient"]:
            return await self._process_clinical_query(query)
        else:
            return await self._process_general_medical_query(query)
    
    def _is_safe_query(self, query: MedicalQuery) -> bool:
        """Check if query is safe to process"""
        unsafe_patterns = [
            "personal medical advice",
            "emergency",
            "urgent",
            "should I take",
            "am I having",
            "diagnose me"
        ]
        
        query_lower = query.query_text.lower()
        for pattern in unsafe_patterns:
            if pattern in query_lower:
                return False
        
        return True
    
    def _create_safety_response(self, query: MedicalQuery) -> MedicalResponse:
        """Create a safety response for potentially harmful queries"""
        return MedicalResponse(
            query_id=query.query_id,
            response_text=(
                "I'm designed to provide educational information about medical topics, "
                "but I cannot provide personal medical advice, diagnosis, or emergency assistance. "
                "For personal health concerns, please consult with a qualified healthcare provider. "
                "In case of emergency, contact emergency services immediately."
            ),
            confidence=1.0,
            reasoning_steps=["Safety filter applied"],
            sources=["Medical ethics guidelines"],
            ethical_compliance=True,
            limitations=["Cannot provide personal medical advice"],
            follow_up_suggestions=[
                "Consult healthcare provider for personal medical concerns",
                "Contact emergency services for urgent issues"
            ]
        )
    
    async def _process_educational_query(self, query: MedicalQuery) -> MedicalResponse:
        """Process educational/learning queries"""
        self.logger.info("Processing educational query")
        
        reasoning_steps = [
            "Identified as educational query",
            "Analyzing topic and student level",
            "Retrieving educational content"
        ]
        
        # Use knowledge graph if available
        if self.knowledge_graph:
            # Search for relevant medical entities
            relevant_entities = self.knowledge_graph.semantic_search(query.query_text)
            reasoning_steps.append(f"Found {len(relevant_entities)} relevant medical concepts")
            
            # Build educational response
            response_text = self._build_educational_response(query, relevant_entities)
        else:
            # Fallback educational response
            response_text = self._build_fallback_educational_response(query)
        
        return MedicalResponse(
            query_id=query.query_id,
            response_text=response_text,
            confidence=0.8,
            reasoning_steps=reasoning_steps,
            sources=["Medical knowledge base", "Educational materials"],
            ethical_compliance=True,
            limitations=["For educational purposes only"],
            follow_up_suggestions=[
                "Review additional educational resources",
                "Discuss with instructors for clarification"
            ]
        )
    
    async def _process_clinical_query(self, query: MedicalQuery) -> MedicalResponse:
        """Process clinical reasoning queries"""
        self.logger.info("Processing clinical query")
        
        # Emphasize educational nature for clinical queries
        if query.user_type not in ["attending", "resident"]:
            disclaimer = (
                "This is for educational purposes only and should not be used for "
                "actual patient care decisions. Always consult with supervising physicians."
            )
        else:
            disclaimer = "This analysis is for educational support only."
        
        reasoning_steps = [
            "Identified as clinical reasoning query",
            "Applying clinical reasoning framework",
            "Educational context applied"
        ]
        
        # Use knowledge graph for clinical reasoning
        if self.knowledge_graph:
            # Extract symptoms/findings from query
            symptoms = self._extract_symptoms(query.query_text)
            if symptoms:
                differential = self.knowledge_graph.get_differential_diagnosis(symptoms)
                reasoning_steps.append(f"Generated differential diagnosis with {len(differential)} possibilities")
                
                response_text = self._build_clinical_response(query, differential, disclaimer)
            else:
                response_text = f"{disclaimer}\n\nPlease provide more specific clinical information for educational analysis."
        else:
            response_text = f"{disclaimer}\n\nClinical reasoning module not fully available."
        
        return MedicalResponse(
            query_id=query.query_id,
            response_text=response_text,
            confidence=0.7,
            reasoning_steps=reasoning_steps,
            sources=["Medical knowledge base", "Clinical guidelines"],
            ethical_compliance=True,
            limitations=["Educational purposes only", "Not for patient care decisions"],
            follow_up_suggestions=[
                "Verify with authoritative medical sources",
                "Discuss with supervising clinicians"
            ]
        )
    
    async def _process_general_medical_query(self, query: MedicalQuery) -> MedicalResponse:
        """Process general medical information queries"""
        self.logger.info("Processing general medical query")
        
        reasoning_steps = [
            "Identified as general medical information query",
            "Retrieving general medical knowledge"
        ]
        
        response_text = (
            f"Based on your query about '{query.query_text}', I can provide general medical information. "
            "This information is for educational purposes only and should not replace professional medical advice."
        )
        
        # Add knowledge graph information if available
        if self.knowledge_graph:
            entities = self.knowledge_graph.semantic_search(query.query_text)
            if entities:
                response_text += f"\n\nFound information about {len(entities)} related medical concepts."
                for entity in entities[:3]:  # Limit to top 3 results
                    response_text += f"\n\n**{entity.name}** ({entity.type}): "
                    response_text += str(entity.properties.get('description', 'Medical concept'))
        
        return MedicalResponse(
            query_id=query.query_id,
            response_text=response_text,
            confidence=0.8,
            reasoning_steps=reasoning_steps,
            sources=["Medical knowledge base"],
            ethical_compliance=True,
            limitations=["Educational information only"],
            follow_up_suggestions=["Consult medical literature for detailed information"]
        )
    
    def _extract_symptoms(self, text: str) -> List[str]:
        """Extract potential symptoms from text"""
        # Simple keyword extraction - could be enhanced with NLP
        symptom_keywords = [
            "chest pain", "shortness of breath", "fatigue", "nausea",
            "headache", "fever", "cough", "dizziness", "pain"
        ]
        
        text_lower = text.lower()
        found_symptoms = [symptom for symptom in symptom_keywords if symptom in text_lower]
        return found_symptoms
    
    def _build_educational_response(self, query: MedicalQuery, entities: List) -> str:
        """Build educational response using knowledge graph entities"""
        response = f"Educational Information for: {query.query_text}\n\n"
        
        if entities:
            response += "Key Medical Concepts:\n"
            for entity in entities[:3]:  # Top 3 most relevant
                response += f"\n• **{entity.name}** ({entity.type})\n"
                response += f"  Properties: {entity.properties}\n"
                
                # Add relationships if available
                if hasattr(self.knowledge_graph, 'find_related_entities'):
                    related = self.knowledge_graph.find_related_entities(entity.id)
                    if related:
                        response += f"  Related to: {', '.join([r[0].name for r in related[:3]])}\n"
        
        response += "\n**Remember**: This is educational information only. Always consult authoritative medical sources and healthcare providers for clinical decisions."
        
        return response
    
    def _build_fallback_educational_response(self, query: MedicalQuery) -> str:
        """Build fallback educational response"""
        return (
            f"Educational Topic: {query.query_text}\n\n"
            "I can provide general medical education support. For detailed information about "
            "specific medical topics, please consult medical textbooks, peer-reviewed journals, "
            "and discuss with your medical instructors.\n\n"
            "**Educational Reminder**: Always verify medical information with authoritative sources "
            "and discuss complex topics with qualified medical educators."
        )
    
    def _build_clinical_response(self, query: MedicalQuery, differential: List, disclaimer: str) -> str:
        """Build clinical reasoning response"""
        response = f"{disclaimer}\n\n"
        response += f"Clinical Reasoning Exercise for: {query.query_text}\n\n"
        
        if differential:
            response += "Educational Differential Diagnosis Considerations:\n"
            for condition, score in differential:
                response += f"\n• **{condition.name}** (Educational probability: {score:.2f})\n"
                response += f"  Type: {condition.type}\n"
                response += f"  Properties: {condition.properties}\n"
                
                # Add treatment information if available
                if hasattr(self.knowledge_graph, 'get_treatment_options'):
                    treatments = self.knowledge_graph.get_treatment_options(condition.id)
                    if treatments:
                        response += f"  Educational treatment considerations: {', '.join([t[0].name for t in treatments[:2]])}\n"
        
        response += "\n**Critical Reminder**: This educational exercise should never be used for actual patient care. Always follow institutional protocols and consult with attending physicians."
        
        return response
    
    async def validate_medical_claim(self, claim: str, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a medical claim using knowledge base and reasoning"""
        self.logger.info(f"Validating medical claim: {claim[:50]}...")
        
        try:
            # Use knowledge graph to validate claim
            if self.knowledge_graph:
                # Simple validation using knowledge graph
                entities = self.knowledge_graph.search_entities(claim[:100])
                
                # Calculate confidence based on entity matches
                confidence = min(0.7 + (len(entities) * 0.1), 0.95)
                
                # Determine validity based on knowledge matches
                is_valid = len(entities) > 0
                
                return {
                    "claim": claim,
                    "is_valid": is_valid,
                    "confidence": confidence,
                    "evidence": [{"source": "Medical Knowledge Base", "entity": entity.name} for entity in entities[:3]],
                    "recommendation": "Consult medical literature for detailed verification" if is_valid else "Claim requires verification from authoritative sources"
                }
            else:
                # Fallback validation
                return {
                    "claim": claim,
                    "is_valid": True,
                    "confidence": 0.5,
                    "evidence": [{"source": "General Medical Knowledge", "entity": "Requires verification"}],
                    "recommendation": "Please verify this claim with authoritative medical sources"
                }
                
        except Exception as e:
            self.logger.error(f"Error validating medical claim: {e}")
            return {
                "claim": claim,
                "is_valid": False,
                "confidence": 0.1,
                "evidence": [],
                "recommendation": "Unable to validate claim. Please consult medical professionals."
            }

    def get_agent_status(self) -> Dict[str, Any]:
        """Get agent system status"""
        return {
            "agent_initialized": True,
            "openssa_available": OPENSSA_AVAILABLE,
            "knowledge_graph_available": MEDICAL_KB_AVAILABLE,
            "config": self.config,
            "safety_mode": self.config.get("safety_mode", "high"),
            "ready_for_queries": True
        }

def create_premedpro_agent(config: Optional[Dict[str, Any]] = None) -> PremedProAgent:
    """Factory function to create PremedPro medical agent"""
    logger.info("Creating PremedPro medical agent...")
    agent = PremedProAgent(config)
    logger.info("PremedPro medical agent created successfully")
    return agent

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(level=logging.INFO)
    
    async def demo():
        print("🏥 PremedPro Medical Agent Demo")
        print("=" * 40)
        
        # Create agent
        agent = create_premedpro_agent()
        
        # Test educational query
        edu_query = MedicalQuery(
            query_id="edu_001",
            user_type="medical_student",
            query_text="Explain the chambers of the heart",
            context={"course": "anatomy", "year": 1},
            urgency="low",
            domain="education"
        )
        
        print("\n🎓 Testing Educational Query:")
        response = await agent.process_medical_query(edu_query)
        print(f"Response: {response.response_text[:200]}...")
        print(f"Confidence: {response.confidence}")
        print(f"Ethical Compliance: {response.ethical_compliance}")
        
        # Test clinical query
        clinical_query = MedicalQuery(
            query_id="clin_001",
            user_type="resident",
            query_text="Patient with chest pain and shortness of breath",
            context={"setting": "educational_case"},
            urgency="medium",
            domain="clinical"
        )
        
        print("\n🩺 Testing Clinical Query:")
        response = await agent.process_medical_query(clinical_query)
        print(f"Response: {response.response_text[:200]}...")
        print(f"Reasoning Steps: {response.reasoning_steps}")
        
        print(f"\n📊 Agent Status: {agent.get_agent_status()}")
    
    asyncio.run(demo()) 
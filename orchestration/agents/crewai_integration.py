"""
CrewAI Integration Wrapper
Provides standardized interface for multi-agent orchestration and role-playing
"""

import sys
import os
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path

# Add CrewAI submodule to path
crewai_path = Path(__file__).parent / "crewai"
if str(crewai_path) not in sys.path:
    sys.path.insert(0, str(crewai_path))

try:
    # Import CrewAI components when available
    from crewai import Agent, Task, Crew, Process
    CREWAI_AVAILABLE = True
except ImportError as e:
    print(f"Warning: CrewAI not available: {e}")
    CREWAI_AVAILABLE = False


class CrewAIIntegration:
    """Integration wrapper for CrewAI multi-agent orchestration"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.agents = {}
        self.crews = {}
        self.active_tasks = {}
        
        if not CREWAI_AVAILABLE:
            print("Warning: CrewAI integration running in mock mode")
    
    def create_medical_agent(self, agent_id: str, role: str, domain_expertise: List[str], 
                           goal: str, backstory: str) -> Optional[Any]:
        """Create a medical domain expert agent"""
        if not CREWAI_AVAILABLE:
            return self._mock_agent(agent_id, role, domain_expertise)
        
        try:
            agent = Agent(
                role=role,
                goal=goal,
                backstory=backstory,
                verbose=self.config.get("verbose", False),
                allow_delegation=self.config.get("allow_delegation", False),
                # Add medical-specific configurations
                max_iter=self.config.get("max_iterations", 5),
                memory=self.config.get("enable_memory", True)
            )
            
            # Store agent with metadata
            self.agents[agent_id] = {
                "agent": agent,
                "role": role,
                "domain_expertise": domain_expertise,
                "goal": goal,
                "backstory": backstory,
                "created_at": self._get_timestamp()
            }
            
            return agent
        except Exception as e:
            print(f"Error creating medical agent {agent_id}: {e}")
            return None
    
    def create_tenth_man_agent(self, config: Dict[str, Any]) -> Optional[Any]:
        """Create the special 10th man (devil's advocate) agent"""
        if not CREWAI_AVAILABLE:
            return self._mock_agent("tenth_man", "devils_advocate", ["dissent", "critical_analysis"])
        
        try:
            tenth_man = Agent(
                role="Devil's Advocate",
                goal="Challenge consensus and identify potential flaws in group decisions",
                backstory="""You are the 10th man - a critical thinker whose role is to disagree 
                with group consensus when stakes are high. You draw from ethical memory to find 
                contrarian perspectives and ensure thorough consideration of alternatives.""",
                verbose=True,
                allow_delegation=False,
                max_iter=10,  # Allow more iterations for thorough analysis
                memory=True
            )
            
            self.agents["tenth_man"] = {
                "agent": tenth_man,
                "role": "devils_advocate",
                "domain_expertise": ["dissent", "critical_analysis", "ethical_review"],
                "special_powers": ["consensus_override", "ethical_memory_access"],
                "created_at": self._get_timestamp()
            }
            
            return tenth_man
        except Exception as e:
            print(f"Error creating 10th man agent: {e}")
            return None
    
    def create_medical_crew(self, crew_id: str, agent_ids: List[str], 
                          process_type: str = "sequential") -> Optional[Any]:
        """Create a crew of medical agents for collaborative reasoning"""
        if not CREWAI_AVAILABLE:
            return self._mock_crew(crew_id, agent_ids)
        
        try:
            # Get agent instances
            crew_agents = []
            for agent_id in agent_ids:
                if agent_id in self.agents:
                    crew_agents.append(self.agents[agent_id]["agent"])
                else:
                    print(f"Warning: Agent {agent_id} not found, skipping")
            
            if not crew_agents:
                print(f"Error: No valid agents found for crew {crew_id}")
                return None
            
            # Determine process type
            process = Process.sequential
            if process_type.lower() == "hierarchical":
                process = Process.hierarchical
            
            crew = Crew(
                agents=crew_agents,
                process=process,
                verbose=self.config.get("verbose", False),
                memory=self.config.get("enable_memory", True)
            )
            
            self.crews[crew_id] = {
                "crew": crew,
                "agent_ids": agent_ids,
                "process_type": process_type,
                "created_at": self._get_timestamp()
            }
            
            return crew
        except Exception as e:
            print(f"Error creating medical crew {crew_id}: {e}")
            return None
    
    def execute_medical_deliberation(self, crew_id: str, medical_query: str, 
                                   context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute multi-agent deliberation on medical query"""
        if not CREWAI_AVAILABLE:
            return self._mock_deliberation_result(crew_id, medical_query, context)
        
        try:
            if crew_id not in self.crews:
                return {"error": f"Crew {crew_id} not found"}
            
            crew = self.crews[crew_id]["crew"]
            
            # Create deliberation task
            task = Task(
                description=f"""
                Medical Query: {medical_query}
                
                Context: {context}
                
                Please collaborate to provide a comprehensive analysis considering:
                1. Medical accuracy and evidence-based reasoning
                2. Ethical implications and patient safety
                3. Alternative perspectives and potential risks
                4. Confidence assessment and uncertainty quantification
                
                If this is a high-stakes decision, the 10th man should provide contrarian analysis.
                """,
                expected_output="Structured medical analysis with consensus and dissent opinions"
            )
            
            # Execute deliberation
            result = crew.kickoff(tasks=[task])
            
            return {
                "crew_id": crew_id,
                "query": medical_query,
                "deliberation_result": str(result),
                "agents_involved": self.crews[crew_id]["agent_ids"],
                "process_type": self.crews[crew_id]["process_type"],
                "execution_timestamp": self._get_timestamp(),
                "success": True
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "crew_id": crew_id,
                "query": medical_query,
                "success": False,
                "execution_timestamp": self._get_timestamp()
            }
    
    def implement_tenth_man_rule(self, crew_id: str, consensus_proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Implement 10th man rule to challenge consensus"""
        if not CREWAI_AVAILABLE:
            return self._mock_tenth_man_response(consensus_proposal)
        
        try:
            if "tenth_man" not in self.agents:
                return {"error": "10th man agent not available"}
            
            tenth_man = self.agents["tenth_man"]["agent"]
            
            # Create dissent analysis task
            dissent_task = Task(
                description=f"""
                The group has reached the following consensus: {consensus_proposal}
                
                As the 10th man, your role is to:
                1. Challenge this consensus with contrarian perspectives
                2. Identify potential flaws or overlooked risks
                3. Propose alternative approaches or solutions
                4. Draw from ethical memory for historical examples of similar decisions
                5. Assess whether this consensus might be groupthink
                
                Provide a structured dissent analysis.
                """,
                expected_output="Detailed contrarian analysis with alternative proposals"
            )
            
            # Execute dissent analysis
            dissent_result = Crew(
                agents=[tenth_man],
                tasks=[dissent_task],
                process=Process.sequential
            ).kickoff()
            
            return {
                "dissent_analysis": str(dissent_result),
                "consensus_challenged": True,
                "tenth_man_active": True,
                "execution_timestamp": self._get_timestamp()
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "dissent_analysis": "Error in 10th man analysis",
                "consensus_challenged": False,
                "tenth_man_active": False
            }
    
    def _mock_agent(self, agent_id: str, role: str, domain_expertise: List[str]) -> Dict[str, Any]:
        """Mock agent for when CrewAI is not available"""
        return {
            "agent_id": agent_id,
            "role": role,
            "domain_expertise": domain_expertise,
            "type": "mock_agent",
            "created_at": self._get_timestamp()
        }
    
    def _mock_crew(self, crew_id: str, agent_ids: List[str]) -> Dict[str, Any]:
        """Mock crew for when CrewAI is not available"""
        return {
            "crew_id": crew_id,
            "agent_ids": agent_ids,
            "type": "mock_crew",
            "created_at": self._get_timestamp()
        }
    
    def _mock_deliberation_result(self, crew_id: str, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Mock deliberation result"""
        return {
            "crew_id": crew_id,
            "query": query,
            "deliberation_result": f"Mock multi-agent deliberation for: {query}",
            "agents_involved": ["mock_agent_1", "mock_agent_2", "tenth_man"],
            "consensus": "Mock consensus reached",
            "dissent": "Mock 10th man dissent provided",
            "confidence": 0.7,
            "mock_mode": True,
            "execution_timestamp": self._get_timestamp()
        }
    
    def _mock_tenth_man_response(self, consensus_proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Mock 10th man response"""
        return {
            "dissent_analysis": f"Mock contrarian analysis of: {consensus_proposal}",
            "alternative_proposals": ["Mock alternative 1", "Mock alternative 2"],
            "risk_assessment": "Mock risk identification",
            "consensus_challenged": True,
            "tenth_man_active": True,
            "mock_mode": True
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.utcnow().isoformat()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get integration system status"""
        return {
            "crewai_available": CREWAI_AVAILABLE,
            "agents_created": len(self.agents),
            "crews_created": len(self.crews),
            "active_tasks": len(self.active_tasks),
            "tenth_man_available": "tenth_man" in self.agents,
            "integration_status": "active" if CREWAI_AVAILABLE else "mock_mode"
        }


# Factory function for easy instantiation
def create_crewai_integration(config: Optional[Dict[str, Any]] = None) -> CrewAIIntegration:
    """Create CrewAI integration instance"""
    return CrewAIIntegration(config)


# Default instance for direct import
default_crewai = create_crewai_integration()
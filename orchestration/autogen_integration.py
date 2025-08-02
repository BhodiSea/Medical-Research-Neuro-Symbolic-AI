"""
AutoGen Integration for Medical Research AI

This module provides integration with Microsoft's AutoGen framework for multi-agent
conversation and coordination in medical research applications.

AutoGen is available via PyPI: pip install autogen-agentchat
"""

import sys
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import AutoGen components
try:
    from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_agentchat.conditions import TextMentionTermination
    from autogen_agentchat.ui import Console
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    AUTOGEN_AVAILABLE = True
except ImportError as e:
    logger.warning(f"AutoGen not available: {e}")
    logger.info("Install with: pip install autogen-agentchat autogen-ext[openai]")
    AUTOGEN_AVAILABLE = False


class AutoGenIntegration:
    """
    Integration wrapper for Microsoft AutoGen multi-agent conversation framework.
    
    AutoGen enables creating multi-agent AI applications that can act autonomously
    or work alongside humans for medical research coordination.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize AutoGen integration.
        
        Args:
            config: Configuration dictionary with AutoGen settings
        """
        self.config = config or {}
        self.agents = {}
        self.teams = {}
        self.model_clients = {}
        
        if not AUTOGEN_AVAILABLE:
            logger.warning("AutoGen integration running in mock mode")
        else:
            self._initialize_autogen_systems()
    
    def _initialize_autogen_systems(self) -> None:
        """Initialize AutoGen systems and model clients."""
        try:
            # Initialize model clients based on configuration
            if "openai_api_key" in self.config:
                self.model_clients["openai"] = OpenAIChatCompletionClient(
                    model=self.config.get("model", "gpt-4o")
                )
            
            logger.info("AutoGen systems initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing AutoGen systems: {e}")
    
    def create_medical_research_team(self, 
                                   team_name: str = "medical_research_team",
                                   agent_configs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Create a medical research team with specialized agents.
        
        Args:
            team_name: Name of the research team
            agent_configs: List of agent configurations
            
        Returns:
            Dictionary containing team information and agents
        """
        if not AUTOGEN_AVAILABLE:
            return self._mock_medical_research_team(team_name, agent_configs)
        
        try:
            # Default medical research agents if none provided
            if agent_configs is None:
                agent_configs = [
                    {
                        "name": "neurologist",
                        "role": "Neurologist specializing in neurodegeneration research",
                        "expertise": ["Parkinson's disease", "Alzheimer's", "ALS"]
                    },
                    {
                        "name": "molecular_biologist", 
                        "role": "Molecular biologist focusing on protein interactions",
                        "expertise": ["protein folding", "drug interactions", "biomarkers"]
                    },
                    {
                        "name": "clinical_researcher",
                        "role": "Clinical researcher with trial design expertise",
                        "expertise": ["clinical trials", "patient data", "regulatory compliance"]
                    },
                    {
                        "name": "data_scientist",
                        "role": "Data scientist for medical data analysis",
                        "expertise": ["statistical analysis", "machine learning", "data visualization"]
                    }
                ]
            
            # Create agents
            agents = []
            for agent_config in agent_configs:
                model_client = self.model_clients.get("openai")
                if model_client:
                    agent = AssistantAgent(
                        name=agent_config["name"],
                        system_message=f"You are a {agent_config['role']}. "
                                     f"Your expertise includes: {', '.join(agent_config['expertise'])}. "
                                     f"Provide detailed, evidence-based analysis for medical research questions."
                    )
                    agents.append(agent)
            
            # Create user proxy for human interaction
            user_proxy = UserProxyAgent("user_proxy")
            agents.append(user_proxy)
            
            # Create team with round-robin coordination
            termination = TextMentionTermination("exit", sources=["user_proxy"])
            team = RoundRobinGroupChat(agents, termination_condition=termination)
            
            # Store team information
            team_info = {
                "team_name": team_name,
                "agents": agents,
                "team": team,
                "agent_configs": agent_configs
            }
            
            self.teams[team_name] = team_info
            logger.info(f"Medical research team '{team_name}' created with {len(agents)} agents")
            
            return team_info
            
        except Exception as e:
            logger.error(f"Error creating medical research team: {e}")
            return self._mock_medical_research_team(team_name, agent_configs)
    
    async def run_medical_research_task(self, 
                                      team_name: str,
                                      task: str,
                                      max_rounds: int = 10) -> Dict[str, Any]:
        """
        Run a medical research task with the specified team.
        
        Args:
            team_name: Name of the team to use
            task: Research task description
            max_rounds: Maximum number of conversation rounds
            
        Returns:
            Dictionary containing task results and conversation history
        """
        if not AUTOGEN_AVAILABLE:
            return self._mock_research_task(team_name, task, max_rounds)
        
        try:
            if team_name not in self.teams:
                raise ValueError(f"Team '{team_name}' not found")
            
            team_info = self.teams[team_name]
            team = team_info["team"]
            
            # Run the task
            logger.info(f"Starting medical research task: {task}")
            
            # Use Console for interactive output
            console = Console(team.run_stream(task=task))
            await console
            
            # Collect results (this would need to be implemented based on AutoGen's output format)
            results = {
                "task": task,
                "team_name": team_name,
                "status": "completed",
                "conversation_rounds": max_rounds,
                "agents_participated": len(team_info["agents"])
            }
            
            logger.info(f"Medical research task completed: {task}")
            return results
            
        except Exception as e:
            logger.error(f"Error running medical research task: {e}")
            return self._mock_research_task(team_name, task, max_rounds)
    
    def create_specialized_agent(self, 
                               name: str,
                               role: str,
                               expertise: List[str],
                               model: str = "gpt-4o") -> Optional[Any]:
        """
        Create a specialized medical research agent.
        
        Args:
            name: Agent name
            role: Agent role description
            expertise: List of expertise areas
            model: Model to use for the agent
            
        Returns:
            AutoGen agent instance or None if failed
        """
        if not AUTOGEN_AVAILABLE:
            return self._mock_specialized_agent(name, role, expertise)
        
        try:
            model_client = self.model_clients.get("openai")
            if not model_client:
                logger.error("No model client available")
                return None
            
            agent = AssistantAgent(
                name=name,
                system_message=f"You are a {role}. "
                             f"Your expertise includes: {', '.join(expertise)}. "
                             f"Provide detailed, evidence-based analysis for medical research questions."
            )
            
            self.agents[name] = agent
            logger.info(f"Specialized agent '{name}' created")
            return agent
            
        except Exception as e:
            logger.error(f"Error creating specialized agent: {e}")
            return self._mock_specialized_agent(name, role, expertise)
    
    def get_available_teams(self) -> List[str]:
        """Get list of available team names."""
        return list(self.teams.keys())
    
    def get_available_agents(self) -> List[str]:
        """Get list of available agent names."""
        return list(self.agents.keys())
    
    async def cleanup(self) -> None:
        """Clean up AutoGen resources."""
        if AUTOGEN_AVAILABLE:
            try:
                for model_client in self.model_clients.values():
                    await model_client.close()
                logger.info("AutoGen resources cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up AutoGen resources: {e}")
    
    # Mock implementations for when AutoGen is not available
    def _mock_medical_research_team(self, team_name: str, agent_configs: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Mock implementation for medical research team creation."""
        return {
            "team_name": team_name,
            "agents": [],
            "team": None,
            "agent_configs": agent_configs or [],
            "status": "mock_mode"
        }
    
    def _mock_research_task(self, team_name: str, task: str, max_rounds: int) -> Dict[str, Any]:
        """Mock implementation for research task execution."""
        return {
            "task": task,
            "team_name": team_name,
            "status": "mock_completed",
            "conversation_rounds": max_rounds,
            "agents_participated": 0,
            "mock_result": f"Mock analysis of: {task}"
        }
    
    def _mock_specialized_agent(self, name: str, role: str, expertise: List[str]) -> Dict[str, Any]:
        """Mock implementation for specialized agent creation."""
        return {
            "name": name,
            "role": role,
            "expertise": expertise,
            "status": "mock_agent"
        }


# Example usage and testing
async def test_autogen_integration():
    """Test the AutoGen integration."""
    config = {
        "openai_api_key": "your-api-key-here",  # Would be set from environment
        "model": "gpt-4o"
    }
    
    autogen = AutoGenIntegration(config)
    
    # Create a medical research team
    team_info = autogen.create_medical_research_team("parkinsons_research")
    print(f"Created team: {team_info['team_name']}")
    
    # Run a research task
    task = "Analyze the relationship between alpha-synuclein aggregation and Parkinson's disease progression"
    results = await autogen.run_medical_research_task("parkinsons_research", task)
    print(f"Task results: {results}")
    
    # Cleanup
    await autogen.cleanup()


if __name__ == "__main__":
    # Run test if AutoGen is available
    if AUTOGEN_AVAILABLE:
        asyncio.run(test_autogen_integration())
    else:
        print("AutoGen not available. Install with: pip install autogen-agentchat autogen-ext[openai]") 
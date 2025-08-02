"""
SuperAGI Integration Wrapper
Provides standardized interface for autonomous agent management in medical research
"""

import sys
import os
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import asyncio
import json

# Add SuperAGI submodule to path
superagi_path = Path(__file__).parent / "superagi"
if str(superagi_path) not in sys.path:
    sys.path.insert(0, str(superagi_path))

try:
    # Import SuperAGI components when available
    from superagi.agent.agent_manager import AgentManager
    from superagi.agent.agent_config import AgentConfig
    from superagi.agent.agent import Agent
    from superagi.agent.agent_workflow_manager import AgentWorkflowManager
    from superagi.agent.agent_tool_step_handler import AgentToolStepHandler
    from superagi.agent.agent_executor import AgentExecutor
    from superagi.agent.agent_prompt_builder import AgentPromptBuilder
    from superagi.agent.agent_prompt_template import AgentPromptTemplate
    from superagi.agent.agent_prompt_template_factory import AgentPromptTemplateFactory
    from superagi.agent.agent_prompt_template_manager import AgentPromptTemplateManager
    from superagi.agent.agent_prompt_template_step_handler import AgentPromptTemplateStepHandler
    from superagi.agent.agent_prompt_template_step_handler_factory import AgentPromptTemplateStepHandlerFactory
    from superagi.agent.agent_prompt_template_step_handler_manager import AgentPromptTemplateStepHandlerManager
    from superagi.agent.agent_prompt_template_step_handler_manager_factory import AgentPromptTemplateStepHandlerManagerFactory
    from superagi.agent.agent_prompt_template_step_handler_manager_factory_factory import AgentPromptTemplateStepHandlerManagerFactoryFactory
    SUPERAGI_AVAILABLE = True
except ImportError as e:
    print(f"Warning: SuperAGI not available: {e}")
    SUPERAGI_AVAILABLE = False


class SuperAGIIntegration:
    """Integration wrapper for SuperAGI autonomous agent management"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.agent_manager = None
        self.workflow_manager = None
        self.medical_agents = {}
        self.research_workflows = {}
        
        if not SUPERAGI_AVAILABLE:
            print("Warning: SuperAGI integration running in mock mode")
        else:
            self._initialize_superagi()
    
    def _initialize_superagi(self) -> None:
        """Initialize SuperAGI components for medical research"""
        try:
            # Initialize agent manager
            self.agent_manager = AgentManager()
            
            # Initialize workflow manager
            self.workflow_manager = AgentWorkflowManager()
            
            # Initialize medical research agents
            self._initialize_medical_agents()
            
        except Exception as e:
            print(f"Error initializing SuperAGI: {e}")
    
    def _initialize_medical_agents(self) -> None:
        """Initialize specialized medical research agents"""
        medical_agent_configs = {
            "biomarker_researcher": {
                "name": "Biomarker Research Agent",
                "description": "Specialized agent for biomarker discovery and validation",
                "tools": ["literature_search", "data_analysis", "statistical_validation"],
                "goals": ["identify_novel_biomarkers", "validate_biomarker_candidates", "assess_clinical_relevance"]
            },
            "drug_discovery_agent": {
                "name": "Drug Discovery Agent", 
                "description": "Agent for drug repurposing and novel compound identification",
                "tools": ["molecular_docking", "drug_similarity", "toxicity_prediction"],
                "goals": ["identify_drug_candidates", "assess_safety_profiles", "optimize_compounds"]
            },
            "clinical_trial_agent": {
                "name": "Clinical Trial Agent",
                "description": "Agent for clinical trial design and optimization",
                "tools": ["trial_design", "statistical_power", "patient_recruitment"],
                "goals": ["optimize_trial_design", "assess_feasibility", "predict_outcomes"]
            },
            "literature_analyst": {
                "name": "Literature Analysis Agent",
                "description": "Agent for systematic literature review and meta-analysis",
                "tools": ["pubmed_search", "text_mining", "evidence_synthesis"],
                "goals": ["conduct_systematic_reviews", "identify_research_gaps", "synthesize_evidence"]
            }
        }
        
        for agent_id, config in medical_agent_configs.items():
            try:
                agent = self._create_medical_agent(agent_id, config)
                self.medical_agents[agent_id] = agent
            except Exception as e:
                print(f"Error creating medical agent {agent_id}: {e}")
    
    def _create_medical_agent(self, agent_id: str, config: Dict[str, Any]) -> Optional[Any]:
        """Create a specialized medical research agent"""
        if not SUPERAGI_AVAILABLE:
            return self._mock_agent(agent_id, config)
        
        try:
            # Create agent configuration
            agent_config = AgentConfig(
                name=config["name"],
                description=config["description"],
                tools=config["tools"],
                goals=config["goals"]
            )
            
            # Create agent instance
            agent = Agent(agent_config)
            
            return agent
            
        except Exception as e:
            print(f"Error creating agent {agent_id}: {e}")
            return None
    
    def create_research_workflow(self, workflow_name: str, 
                               agents: List[str], 
                               workflow_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a medical research workflow with multiple agents"""
        if not SUPERAGI_AVAILABLE:
            return self._mock_workflow(workflow_name, agents, workflow_steps)
        
        try:
            # Create workflow configuration
            workflow_config = {
                "name": workflow_name,
                "agents": agents,
                "steps": workflow_steps,
                "medical_domain": "neurodegeneration_research"
            }
            
            # Register workflow
            workflow_id = f"medical_workflow_{len(self.research_workflows)}"
            self.research_workflows[workflow_id] = workflow_config
            
            return {
                "workflow_id": workflow_id,
                "status": "created",
                "config": workflow_config
            }
            
        except Exception as e:
            print(f"Error creating research workflow: {e}")
            return self._mock_workflow(workflow_name, agents, workflow_steps)
    
    async def execute_research_workflow(self, workflow_id: str, 
                                      research_question: str,
                                      parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a medical research workflow"""
        if not SUPERAGI_AVAILABLE:
            return await self._mock_workflow_execution(workflow_id, research_question, parameters)
        
        try:
            workflow_config = self.research_workflows.get(workflow_id)
            if not workflow_config:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            # Execute workflow steps
            results = []
            for step in workflow_config["steps"]:
                step_result = await self._execute_workflow_step(step, research_question, parameters)
                results.append(step_result)
            
            # Synthesize results
            final_result = self._synthesize_workflow_results(results, research_question)
            
            return {
                "workflow_id": workflow_id,
                "research_question": research_question,
                "status": "completed",
                "results": results,
                "final_result": final_result,
                "confidence": self._calculate_workflow_confidence(results)
            }
            
        except Exception as e:
            print(f"Error executing research workflow: {e}")
            return await self._mock_workflow_execution(workflow_id, research_question, parameters)
    
    async def _execute_workflow_step(self, step: Dict[str, Any], 
                                   research_question: str,
                                   parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single workflow step"""
        try:
            step_type = step.get("type", "agent_task")
            agent_id = step.get("agent_id")
            
            if step_type == "agent_task" and agent_id:
                agent = self.medical_agents.get(agent_id)
                if agent:
                    # Execute agent task
                    task_result = await self._execute_agent_task(agent, step, research_question, parameters)
                    return {
                        "step_type": step_type,
                        "agent_id": agent_id,
                        "result": task_result,
                        "status": "completed"
                    }
            
            return {
                "step_type": step_type,
                "status": "skipped",
                "reason": "Unsupported step type or missing agent"
            }
            
        except Exception as e:
            print(f"Error executing workflow step: {e}")
            return {
                "step_type": step.get("type", "unknown"),
                "status": "failed",
                "error": str(e)
            }
    
    async def _execute_agent_task(self, agent: Any, step: Dict[str, Any], 
                                research_question: str,
                                parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task with a specific agent"""
        try:
            task_description = step.get("task", research_question)
            task_tools = step.get("tools", [])
            
            # Create task execution context
            task_context = {
                "research_question": research_question,
                "parameters": parameters,
                "tools": task_tools,
                "medical_domain": "neurodegeneration"
            }
            
            # Execute agent task
            result = await agent.execute_task(task_description, task_context)
            
            return {
                "task_description": task_description,
                "result": result,
                "tools_used": task_tools,
                "execution_time": result.get("execution_time", 0)
            }
            
        except Exception as e:
            print(f"Error executing agent task: {e}")
            return {
                "task_description": step.get("task", "unknown"),
                "error": str(e),
                "status": "failed"
            }
    
    def _synthesize_workflow_results(self, results: List[Dict[str, Any]], 
                                   research_question: str) -> Dict[str, Any]:
        """Synthesize results from multiple workflow steps"""
        try:
            # Extract successful results
            successful_results = [r for r in results if r.get("status") == "completed"]
            
            # Combine insights from all agents
            combined_insights = []
            for result in successful_results:
                if "result" in result and "insights" in result["result"]:
                    combined_insights.extend(result["result"]["insights"])
            
            # Generate synthesis
            synthesis = {
                "research_question": research_question,
                "total_steps": len(results),
                "successful_steps": len(successful_results),
                "combined_insights": combined_insights,
                "recommendations": self._generate_recommendations(combined_insights),
                "next_steps": self._suggest_next_steps(combined_insights)
            }
            
            return synthesis
            
        except Exception as e:
            print(f"Error synthesizing workflow results: {e}")
            return {
                "research_question": research_question,
                "error": str(e),
                "status": "synthesis_failed"
            }
    
    def _calculate_workflow_confidence(self, results: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for workflow execution"""
        try:
            if not results:
                return 0.0
            
            # Calculate confidence based on successful steps and result quality
            successful_steps = len([r for r in results if r.get("status") == "completed"])
            total_steps = len(results)
            
            success_rate = successful_steps / total_steps if total_steps > 0 else 0.0
            
            # Additional confidence factors
            confidence_factors = []
            for result in results:
                if result.get("status") == "completed":
                    confidence = result.get("result", {}).get("confidence", 0.5)
                    confidence_factors.append(confidence)
            
            avg_confidence = sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
            
            # Combined confidence score
            final_confidence = (success_rate * 0.6) + (avg_confidence * 0.4)
            return min(max(final_confidence, 0.0), 1.0)
            
        except Exception as e:
            print(f"Error calculating workflow confidence: {e}")
            return 0.5
    
    def _generate_recommendations(self, insights: List[str]) -> List[str]:
        """Generate recommendations based on workflow insights"""
        try:
            recommendations = []
            
            # Analyze insights for patterns
            if any("biomarker" in insight.lower() for insight in insights):
                recommendations.append("Consider biomarker validation studies")
            
            if any("drug" in insight.lower() for insight in insights):
                recommendations.append("Evaluate drug repurposing opportunities")
            
            if any("clinical" in insight.lower() for insight in insights):
                recommendations.append("Design clinical trial protocols")
            
            if any("literature" in insight.lower() for insight in insights):
                recommendations.append("Conduct systematic literature review")
            
            return recommendations[:5]  # Limit to top 5 recommendations
            
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return ["Continue research investigation"]
    
    def _suggest_next_steps(self, insights: List[str]) -> List[str]:
        """Suggest next research steps based on insights"""
        try:
            next_steps = []
            
            # Suggest next steps based on insights
            if insights:
                next_steps.append("Validate findings with additional datasets")
                next_steps.append("Conduct experimental validation")
                next_steps.append("Prepare research publication")
            
            return next_steps[:3]  # Limit to top 3 next steps
            
        except Exception as e:
            print(f"Error suggesting next steps: {e}")
            return ["Continue research investigation"]
    
    def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get status of a specific agent"""
        if not SUPERAGI_AVAILABLE:
            return self._mock_agent_status(agent_id)
        
        try:
            agent = self.medical_agents.get(agent_id)
            if agent:
                return {
                    "agent_id": agent_id,
                    "status": "active",
                    "name": agent.name if hasattr(agent, 'name') else agent_id,
                    "tools": agent.tools if hasattr(agent, 'tools') else [],
                    "goals": agent.goals if hasattr(agent, 'goals') else []
                }
            else:
                return {
                    "agent_id": agent_id,
                    "status": "not_found"
                }
                
        except Exception as e:
            print(f"Error getting agent status: {e}")
            return self._mock_agent_status(agent_id)
    
    def list_available_agents(self) -> List[Dict[str, Any]]:
        """List all available medical research agents"""
        if not SUPERAGI_AVAILABLE:
            return self._mock_agent_list()
        
        try:
            agents = []
            for agent_id, agent in self.medical_agents.items():
                agent_info = {
                    "agent_id": agent_id,
                    "name": agent.name if hasattr(agent, 'name') else agent_id,
                    "description": agent.description if hasattr(agent, 'description') else "",
                    "tools": agent.tools if hasattr(agent, 'tools') else [],
                    "goals": agent.goals if hasattr(agent, 'goals') else []
                }
                agents.append(agent_info)
            
            return agents
            
        except Exception as e:
            print(f"Error listing agents: {e}")
            return self._mock_agent_list()
    
    # Mock implementations for graceful degradation
    def _mock_agent(self, agent_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Mock agent when SuperAGI is not available"""
        return {
            "agent_id": agent_id,
            "name": config.get("name", agent_id),
            "description": config.get("description", ""),
            "tools": config.get("tools", []),
            "goals": config.get("goals", []),
            "status": "mock_agent"
        }
    
    def _mock_workflow(self, workflow_name: str, agents: List[str], 
                      workflow_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Mock workflow when SuperAGI is not available"""
        return {
            "workflow_id": f"mock_workflow_{workflow_name}",
            "name": workflow_name,
            "agents": agents,
            "steps": workflow_steps,
            "status": "mock_workflow"
        }
    
    async def _mock_workflow_execution(self, workflow_id: str, research_question: str,
                                     parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Mock workflow execution when SuperAGI is not available"""
        return {
            "workflow_id": workflow_id,
            "research_question": research_question,
            "status": "mock_completed",
            "results": [{"step_type": "mock", "status": "completed"}],
            "final_result": {
                "research_question": research_question,
                "combined_insights": ["Mock insight from workflow execution"],
                "recommendations": ["Mock recommendation"],
                "next_steps": ["Mock next step"]
            },
            "confidence": 0.5
        }
    
    def _mock_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Mock agent status when SuperAGI is not available"""
        return {
            "agent_id": agent_id,
            "status": "mock_active",
            "name": f"Mock {agent_id}",
            "tools": ["mock_tool"],
            "goals": ["mock_goal"]
        }
    
    def _mock_agent_list(self) -> List[Dict[str, Any]]:
        """Mock agent list when SuperAGI is not available"""
        return [
            {
                "agent_id": "biomarker_researcher",
                "name": "Mock Biomarker Research Agent",
                "description": "Mock agent for biomarker discovery",
                "tools": ["mock_literature_search", "mock_data_analysis"],
                "goals": ["mock_biomarker_identification"]
            },
            {
                "agent_id": "drug_discovery_agent",
                "name": "Mock Drug Discovery Agent",
                "description": "Mock agent for drug discovery",
                "tools": ["mock_molecular_docking", "mock_toxicity_prediction"],
                "goals": ["mock_drug_identification"]
            }
        ]

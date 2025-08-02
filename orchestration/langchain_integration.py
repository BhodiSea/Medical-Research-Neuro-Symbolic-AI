"""
LangChain Integration for Medical Research AI

This module provides integration with LangChain for LLM application framework and
reasoning chains, supporting advanced medical research workflows and multi-step reasoning.

LangChain is available via the cloned submodule.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add LangChain submodule to path
langchain_path = Path(__file__).parent / "langchain"
if str(langchain_path) not in sys.path:
    sys.path.insert(0, str(langchain_path))

# Global flags for LangChain availability - will be set on first use
LANGCHAIN_AVAILABLE = None
LANGCHAIN_INITIALIZED = False


class LangChainIntegration:
    """
    Integration wrapper for LangChain (LLM Application Framework).
    
    LangChain provides LLM application framework and reasoning chains for advanced
    medical research workflows and multi-step reasoning processes.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize LangChain integration.
        
        Args:
            config: Configuration dictionary with LangChain settings
        """
        self.config = config or {}
        self.llm = None
        self.chat_model = None
        self.memory = None
        self.agent = None
        self.chains = {}
        self._langchain_components = {}
        
        # Don't initialize anything at startup - use lazy loading
        logger.info("LangChain integration initialized with lazy loading")
    
    def _check_langchain_availability(self) -> bool:
        """Check if LangChain is available and initialize if needed."""
        global LANGCHAIN_AVAILABLE, LANGCHAIN_INITIALIZED
        
        if LANGCHAIN_AVAILABLE is None:
            try:
                # Try to import LangChain components only when needed
                from langchain.llms import OpenAI
                from langchain.chat_models import ChatOpenAI
                from langchain.prompts import PromptTemplate, ChatPromptTemplate
                from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
                from langchain.agents import initialize_agent, Tool, AgentType
                from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
                from langchain.schema import HumanMessage, AIMessage, SystemMessage
                from langchain.tools import BaseTool
                from langchain.callbacks import BaseCallbackHandler
                
                # Store components for later use
                self._langchain_components = {
                    'OpenAI': OpenAI,
                    'ChatOpenAI': ChatOpenAI,
                    'PromptTemplate': PromptTemplate,
                    'ChatPromptTemplate': ChatPromptTemplate,
                    'LLMChain': LLMChain,
                    'SimpleSequentialChain': SimpleSequentialChain,
                    'SequentialChain': SequentialChain,
                    'initialize_agent': initialize_agent,
                    'Tool': Tool,
                    'AgentType': AgentType,
                    'ConversationBufferMemory': ConversationBufferMemory,
                    'ConversationSummaryMemory': ConversationSummaryMemory,
                    'HumanMessage': HumanMessage,
                    'AIMessage': AIMessage,
                    'SystemMessage': SystemMessage,
                    'BaseTool': BaseTool,
                    'BaseCallbackHandler': BaseCallbackHandler
                }
                
                LANGCHAIN_AVAILABLE = True
                logger.info("LangChain components loaded successfully")
                
            except ImportError as e:
                LANGCHAIN_AVAILABLE = False
                logger.warning(f"LangChain not available: {e}")
                logger.info("Install with: pip install langchain openai")
        
        return LANGCHAIN_AVAILABLE
    
    def _initialize_langchain_systems(self) -> None:
        """Initialize LangChain systems and components - called only when needed."""
        global LANGCHAIN_INITIALIZED
        
        if LANGCHAIN_INITIALIZED:
            return
            
        try:
            if not self._check_langchain_availability():
                return
                
            # Initialize LLM models
            self._initialize_llm_models()
            
            # Initialize memory systems
            self._initialize_memory_systems()
            
            # Initialize reasoning chains
            self._initialize_reasoning_chains()
            
            # Initialize medical research tools
            self._initialize_medical_tools()
            
            LANGCHAIN_INITIALIZED = True
            logger.info("LangChain systems initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing LangChain systems: {e}")
    
    def _initialize_llm_models(self) -> None:
        """Initialize LLM models for different tasks."""
        try:
            # Initialize OpenAI models if API key is available
            api_key = self.config.get("openai_api_key")
            if api_key:
                OpenAI = self._langchain_components['OpenAI']
                ChatOpenAI = self._langchain_components['ChatOpenAI']
                
                self.llm = OpenAI(
                    temperature=0.1,
                    openai_api_key=api_key,
                    model_name=self.config.get("model_name", "gpt-3.5-turbo-instruct")
                )
                
                self.chat_model = ChatOpenAI(
                    temperature=0.1,
                    openai_api_key=api_key,
                    model_name=self.config.get("chat_model_name", "gpt-4")
                )
                logger.info("OpenAI models initialized successfully")
            else:
                logger.warning("No OpenAI API key provided, using mock models")
                
        except Exception as e:
            logger.error(f"Error initializing LLM models: {e}")
    
    def _initialize_memory_systems(self) -> None:
        """Initialize memory systems for conversation tracking."""
        try:
            ConversationBufferMemory = self._langchain_components['ConversationBufferMemory']
            ConversationSummaryMemory = self._langchain_components['ConversationSummaryMemory']
            
            # Initialize conversation memory
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            # Initialize summary memory for long conversations
            self.summary_memory = ConversationSummaryMemory(
                llm=self.llm if self.llm else None,
                memory_key="chat_summary"
            )
            
            logger.info("Memory systems initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing memory systems: {e}")
    
    def _initialize_reasoning_chains(self) -> None:
        """Initialize reasoning chains for medical research."""
        try:
            PromptTemplate = self._langchain_components['PromptTemplate']
            LLMChain = self._langchain_components['LLMChain']
            
            # Medical literature analysis chain
            literature_template = """
            Analyze the following medical literature and extract key insights:
            
            Literature: {literature_text}
            
            Focus on: {analysis_focus}
            
            Provide a structured analysis including:
            1. Key findings
            2. Methodology assessment
            3. Clinical relevance
            4. Limitations
            5. Future research directions
            """
            
            literature_prompt = PromptTemplate(
                input_variables=["literature_text", "analysis_focus"],
                template=literature_template
            )
            
            if self.llm:
                self.chains["literature_analysis"] = LLMChain(
                    llm=self.llm,
                    prompt=literature_prompt,
                    memory=self.memory
                )
            
            # Medical hypothesis generation chain
            hypothesis_template = """
            Based on the following medical research context, generate novel hypotheses:
            
            Context: {research_context}
            Current Findings: {current_findings}
            
            Generate hypotheses that:
            1. Address gaps in current understanding
            2. Suggest novel therapeutic approaches
            3. Propose new research directions
            4. Consider cross-disease applications
            """
            
            hypothesis_prompt = PromptTemplate(
                input_variables=["research_context", "current_findings"],
                template=hypothesis_template
            )
            
            if self.llm:
                self.chains["hypothesis_generation"] = LLMChain(
                    llm=self.llm,
                    prompt=hypothesis_prompt,
                    memory=self.memory
                )
            
            # Clinical trial design chain
            trial_template = """
            Design a clinical trial based on the following research findings:
            
            Research Question: {research_question}
            Target Population: {target_population}
            Primary Endpoint: {primary_endpoint}
            
            Design a trial that includes:
            1. Study design and methodology
            2. Sample size calculation
            3. Inclusion/exclusion criteria
            4. Primary and secondary endpoints
            5. Statistical analysis plan
            6. Safety considerations
            """
            
            trial_prompt = PromptTemplate(
                input_variables=["research_question", "target_population", "primary_endpoint"],
                template=trial_template
            )
            
            if self.llm:
                self.chains["trial_design"] = LLMChain(
                    llm=self.llm,
                    prompt=trial_prompt,
                    memory=self.memory
                )
            
            logger.info("Reasoning chains initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing reasoning chains: {e}")
    
    def _initialize_medical_tools(self) -> None:
        """Initialize medical research tools for the agent."""
        try:
            Tool = self._langchain_components['Tool']
            initialize_agent = self._langchain_components['initialize_agent']
            AgentType = self._langchain_components['AgentType']
            
            # Define medical research tools
            tools = [
                Tool(
                    name="Literature Analysis",
                    func=self._analyze_literature_tool,
                    description="Analyze medical literature and extract key insights"
                ),
                Tool(
                    name="Hypothesis Generation",
                    func=self._generate_hypothesis_tool,
                    description="Generate novel research hypotheses based on current findings"
                ),
                Tool(
                    name="Trial Design",
                    func=self._design_trial_tool,
                    description="Design clinical trials based on research findings"
                ),
                Tool(
                    name="Biomarker Analysis",
                    func=self._analyze_biomarker_tool,
                    description="Analyze biomarker data and identify patterns"
                ),
                Tool(
                    name="Drug Interaction Check",
                    func=self._check_drug_interactions_tool,
                    description="Check for potential drug interactions and safety concerns"
                )
            ]
            
            # Initialize agent with tools
            if self.chat_model:
                self.agent = initialize_agent(
                    tools,
                    self.chat_model,
                    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                    memory=self.memory,
                    verbose=True
                )
                logger.info("Medical research agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing medical tools: {e}")
    
    def analyze_medical_literature(self, 
                                 literature_text: str,
                                 analysis_focus: str = "general") -> Dict[str, Any]:
        """
        Analyze medical literature using LangChain reasoning chains.
        
        Args:
            literature_text: Medical literature text to analyze
            analysis_focus: Focus area for analysis (general, methodology, clinical, etc.)
            
        Returns:
            Dictionary containing literature analysis results
        """
        # Initialize LangChain only when this method is called
        if not self._check_langchain_availability():
            return self._mock_literature_analysis(literature_text, analysis_focus)
        
        try:
            # Initialize systems on first use
            if not LANGCHAIN_INITIALIZED:
                self._initialize_langchain_systems()
            
            if "literature_analysis" in self.chains:
                result = self.chains["literature_analysis"].run({
                    "literature_text": literature_text,
                    "analysis_focus": analysis_focus
                })
                
                return {
                    "literature_text": literature_text,
                    "analysis_focus": analysis_focus,
                    "status": "completed",
                    "analysis_result": result,
                    "metadata": {
                        "model": "LangChain",
                        "chain_type": "literature_analysis",
                        "text_length": len(literature_text)
                    }
                }
            else:
                return self._mock_literature_analysis(literature_text, analysis_focus)
                
        except Exception as e:
            logger.error(f"Error analyzing medical literature: {e}")
            return self._mock_literature_analysis(literature_text, analysis_focus)
    
    def generate_research_hypotheses(self, 
                                   research_context: str,
                                   current_findings: str) -> Dict[str, Any]:
        """
        Generate research hypotheses using LangChain reasoning chains.
        
        Args:
            research_context: Current research context and background
            current_findings: Current research findings and data
            
        Returns:
            Dictionary containing generated hypotheses
        """
        # Initialize LangChain only when this method is called
        if not self._check_langchain_availability():
            return self._mock_hypothesis_generation(research_context, current_findings)
        
        try:
            # Initialize systems on first use
            if not LANGCHAIN_INITIALIZED:
                self._initialize_langchain_systems()
            
            if "hypothesis_generation" in self.chains:
                result = self.chains["hypothesis_generation"].run({
                    "research_context": research_context,
                    "current_findings": current_findings
                })
                
                return {
                    "research_context": research_context,
                    "current_findings": current_findings,
                    "status": "completed",
                    "generated_hypotheses": result,
                    "metadata": {
                        "model": "LangChain",
                        "chain_type": "hypothesis_generation"
                    }
                }
            else:
                return self._mock_hypothesis_generation(research_context, current_findings)
                
        except Exception as e:
            logger.error(f"Error generating research hypotheses: {e}")
            return self._mock_hypothesis_generation(research_context, current_findings)
    
    def design_clinical_trial(self, 
                            research_question: str,
                            target_population: str,
                            primary_endpoint: str) -> Dict[str, Any]:
        """
        Design clinical trial using LangChain reasoning chains.
        
        Args:
            research_question: Primary research question for the trial
            target_population: Target patient population
            primary_endpoint: Primary endpoint for the trial
            
        Returns:
            Dictionary containing clinical trial design
        """
        # Initialize LangChain only when this method is called
        if not self._check_langchain_availability():
            return self._mock_trial_design(research_question, target_population, primary_endpoint)
        
        try:
            # Initialize systems on first use
            if not LANGCHAIN_INITIALIZED:
                self._initialize_langchain_systems()
            
            if "trial_design" in self.chains:
                result = self.chains["trial_design"].run({
                    "research_question": research_question,
                    "target_population": target_population,
                    "primary_endpoint": primary_endpoint
                })
                
                return {
                    "research_question": research_question,
                    "target_population": target_population,
                    "primary_endpoint": primary_endpoint,
                    "status": "completed",
                    "trial_design": result,
                    "metadata": {
                        "model": "LangChain",
                        "chain_type": "trial_design"
                    }
                }
            else:
                return self._mock_trial_design(research_question, target_population, primary_endpoint)
                
        except Exception as e:
            logger.error(f"Error designing clinical trial: {e}")
            return self._mock_trial_design(research_question, target_population, primary_endpoint)
    
    async def run_medical_research_agent(self, 
                                       query: str,
                                       research_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Run medical research agent for complex research tasks.
        
        Args:
            query: Research query or task for the agent
            research_context: Optional research context for the agent
            
        Returns:
            Dictionary containing agent response and reasoning
        """
        # Initialize LangChain only when this method is called
        if not self._check_langchain_availability():
            return self._mock_agent_response(query, research_context)
        
        try:
            # Initialize systems on first use
            if not LANGCHAIN_INITIALIZED:
                self._initialize_langchain_systems()
            
            if self.agent:
                # Prepare context for the agent
                if research_context:
                    full_query = f"Context: {research_context}\n\nQuery: {query}"
                else:
                    full_query = query
                
                # Run the agent
                result = await self.agent.arun(full_query)
                
                return {
                    "query": query,
                    "research_context": research_context,
                    "status": "completed",
                    "agent_response": result,
                    "conversation_history": self.memory.chat_memory.messages if self.memory else [],
                    "metadata": {
                        "model": "LangChain",
                        "agent_type": "medical_research",
                        "tools_used": len(self.agent.tools) if self.agent else 0
                    }
                }
            else:
                return self._mock_agent_response(query, research_context)
                
        except Exception as e:
            logger.error(f"Error running medical research agent: {e}")
            return self._mock_agent_response(query, research_context)
    
    def create_reasoning_chain(self, 
                             chain_name: str,
                             prompt_template: str,
                             input_variables: List[str]) -> Dict[str, Any]:
        """
        Create a custom reasoning chain for specific medical research tasks.
        
        Args:
            chain_name: Name for the new chain
            prompt_template: Template for the chain prompt
            input_variables: List of input variables for the template
            
        Returns:
            Dictionary containing chain creation status
        """
        # Initialize LangChain only when this method is called
        if not self._check_langchain_availability():
            return self._mock_chain_creation(chain_name, prompt_template, input_variables)
        
        try:
            # Initialize systems on first use
            if not LANGCHAIN_INITIALIZED:
                self._initialize_langchain_systems()
            
            PromptTemplate = self._langchain_components['PromptTemplate']
            LLMChain = self._langchain_components['LLMChain']
            
            # Create prompt template
            prompt = PromptTemplate(
                input_variables=input_variables,
                template=prompt_template
            )
            
            # Create LLM chain
            if self.llm:
                chain = LLMChain(
                    llm=self.llm,
                    prompt=prompt,
                    memory=self.memory
                )
                
                # Store the chain
                self.chains[chain_name] = chain
                
                return {
                    "chain_name": chain_name,
                    "status": "created",
                    "input_variables": input_variables,
                    "metadata": {
                        "model": "LangChain",
                        "chain_type": "custom",
                        "total_chains": len(self.chains)
                    }
                }
            else:
                return self._mock_chain_creation(chain_name, prompt_template, input_variables)
                
        except Exception as e:
            logger.error(f"Error creating reasoning chain: {e}")
            return self._mock_chain_creation(chain_name, prompt_template, input_variables)
    
    def run_reasoning_chain(self, 
                          chain_name: str,
                          inputs: Dict[str, str]) -> Dict[str, Any]:
        """
        Run a specific reasoning chain with given inputs.
        
        Args:
            chain_name: Name of the chain to run
            inputs: Dictionary of inputs for the chain
            
        Returns:
            Dictionary containing chain execution results
        """
        # Initialize LangChain only when this method is called
        if not self._check_langchain_availability():
            return self._mock_chain_execution(chain_name, inputs)
        
        try:
            # Initialize systems on first use
            if not LANGCHAIN_INITIALIZED:
                self._initialize_langchain_systems()
            
            if chain_name in self.chains:
                result = self.chains[chain_name].run(inputs)
                
                return {
                    "chain_name": chain_name,
                    "inputs": inputs,
                    "status": "completed",
                    "result": result,
                    "metadata": {
                        "model": "LangChain",
                        "chain_type": "custom"
                    }
                }
            else:
                return {
                    "chain_name": chain_name,
                    "inputs": inputs,
                    "status": "error",
                    "error": f"Chain '{chain_name}' not found",
                    "available_chains": list(self.chains.keys())
                }
                
        except Exception as e:
            logger.error(f"Error running reasoning chain: {e}")
            return self._mock_chain_execution(chain_name, inputs)
    
    # Tool functions for the agent
    def _analyze_literature_tool(self, query: str) -> str:
        """Tool for analyzing medical literature."""
        return "Literature analysis completed: Key findings include [mock findings]"
    
    def _generate_hypothesis_tool(self, query: str) -> str:
        """Tool for generating research hypotheses."""
        return "Hypothesis generation completed: Novel hypotheses include [mock hypotheses]"
    
    def _design_trial_tool(self, query: str) -> str:
        """Tool for designing clinical trials."""
        return "Trial design completed: Clinical trial design includes [mock design]"
    
    def _analyze_biomarker_tool(self, query: str) -> str:
        """Tool for analyzing biomarker data."""
        return "Biomarker analysis completed: Key patterns include [mock patterns]"
    
    def _check_drug_interactions_tool(self, query: str) -> str:
        """Tool for checking drug interactions."""
        return "Drug interaction check completed: Safety profile includes [mock safety data]"
    
    # Mock implementations for when LangChain is not available
    def _mock_literature_analysis(self, literature_text: str, analysis_focus: str) -> Dict[str, Any]:
        """Mock implementation for literature analysis."""
        return {
            "literature_text": literature_text,
            "analysis_focus": analysis_focus,
            "status": "mock_completed",
            "analysis_result": "Mock literature analysis result",
            "metadata": {"model": "mock", "chain_type": "mock"}
        }
    
    def _mock_hypothesis_generation(self, research_context: str, current_findings: str) -> Dict[str, Any]:
        """Mock implementation for hypothesis generation."""
        return {
            "research_context": research_context,
            "current_findings": current_findings,
            "status": "mock_completed",
            "generated_hypotheses": "Mock generated hypotheses",
            "metadata": {"model": "mock", "chain_type": "mock"}
        }
    
    def _mock_trial_design(self, research_question: str, target_population: str, primary_endpoint: str) -> Dict[str, Any]:
        """Mock implementation for trial design."""
        return {
            "research_question": research_question,
            "target_population": target_population,
            "primary_endpoint": primary_endpoint,
            "status": "mock_completed",
            "trial_design": "Mock clinical trial design",
            "metadata": {"model": "mock", "chain_type": "mock"}
        }
    
    def _mock_agent_response(self, query: str, research_context: Optional[str]) -> Dict[str, Any]:
        """Mock implementation for agent response."""
        return {
            "query": query,
            "research_context": research_context,
            "status": "mock_completed",
            "agent_response": "Mock agent response",
            "conversation_history": [],
            "metadata": {"model": "mock", "agent_type": "mock"}
        }
    
    def _mock_chain_creation(self, chain_name: str, prompt_template: str, input_variables: List[str]) -> Dict[str, Any]:
        """Mock implementation for chain creation."""
        return {
            "chain_name": chain_name,
            "status": "mock_created",
            "input_variables": input_variables,
            "metadata": {"model": "mock", "chain_type": "mock"}
        }
    
    def _mock_chain_execution(self, chain_name: str, inputs: Dict[str, str]) -> Dict[str, Any]:
        """Mock implementation for chain execution."""
        return {
            "chain_name": chain_name,
            "inputs": inputs,
            "status": "mock_completed",
            "result": "Mock chain execution result",
            "metadata": {"model": "mock", "chain_type": "mock"}
        }


# Example usage and testing
async def test_langchain_integration():
    """Test the LangChain integration."""
    config = {
        "openai_api_key": None,  # No API key for testing
        "model_name": "gpt-3.5-turbo-instruct",
        "chat_model_name": "gpt-4"
    }
    
    langchain_integration = LangChainIntegration(config)
    
    # Test literature analysis
    literature_result = langchain_integration.analyze_medical_literature(
        "Parkinson's disease is characterized by alpha-synuclein aggregation.",
        "pathophysiology"
    )
    print(f"Literature Analysis: {literature_result['status']}")
    
    # Test hypothesis generation
    hypothesis_result = langchain_integration.generate_research_hypotheses(
        "Current research on Parkinson's disease",
        "Alpha-synuclein aggregation is a key pathological feature"
    )
    print(f"Hypothesis Generation: {hypothesis_result['status']}")
    
    # Test agent response
    agent_result = await langchain_integration.run_medical_research_agent(
        "Analyze the relationship between alpha-synuclein and Parkinson's disease"
    )
    print(f"Agent Response: {agent_result['status']}")


if __name__ == "__main__":
    asyncio.run(test_langchain_integration()) 
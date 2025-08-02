"""
Mem0 Integration Wrapper
Provides standardized interface for universal memory layer and long-term ethical storage
"""

import sys
import os
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

# Add Mem0 submodule to path
mem0_path = Path(__file__).parent / "mem0"
if str(mem0_path) not in sys.path:
    sys.path.insert(0, str(mem0_path))

try:
    # Import Mem0 components when available
    from mem0 import Memory
    MEM0_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Mem0 not available: {e}")
    MEM0_AVAILABLE = False


class Mem0Integration:
    """Integration wrapper for Mem0 universal memory layer"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.memory_instances = {}
        self.ethical_memories = {}
        self.agent_memories = {}
        
        if not MEM0_AVAILABLE:
            print("Warning: Mem0 integration running in mock mode")
        else:
            self._initialize_memory_systems()
    
    def _initialize_memory_systems(self) -> None:
        """Initialize different memory systems for different purposes"""
        try:
            # Main ethical memory system
            self.memory_instances["ethical"] = Memory(
                config={
                    "memory_type": "long_term",
                    "storage_backend": self.config.get("storage_backend", "local"),
                    "embedding_model": self.config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
                    "namespace": "ethical_memory"
                }
            )
            
            # Agent-specific memories
            self.memory_instances["agents"] = Memory(
                config={
                    "memory_type": "episodic",
                    "storage_backend": self.config.get("storage_backend", "local"),
                    "namespace": "agent_memory"
                }
            )
            
            # Simulation outcome memories
            self.memory_instances["simulations"] = Memory(
                config={
                    "memory_type": "semantic",
                    "storage_backend": self.config.get("storage_backend", "local"),
                    "namespace": "simulation_memory"
                }
            )
            
        except Exception as e:
            print(f"Error initializing memory systems: {e}")
    
    def store_ethical_insight(self, insight: Dict[str, Any], agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Store ethical insight in long-term memory"""
        if not MEM0_AVAILABLE:
            return self._mock_memory_storage(insight, "ethical")
        
        try:
            memory = self.memory_instances.get("ethical")
            if not memory:
                return {"error": "Ethical memory system not initialized"}
            
            # Format insight for storage
            memory_content = {
                "type": "ethical_insight",
                "content": insight.get("description", ""),
                "principles": insight.get("ethical_principles", []),
                "context": insight.get("context", {}),
                "confidence": insight.get("confidence", 0.0),
                "source": insight.get("source", "unknown"),
                "agent_id": agent_id,
                "timestamp": self._get_timestamp()
            }
            
            # Store in memory
            result = memory.add(
                messages=[memory_content],
                user_id=agent_id or "system"
            )
            
            return {
                "stored": True,
                "memory_id": result.get("memory_id"),
                "insight_type": "ethical",
                "storage_timestamp": self._get_timestamp()
            }
            
        except Exception as e:
            return {
                "stored": False,
                "error": str(e),
                "insight_type": "ethical"
            }
    
    def store_simulation_outcome(self, simulation_id: str, outcomes: Dict[str, Any]) -> Dict[str, Any]:
        """Store simulation outcomes for future ethical learning"""
        if not MEM0_AVAILABLE:
            return self._mock_memory_storage(outcomes, "simulation")
        
        try:
            memory = self.memory_instances.get("simulations")
            if not memory:
                return {"error": "Simulation memory system not initialized"}
            
            # Format simulation outcome for storage
            memory_content = {
                "type": "simulation_outcome",
                "simulation_id": simulation_id,
                "outcomes": outcomes.get("outcomes", []),
                "ethical_lessons": outcomes.get("ethical_lessons", []),
                "decision_points": outcomes.get("decision_points", []),
                "alternative_paths": outcomes.get("alternative_paths", []),
                "moral_weight": outcomes.get("moral_weight", 0.0),
                "generalizability": outcomes.get("generalizability", 0.0),
                "timestamp": self._get_timestamp()
            }
            
            # Store in memory
            result = memory.add(
                messages=[memory_content],
                user_id=f"simulation_{simulation_id}"
            )
            
            return {
                "stored": True,
                "memory_id": result.get("memory_id"),
                "simulation_id": simulation_id,
                "outcome_type": "simulation",
                "storage_timestamp": self._get_timestamp()
            }
            
        except Exception as e:
            return {
                "stored": False,
                "error": str(e),
                "simulation_id": simulation_id
            }
    
    def retrieve_ethical_memories(self, query: str, context: Dict[str, Any], 
                                limit: int = 10) -> Dict[str, Any]:
        """Retrieve relevant ethical memories for current context"""
        if not MEM0_AVAILABLE:
            return self._mock_memory_retrieval(query, "ethical")
        
        try:
            memory = self.memory_instances.get("ethical")
            if not memory:
                return {"error": "Ethical memory system not initialized"}
            
            # Search for relevant memories
            memories = memory.search(
                query=query,
                user_id="system",
                limit=limit
            )
            
            # Filter and rank memories by relevance to context
            relevant_memories = []
            for mem in memories:
                relevance_score = self._calculate_memory_relevance(mem, context)
                if relevance_score > 0.3:  # Relevance threshold
                    relevant_memories.append({
                        "memory": mem,
                        "relevance_score": relevance_score,
                        "memory_id": mem.get("id"),
                        "content": mem.get("content")
                    })
            
            # Sort by relevance
            relevant_memories.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            return {
                "retrieved": True,
                "memories": relevant_memories,
                "query": query,
                "total_found": len(relevant_memories),
                "retrieval_timestamp": self._get_timestamp()
            }
            
        except Exception as e:
            return {
                "retrieved": False,
                "error": str(e),
                "query": query,
                "memories": []
            }
    
    def store_agent_experience(self, agent_id: str, experience: Dict[str, Any]) -> Dict[str, Any]:
        """Store agent experience for ethical learning"""
        if not MEM0_AVAILABLE:
            return self._mock_memory_storage(experience, "agent")
        
        try:
            memory = self.memory_instances.get("agents")
            if not memory:
                return {"error": "Agent memory system not initialized"}
            
            # Format experience for storage
            memory_content = {
                "type": "agent_experience",
                "agent_id": agent_id,
                "experience_type": experience.get("type", "general"),
                "content": experience.get("description", ""),
                "ethical_implications": experience.get("ethical_implications", []),
                "learning_outcomes": experience.get("learning_outcomes", []),
                "confidence_change": experience.get("confidence_change", 0.0),
                "moral_weight": experience.get("moral_weight", 0.0),
                "timestamp": self._get_timestamp()
            }
            
            # Store in memory
            result = memory.add(
                messages=[memory_content],
                user_id=agent_id
            )
            
            return {
                "stored": True,
                "memory_id": result.get("memory_id"),
                "agent_id": agent_id,
                "experience_type": experience.get("type", "general"),
                "storage_timestamp": self._get_timestamp()
            }
            
        except Exception as e:
            return {
                "stored": False,
                "error": str(e),
                "agent_id": agent_id
            }
    
    def create_medical_memory(self, memory_id: str, content: str, memory_type: str) -> Dict[str, Any]:
        """Create a medical memory entry"""
        if not MEM0_AVAILABLE:
            return self._mock_memory_storage({"content": content, "type": memory_type}, "medical")
        
        try:
            memory = self.memory_instances.get("ethical")
            if not memory:
                return {"error": "Memory system not initialized"}
            
            # Format medical memory for storage
            memory_content = {
                "type": "medical_memory",
                "memory_id": memory_id,
                "content": content,
                "memory_type": memory_type,
                "timestamp": self._get_timestamp()
            }
            
            # Store in memory
            result = memory.add(
                messages=[memory_content],
                user_id="medical_system"
            )
            
            return {
                "stored": True,
                "memory_id": result.get("memory_id"),
                "content": content,
                "memory_type": memory_type,
                "storage_timestamp": self._get_timestamp()
            }
            
        except Exception as e:
            return {
                "stored": False,
                "error": str(e),
                "memory_id": memory_id
            }
    
    def retrieve_medical_memory(self, memory_id: str) -> Dict[str, Any]:
        """Retrieve a medical memory entry"""
        if not MEM0_AVAILABLE:
            return self._mock_memory_retrieval(f"medical_memory_{memory_id}", "medical")
        
        try:
            memory = self.memory_instances.get("ethical")
            if not memory:
                return {"error": "Memory system not initialized"}
            
            # Query memory for specific medical memory
            query = f"memory_id:{memory_id} AND type:medical_memory"
            result = memory.query(
                query=query,
                user_id="medical_system"
            )
            
            if result and result.get("messages"):
                memory_data = result["messages"][0]
                return {
                    "found": True,
                    "memory_id": memory_id,
                    "content": memory_data.get("content", ""),
                    "memory_type": memory_data.get("memory_type", ""),
                    "timestamp": memory_data.get("timestamp", ""),
                    "retrieval_timestamp": self._get_timestamp()
                }
            else:
                return {
                    "found": False,
                    "memory_id": memory_id,
                    "error": "Memory not found"
                }
                
        except Exception as e:
            return {
                "found": False,
                "error": str(e),
                "memory_id": memory_id
            }
    
    def store_ethical_memory(self, memory_id: str, content: str, ethical_framework: str) -> Dict[str, Any]:
        """Store ethical memory entry"""
        if not MEM0_AVAILABLE:
            return self._mock_memory_storage({"content": content, "framework": ethical_framework}, "ethical")
        
        try:
            memory = self.memory_instances.get("ethical")
            if not memory:
                return {"error": "Ethical memory system not initialized"}
            
            # Format ethical memory for storage
            memory_content = {
                "type": "ethical_memory",
                "memory_id": memory_id,
                "content": content,
                "ethical_framework": ethical_framework,
                "timestamp": self._get_timestamp()
            }
            
            # Store in memory
            result = memory.add(
                messages=[memory_content],
                user_id="ethical_system"
            )
            
            return {
                "stored": True,
                "memory_id": result.get("memory_id"),
                "content": content,
                "ethical_framework": ethical_framework,
                "storage_timestamp": self._get_timestamp()
            }
            
        except Exception as e:
            return {
                "stored": False,
                "error": str(e),
                "memory_id": memory_id
            }
    
    def get_agent_memory_context(self, agent_id: str, current_situation: str) -> Dict[str, Any]:
        """Get relevant memory context for agent's current situation"""
        if not MEM0_AVAILABLE:
            return self._mock_memory_retrieval(current_situation, "agent")
        
        try:
            memory = self.memory_instances.get("agents")
            if not memory:
                return {"error": "Agent memory system not initialized"}
            
            # Retrieve agent's relevant memories
            memories = memory.search(
                query=current_situation,
                user_id=agent_id,
                limit=5
            )
            
            # Format for agent consumption
            memory_context = {
                "agent_id": agent_id,
                "current_situation": current_situation,
                "relevant_experiences": [],
                "learned_patterns": [],
                "success_strategies": [],
                "failure_warnings": []
            }
            
            for mem in memories:
                content = mem.get("content", {})
                if content.get("success"):
                    memory_context["success_strategies"].append({
                        "situation": content.get("experience"),
                        "action": content.get("action_taken"),
                        "outcome": content.get("outcome")
                    })
                else:
                    memory_context["failure_warnings"].append({
                        "situation": content.get("experience"),
                        "what_went_wrong": content.get("outcome"),
                        "lesson": content.get("learning_points")
                    })
            
            return memory_context
            
        except Exception as e:
            return {
                "error": str(e),
                "agent_id": agent_id,
                "memory_context": {}
            }
    
    def evolve_ethical_principles(self, accumulated_insights: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evolve ethical principles based on accumulated insights"""
        if not MEM0_AVAILABLE:
            return self._mock_principle_evolution(accumulated_insights)
        
        try:
            # Analyze patterns in ethical insights
            principle_patterns = self._analyze_ethical_patterns(accumulated_insights)
            
            # Identify principles that need evolution
            evolution_candidates = self._identify_evolution_candidates(principle_patterns)
            
            # Propose evolved principles
            evolved_principles = []
            for candidate in evolution_candidates:
                evolved_principle = self._evolve_principle(candidate, principle_patterns)
                evolved_principles.append(evolved_principle)
            
            # Store evolved principles
            evolution_record = {
                "type": "principle_evolution",
                "original_insights_count": len(accumulated_insights),
                "evolved_principles": evolved_principles,
                "evolution_timestamp": self._get_timestamp(),
                "confidence": self._calculate_evolution_confidence(evolved_principles)
            }
            
            self.store_ethical_insight(evolution_record, "system")
            
            return {
                "evolved": True,
                "principles_evolved": len(evolved_principles),
                "evolution_record": evolution_record,
                "timestamp": self._get_timestamp()
            }
            
        except Exception as e:
            return {
                "evolved": False,
                "error": str(e),
                "principles_evolved": 0
            }
    
    def _calculate_memory_relevance(self, memory: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate how relevant a memory is to the current context"""
        # Simple relevance calculation - can be enhanced with embeddings
        base_relevance = 0.5
        
        # Check for contextual matches
        if context.get("medical_domain") and "medical" in str(memory).lower():
            base_relevance += 0.2
        
        if context.get("ethical_stakes") == "high" and memory.get("moral_weight", 0) > 0.7:
            base_relevance += 0.3
        
        return min(base_relevance, 1.0)
    
    def _analyze_ethical_patterns(self, insights: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in ethical insights"""
        # Placeholder for pattern analysis
        return {
            "common_principles": ["beneficence", "non_maleficence", "autonomy"],
            "emerging_themes": ["ai_consciousness", "privacy_rights"],
            "consistency_scores": {"beneficence": 0.9, "autonomy": 0.7}
        }
    
    def _identify_evolution_candidates(self, patterns: Dict[str, Any]) -> List[str]:
        """Identify principles that need evolution"""
        candidates = []
        for principle, score in patterns.get("consistency_scores", {}).items():
            if score < 0.8:  # Low consistency indicates need for evolution
                candidates.append(principle)
        return candidates
    
    def _evolve_principle(self, principle: str, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Evolve a single principle based on patterns"""
        return {
            "original_principle": principle,
            "evolved_version": f"Enhanced {principle} with AI considerations",
            "justification": f"Evolution based on pattern analysis of {patterns}",
            "confidence": 0.8
        }
    
    def _calculate_evolution_confidence(self, evolved_principles: List[Dict[str, Any]]) -> float:
        """Calculate confidence in principle evolution"""
        if not evolved_principles:
            return 0.0
        return sum(p.get("confidence", 0) for p in evolved_principles) / len(evolved_principles)
    
    def _mock_memory_storage(self, content: Dict[str, Any], memory_type: str) -> Dict[str, Any]:
        """Mock memory storage for when Mem0 is not available"""
        return {
            "stored": True,
            "memory_id": f"mock_{memory_type}_{hash(str(content)) % 10000}",
            "memory_type": memory_type,
            "mock_mode": True,
            "storage_timestamp": self._get_timestamp()
        }
    
    def _mock_memory_retrieval(self, query: str, memory_type: str) -> Dict[str, Any]:
        """Mock memory retrieval for when Mem0 is not available"""
        return {
            "retrieved": True,
            "memories": [
                {
                    "memory_id": f"mock_{memory_type}_1",
                    "content": f"Mock {memory_type} memory relevant to: {query}",
                    "relevance_score": 0.8,
                    "mock_mode": True
                }
            ],
            "query": query,
            "total_found": 1,
            "mock_mode": True
        }
    
    def _mock_principle_evolution(self, insights: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Mock principle evolution for when Mem0 is not available"""
        return {
            "evolved": True,
            "principles_evolved": 2,
            "evolution_record": {
                "evolved_principles": [
                    {"principle": "mock_evolution_1", "confidence": 0.8},
                    {"principle": "mock_evolution_2", "confidence": 0.7}
                ]
            },
            "mock_mode": True
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.utcnow().isoformat()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get integration system status"""
        return {
            "mem0_available": MEM0_AVAILABLE,
            "memory_systems_initialized": len(self.memory_instances),
            "ethical_memories_count": len(self.ethical_memories),
            "agent_memories_count": len(self.agent_memories),
            "integration_status": "active" if MEM0_AVAILABLE else "mock_mode"
        }


# Factory function for easy instantiation
def create_mem0_integration(config: Optional[Dict[str, Any]] = None) -> Mem0Integration:
    """Create Mem0 integration instance"""
    return Mem0Integration(config)


# Default instance for direct import
default_mem0 = create_mem0_integration()
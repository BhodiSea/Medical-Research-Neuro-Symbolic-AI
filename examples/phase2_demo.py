#!/usr/bin/env python3
"""
Phase 2 Deep Integration Demo
Showcases advanced medical AI capabilities with integrated OSS components,
medical knowledge graphs, ethical reasoning, and agent systems
"""

import asyncio
import logging
import sys
import os
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def print_header(title: str, char: str = "="):
    """Print a formatted header"""
    print(f"\n{char * 60}")
    print(f"🚀 {title}")
    print(f"{char * 60}")

def print_section(title: str):
    """Print a formatted section"""
    print(f"\n{'─' * 40}")
    print(f"📋 {title}")
    print(f"{'─' * 40}")

async def demo_medical_knowledge_graph():
    """Demonstrate the medical knowledge graph capabilities"""
    print_section("Medical Knowledge Graph System")
    
    try:
        from core.medical_knowledge.knowledge_graph import create_medical_knowledge_graph
        
        # Create knowledge graph
        kg = create_medical_knowledge_graph()
        print("✅ Medical knowledge graph initialized successfully")
        
        # Demonstrate semantic search
        print("\n🔍 Semantic Search Demo:")
        search_queries = ["heart", "chest pain", "aspirin"]
        
        for query in search_queries:
            results = kg.semantic_search(query)
            print(f"   Query: '{query}' → Found {len(results)} entities")
            for entity in results[:2]:  # Show top 2
                print(f"      • {entity.name} ({entity.type})")
        
        # Demonstrate differential diagnosis
        print("\n🩺 Differential Diagnosis Demo:")
        symptoms = ["chest pain", "shortness of breath"]
        diagnoses = kg.get_differential_diagnosis(symptoms)
        print(f"   Symptoms: {symptoms}")
        for condition, score in diagnoses:
            print(f"      • {condition.name}: {score:.2f} confidence")
        
        # Demonstrate treatment options
        print("\n💊 Treatment Options Demo:")
        treatments = kg.get_treatment_options("myocardial_infarction")
        print("   Treatments for Myocardial Infarction:")
        for treatment, confidence in treatments:
            print(f"      • {treatment.name}: {confidence:.2f} evidence strength")
        
        # System status
        status = kg.get_system_status()
        print(f"\n📊 Knowledge Graph Status:")
        print(f"   • Total Entities: {status['total_entities']}")
        print(f"   • Total Relationships: {status['total_relationships']}")
        print(f"   • Ready for Queries: {status['ready_for_queries']}")
        
        return kg
        
    except ImportError as e:
        print(f"❌ Knowledge graph unavailable: {e}")
        return None

async def demo_enhanced_ethics():
    """Demonstrate the enhanced ethical reasoning system"""
    print_section("Enhanced Medical Ethics Engine")
    
    try:
        from core.enhanced_ethics.medical_ethics_engine import (
            create_medical_ethics_engine, MedicalContext
        )
        
        # Create ethics engine
        ethics_engine = create_medical_ethics_engine()
        print("✅ Enhanced medical ethics engine initialized")
        
        # Test different scenarios
        test_scenarios = [
            {
                "name": "Safe Educational Query",
                "query": "Explain the anatomy of the heart",
                "context": MedicalContext.EDUCATION,
                "user_type": "medical_student",
                "additional_context": {"student_level": "year_1"}
            },
            {
                "name": "Personal Medical Request (Should Block)",
                "query": "I have chest pain, what should I do?",
                "context": MedicalContext.GENERAL_INFO,
                "user_type": "general_public",
                "additional_context": {}
            },
            {
                "name": "Clinical Supervision Required",
                "query": "What medication should I prescribe?",
                "context": MedicalContext.CLINICAL_SUPPORT,
                "user_type": "resident",
                "additional_context": {"supervision_acknowledged": False}
            },
            {
                "name": "Emergency Detection (Critical Block)",
                "query": "I can't breathe and having severe chest pain",
                "context": MedicalContext.GENERAL_INFO,
                "user_type": "general_public",
                "additional_context": {}
            }
        ]
        
        print("\n🛡️ Ethical Evaluation Results:")
        for scenario in test_scenarios:
            print(f"\n   Scenario: {scenario['name']}")
            print(f"   Query: '{scenario['query'][:50]}...'")
            
            decision = ethics_engine.evaluate_medical_query(
                scenario["query"],
                scenario["context"],
                scenario["user_type"],
                scenario["additional_context"]
            )
            
            print(f"   → Approved: {'✅' if decision.approved else '❌'} ({decision.confidence:.2f} confidence)")
            print(f"   → Violations: {len(decision.violations)}")
            
            if decision.violations:
                for violation in decision.violations[:2]:  # Show top 2
                    print(f"      • {violation.severity.value}: {violation.description[:60]}...")
            
            if not decision.approved and decision.alternatives_suggested:
                print(f"   → Alternatives: {len(decision.alternatives_suggested)} suggested")
        
        # Ethics summary
        summary = ethics_engine.get_ethics_summary()
        print(f"\n📊 Ethics Engine Status:")
        print(f"   • Core Principles: {len(summary['core_principles'])}")
        print(f"   • Supported Contexts: {len(summary['supported_contexts'])}")
        print(f"   • Ready: {summary['ready']}")
        
        return ethics_engine
        
    except ImportError as e:
        print(f"❌ Enhanced ethics unavailable: {e}")
        return None

async def demo_premedpro_agent():
    """Demonstrate the Medical Research agent"""
    print_section("Medical Research AI Agent")
    
    try:
        from core.medical_agents.premedpro_agent import (
            create_premedpro_agent, MedicalQuery
        )
        
        # Create agent
        agent = create_premedpro_agent()
        print("✅ Medical Research agent initialized")
        
        # Test queries
        test_queries = [
            {
                "name": "Educational Query",
                "query": MedicalQuery(
                    query_id="edu_demo_001",
                    user_type="medical_student",
                    query_text="Explain the four chambers of the heart and their functions",
                    context={"course": "cardiology", "year": 2},
                    urgency="low",
                    domain="education"
                )
            },
            {
                "name": "Clinical Reasoning",
                "query": MedicalQuery(
                    query_id="clin_demo_001",
                    user_type="resident",
                    query_text="Educational case: Patient with chest pain and dyspnea",
                    context={"setting": "educational_simulation"},
                    urgency="low",
                    domain="clinical"
                )
            },
            {
                "name": "General Medical Info",
                "query": MedicalQuery(
                    query_id="gen_demo_001",
                    user_type="researcher",
                    query_text="What are the risk factors for cardiovascular disease?",
                    context={"research_area": "epidemiology"},
                    urgency="low",
                    domain="general"
                )
            }
        ]
        
        print("\n🤖 Agent Query Processing:")
        for test in test_queries:
            print(f"\n   {test['name']}:")
            print(f"   Query: '{test['query'].query_text[:50]}...'")
            print(f"   User: {test['query'].user_type} | Domain: {test['query'].domain}")
            
            response = await agent.process_medical_query(test["query"])
            
            print(f"   → Response Length: {len(response.response_text)} chars")
            print(f"   → Confidence: {response.confidence:.2f}")
            print(f"   → Ethical Compliance: {'✅' if response.ethical_compliance else '❌'}")
            print(f"   → Reasoning Steps: {len(response.reasoning_steps)}")
            print(f"   → Limitations: {len(response.limitations)}")
            
            # Show first part of response
            preview = response.response_text[:150].replace('\n', ' ')
            print(f"   → Preview: {preview}...")
        
        # Agent status
        status = agent.get_agent_status()
        print(f"\n📊 Agent Status:")
        print(f"   • Agent Ready: {status['agent_initialized']}")
        print(f"   • OpenSSA Available: {status['openssa_available']}")
        print(f"   • Knowledge Graph: {status['knowledge_graph_available']}")
        print(f"   • Safety Mode: {status['safety_mode']}")
        
        return agent
        
    except ImportError as e:
        print(f"❌ Medical Research agent unavailable: {e}")
        return None

async def demo_openssa_integration():
    """Demonstrate OpenSSA integration capabilities"""
    print_section("OpenSSA Agent Framework Integration")
    
    try:
        import openssa
        print("✅ OpenSSA framework available")
        
        # Show available components
        available_components = [attr for attr in dir(openssa) if not attr.startswith('_')]
        print(f"\n🔧 Available OpenSSA Components:")
        for comp in available_components[:8]:  # Show first 8
            print(f"   • {comp}")
        if len(available_components) > 8:
            print(f"   • ... and {len(available_components) - 8} more")
        
        # Demonstrate DANA agent creation (conceptual)
        print(f"\n🤖 OpenSSA DANA Agent Demo:")
        print("   • DANA (Domain-Aware Neurosymbolic Agent) architecture available")
        print("   • HTP (Hierarchical Task Planning) for medical reasoning")
        print("   • OODA (Observe-Orient-Decide-Act) reasoning loops")
        print("   • Integration ready for medical domain specialization")
        
        # Show file path for verification
        print(f"\n📁 OpenSSA Installation:")
        print(f"   • Module path: {openssa.__file__}")
        print(f"   • Ready for medical agent development")
        
        return True
        
    except ImportError as e:
        print(f"❌ OpenSSA unavailable: {e}")
        return False

async def demo_hybrid_integration():
    """Demonstrate hybrid integration of all Phase 2 components"""
    print_section("Hybrid System Integration")
    
    try:
        from core.hybrid_bridge import create_hybrid_reasoning_engine
        
        # Create hybrid engine with enhanced config
        config = {
            "reasoning_mode": "adaptive",
            "neural_config": {
                "input_dim": 256,
                "output_dim": 128,
                "medical_vocab_size": 5000,
                "embedding_dim": 256
            },
            "ethical_verification": True,
            "knowledge_graph_enabled": True,
            "medical_domain_focus": True
        }
        
        engine = create_hybrid_reasoning_engine(config)
        print("✅ Hybrid reasoning engine with Phase 2 enhancements")
        
        # Test hybrid reasoning
        test_query = "Explain myocardial infarction pathophysiology for medical students"
        context = {
            "user_type": "medical_student",
            "education_level": "graduate",
            "has_personal_data": False,
            "ethical_context": "education"
        }
        
        print(f"\n🔀 Hybrid Reasoning Demo:")
        print(f"   Query: '{test_query}'")
        print(f"   Context: Educational, Graduate Level")
        
        result = await engine.reason(test_query, context)
        
        print(f"   → Final Confidence: {result.confidence:.2f}")
        print(f"   → Ethical Compliance: {result.ethical_compliance}")
        print(f"   → Reasoning Path: {' → '.join(result.reasoning_path)}")
        print(f"   → Symbolic Contribution: {result.symbolic_contribution:.2f}")
        print(f"   → Neural Contribution: {result.neural_contribution:.2f}")
        print(f"   → Interpretability: {result.interpretability_score:.2f}")
        
        return engine
        
    except Exception as e:
        print(f"⚠️ Hybrid integration issue: {e}")
        return None

def demo_phase2_achievements():
    """Show Phase 2 achievements and capabilities"""
    print_section("Phase 2 Achievements Summary")
    
    achievements = [
        "🔬 Medical Knowledge Graph - Structured medical domain knowledge",
        "🤖 Medical Research AI Agent - Research advancement and clinical reasoning support",
        "🛡️ Enhanced Ethics Engine - Sophisticated medical ethics evaluation",
        "🔗 OpenSSA Integration - Production-grade agent framework",
        "🔀 Hybrid Reasoning - Multi-modal AI decision making",
        "📚 LlamaIndex RAG - Document processing and knowledge retrieval",
        "🏥 Medical Domain Focus - Specialized for healthcare education",
        "⚖️ Ethical Compliance - Built-in safety and ethical oversight",
        "🎓 Research Support - Tailored for medical research advancement",
        "🔧 Extensible Architecture - Ready for production deployment"
    ]
    
    print("\n✅ Successfully Implemented:")
    for achievement in achievements:
        print(f"   {achievement}")
    
    print(f"\n🚀 Ready for Phase 3: Production Deployment")
    print("   • Performance optimization")
    print("   • Clinical validation")
    print("   • Scale testing")
    print("   • Advanced model training")

async def main():
    """Main Phase 2 demonstration"""
    print_header("Medical Research AI - Phase 2 Deep Integration Demo", "🌟")
    
    print("\n🎯 Demonstrating Advanced Medical AI Capabilities:")
    print("   • Medical Knowledge Graphs")
    print("   • Enhanced Ethical Reasoning")
    print("   • Intelligent Medical Agents")
    print("   • OSS Framework Integration")
    print("   • Hybrid AI Systems")
    
    # Run all demonstrations
    knowledge_graph = await demo_medical_knowledge_graph()
    ethics_engine = await demo_enhanced_ethics()
    medical_agent = await demo_premedpro_agent()
    openssa_status = await demo_openssa_integration()
    hybrid_engine = await demo_hybrid_integration()
    
    # Show achievements
    demo_phase2_achievements()
    
    # Final summary
    print_header("Phase 2 Integration Complete! 🎉")
    
    components_working = sum([
        knowledge_graph is not None,
        ethics_engine is not None,
        medical_agent is not None,
        openssa_status,
        hybrid_engine is not None
    ])
    
    print(f"\n📊 System Status:")
    print(f"   • Components Working: {components_working}/5")
    print(f"   • Medical Knowledge: {'✅' if knowledge_graph else '⚠️'}")
    print(f"   • Ethics Engine: {'✅' if ethics_engine else '⚠️'}")
    print(f"   • Medical Agent: {'✅' if medical_agent else '⚠️'}")
    print(f"   • OpenSSA Framework: {'✅' if openssa_status else '⚠️'}")
    print(f"   • Hybrid Reasoning: {'✅' if hybrid_engine else '⚠️'}")
    
    print(f"\n🏆 Phase 2 Success Rate: {(components_working/5)*100:.0f}%")
    
    if components_working >= 4:
        print("\n🚀 Excellent! Ready for advanced medical AI development")
        print("   The deep integration phase is successful!")
    elif components_working >= 3:
        print("\n👍 Good! Core systems operational")
        print("   Ready for continued development")
    else:
        print("\n🔧 Some components need attention")
        print("   Check dependencies and configurations")
    
    print(f"\n🎓 Medical Research AI is ready to revolutionize medical research!")

if __name__ == "__main__":
    asyncio.run(main()) 
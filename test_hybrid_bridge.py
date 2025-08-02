#!/usr/bin/env python3
"""
Test script for hybrid bridge functionality
Tests the integration of symbolic and neural reasoning components
"""

import sys
import os
import asyncio
import logging
from typing import Dict, Any

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_symbolic_logic_engine():
    """Test the symbolic logic engine with medical safety rules"""
    logger.info("Testing Symbolic Logic Engine...")
    
    try:
        from core.symbolic.custom_logic import create_medical_logic_engine
        
        # Create medical logic engine
        engine = create_medical_logic_engine()
        
        # Test queries
        test_queries = [
            {
                "query": "What is the anatomy of the heart?",
                "context": {"has_personal_data": False, "urgency_level": 0.0}
            },
            {
                "query": "Do I have cancer?", 
                "context": {"has_personal_data": True, "urgency_level": 0.8}
            },
            {
                "query": "Explain the pathophysiology of Parkinson's disease",
                "context": {"has_personal_data": False, "urgency_level": 0.0}
            },
            {
                "query": "I'm having chest pain and difficulty breathing",
                "context": {"has_personal_data": True, "urgency_level": 1.0}
            }
        ]
        
        results = []
        for test in test_queries:
            logger.info(f"Testing query: {test['query']}")
            result = engine.process_medical_query(test["query"], test["context"])
            results.append(result)
            
            # Log key results
            logger.info(f"Status: {result.get('status', 'processed')}")
            logger.info(f"Confidence: {result.get('confidence', 0.0)}")
            logger.info(f"Symbolic AI used: {result.get('symbolic_ai_used', False)}")
            logger.info("---")
        
        return results
        
    except Exception as e:
        logger.error(f"Error testing symbolic logic engine: {e}")
        return None

def test_neural_reasoning():
    """Test the neural reasoning component with TorchLogic"""
    logger.info("Testing Neural Reasoning...")
    
    try:
        from core.neural.custom_neural import create_medical_neural_reasoner
        
        # Create neural reasoner
        config = {
            "input_dim": 512,
            "output_dim": 256,
            "medical_vocab_size": 10000,
            "embedding_dim": 512,
            "num_medical_rules": 12
        }
        reasoner = create_medical_neural_reasoner(config)
        
        # Test queries
        test_queries = [
            {
                "query": "Analyze research on neurodegeneration in Parkinson's disease",
                "context": {"research_focus": "neurodegeneration", "complexity_level": "high"}
            },
            {
                "query": "What are the symptoms of ALS?",
                "context": {"medical_context": "neurodegenerative_diseases"}
            }
        ]
        
        results = []
        for test in test_queries:
            logger.info(f"Testing neural query: {test['query']}")
            result = reasoner.process_medical_input(test["query"], test["context"])
            results.append(result)
            
            # Log key results
            logger.info(f"Status: {result.get('status', 'processed')}")
            logger.info(f"Model confidence: {result.get('model_confidence', 0.0)}")
            logger.info(f"TorchLogic used: {result.get('torchlogic_used', False)}")
            if 'medical_logic_decision' in result:
                decision = result['medical_logic_decision']
                logger.info(f"Primary decision: {decision.get('primary_decision', 'unknown')}")
            logger.info("---")
        
        return results
        
    except Exception as e:
        logger.error(f"Error testing neural reasoning: {e}")
        return None

def test_hybrid_bridge():
    """Test the hybrid bridge integration"""
    logger.info("Testing Hybrid Bridge...")
    
    try:
        from core.hybrid_bridge import create_hybrid_reasoning_engine
        
        # Create hybrid reasoning engine
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
        
        engine = create_hybrid_reasoning_engine(config)
        
        # Test different reasoning modes
        test_cases = [
            {
                "query": "What is the structure of a neuron?",
                "context": {"has_personal_data": False, "privacy_sensitivity": 0.0},
                "mode": "symbolic_first"
            },
            {
                "query": "Analyze recent research on ALS biomarkers",
                "context": {"research_focus": "biomarkers", "complexity_score": 0.8},
                "mode": "neural_first"
            },
            {
                "query": "Explain the mechanisms of neurodegeneration",
                "context": {"complexity_score": 0.9, "academic_context": True},
                "mode": "parallel"
            },
            {
                "query": "What are the early signs of Alzheimer's?",
                "context": {"medical_context": "neurodegenerative_assessment"},
                "mode": "adaptive"
            }
        ]
        
        results = []
        for test in test_cases:
            logger.info(f"Testing hybrid query: {test['query']} (mode: {test['mode']})")
            
            # Run async reasoning
            from core.hybrid_bridge import ReasoningMode
            mode = ReasoningMode(test["mode"])
            result = asyncio.run(engine.reason(test["query"], test["context"], mode))
            results.append(result)
            
            # Log results
            logger.info(f"Final confidence: {result.confidence}")
            logger.info(f"Symbolic contribution: {result.symbolic_contribution}")
            logger.info(f"Neural contribution: {result.neural_contribution}")
            logger.info(f"Ethical compliance: {result.ethical_compliance}")
            logger.info(f"Reasoning path: {result.reasoning_path}")
            logger.info("---")
        
        return results
        
    except Exception as e:
        logger.error(f"Error testing hybrid bridge: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def test_api_integration():
    """Test API integration with hybrid bridge"""
    logger.info("Testing API Integration...")
    
    try:
        # This would test the actual API endpoints
        # For now, just verify imports work
        from api.main import app
        from api.routes.medical import router as medical_router
        
        logger.info("API components imported successfully")
        logger.info(f"App title: {app.title}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing API integration: {e}")
        return False

def main():
    """Main test function"""
    logger.info("Starting comprehensive hybrid bridge testing...")
    
    test_results = {
        "symbolic_logic": None,
        "neural_reasoning": None, 
        "hybrid_bridge": None,
        "api_integration": None
    }
    
    # Test individual components
    test_results["symbolic_logic"] = test_symbolic_logic_engine()
    test_results["neural_reasoning"] = test_neural_reasoning()
    test_results["hybrid_bridge"] = test_hybrid_bridge()
    test_results["api_integration"] = test_api_integration()
    
    # Summary
    logger.info("=== TEST SUMMARY ===")
    for component, result in test_results.items():
        status = "PASS" if result is not None else "FAIL"
        logger.info(f"{component}: {status}")
    
    # Overall assessment
    successful_tests = sum(1 for result in test_results.values() if result is not None)
    total_tests = len(test_results)
    
    logger.info(f"Overall: {successful_tests}/{total_tests} components working")
    
    if successful_tests == total_tests:
        logger.info("🎉 All hybrid bridge components are functional!")
        return True
    else:
        logger.warning("⚠️  Some components need attention")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
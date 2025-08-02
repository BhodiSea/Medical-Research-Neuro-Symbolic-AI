#!/usr/bin/env python3
"""
Simple Integration Test Script

This script tests the newly created integration wrappers without triggering
lock blocking issues by only testing imports and basic initialization.
"""

import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_monai_import():
    """Test MONAI integration import."""
    try:
        from core.neural.monai_integration import MONAIIntegration
        logger.info("✅ MONAI Integration: Import successful")
        
        # Test basic initialization
        config = {"deterministic_seed": 42, "gpu_available": False}
        monai_integration = MONAIIntegration(config)
        logger.info("✅ MONAI Integration: Initialization successful")
        
        return True
    except Exception as e:
        logger.error(f"❌ MONAI Integration Test Failed: {e}")
        return False

def test_medclip_import():
    """Test MedCLIP integration import."""
    try:
        from core.neural.medclip_integration import MedCLIPIntegration
        logger.info("✅ MedCLIP Integration: Import successful")
        
        # Test basic initialization
        config = {"model_type": "MedCLIP", "device": "auto"}
        medclip_integration = MedCLIPIntegration(config)
        logger.info("✅ MedCLIP Integration: Initialization successful")
        
        return True
    except Exception as e:
        logger.error(f"❌ MedCLIP Integration Test Failed: {e}")
        return False

def test_biobert_import():
    """Test BioBERT integration import."""
    try:
        from core.neural.biobert_integration import BioBERTIntegration
        logger.info("✅ BioBERT Integration: Import successful")
        
        # Test basic initialization
        config = {"model_name": "dmis-lab/biobert-base-cased-v1.2", "device": "auto"}
        biobert_integration = BioBERTIntegration(config)
        logger.info("✅ BioBERT Integration: Initialization successful")
        
        return True
    except Exception as e:
        logger.error(f"❌ BioBERT Integration Test Failed: {e}")
        return False

def test_langchain_import():
    """Test LangChain integration import."""
    try:
        from orchestration.langchain_integration import LangChainIntegration
        logger.info("✅ LangChain Integration: Import successful")
        
        # Test basic initialization
        config = {
            "openai_api_key": None,
            "model_name": "gpt-3.5-turbo-instruct",
            "chat_model_name": "gpt-4"
        }
        langchain_integration = LangChainIntegration(config)
        logger.info("✅ LangChain Integration: Initialization successful")
        
        return True
    except Exception as e:
        logger.error(f"❌ LangChain Integration Test Failed: {e}")
        return False

def test_autogen_import():
    """Test AutoGen integration import."""
    try:
        from orchestration.autogen_integration import AutoGenIntegration
        logger.info("✅ AutoGen Integration: Import successful")
        
        # Test basic initialization
        config = {"openai_api_key": None}
        autogen_integration = AutoGenIntegration(config)
        logger.info("✅ AutoGen Integration: Initialization successful")
        
        return True
    except Exception as e:
        logger.error(f"❌ AutoGen Integration Test Failed: {e}")
        return False

def test_existing_integrations():
    """Test existing integrations to ensure they still work."""
    try:
        # Test DeepChem integration
        from core.neural.deepchem_integration import DeepChemIntegration
        deepchem_integration = DeepChemIntegration()
        logger.info("✅ DeepChem Integration: Import successful")
        
        # Test SymbolicAI integration
        from core.neural.symbolicai_integration import SymbolicAIIntegration
        symbolicai_integration = SymbolicAIIntegration()
        logger.info("✅ SymbolicAI Integration: Import successful")
        
        # Test CrewAI integration
        from orchestration.agents.crewai_integration import CrewAIIntegration
        crewai_integration = CrewAIIntegration()
        logger.info("✅ CrewAI Integration: Import successful")
        
        # Test AIX360 integration
        from ethical_audit.py_bindings.aix360_integration import AIX360Integration
        aix360_integration = AIX360Integration()
        logger.info("✅ AIX360 Integration: Import successful")
        
        return True
    except Exception as e:
        logger.error(f"❌ Existing Integration Test Failed: {e}")
        return False

def main():
    """Main function to run the simple integration tests."""
    logger.info("🚀 Starting Simple Integration Tests")
    logger.info("=" * 50)
    
    test_results = {}
    
    # Test new integrations
    test_results["MONAI"] = test_monai_import()
    test_results["MedCLIP"] = test_medclip_import()
    test_results["BioBERT"] = test_biobert_import()
    test_results["LangChain"] = test_langchain_import()
    test_results["AutoGen"] = test_autogen_import()
    
    # Test existing integrations
    test_results["Existing"] = test_existing_integrations()
    
    # Summary
    logger.info("=" * 50)
    logger.info("📊 Integration Test Results Summary")
    logger.info("=" * 50)
    
    passed = 0
    total = len(test_results)
    
    for integration, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{integration:12} : {status}")
        if result:
            passed += 1
    
    logger.info("=" * 50)
    logger.info(f"Overall Result: {passed}/{total} integrations passed")
    
    if passed == total:
        logger.info("🎉 All integrations are working correctly!")
        logger.info("💡 Note: These tests only verify imports and initialization.")
        logger.info("   Full functionality tests would require the actual AI systems.")
        return True
    else:
        logger.warning(f"⚠️  {total - passed} integration(s) need attention")
        return False

if __name__ == "__main__":
    try:
        success = main()
        if success:
            logger.info("✅ All integration tests completed successfully!")
            sys.exit(0)
        else:
            logger.error("❌ Some integration tests failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Unexpected error during testing: {e}")
        sys.exit(1) 
#!/usr/bin/env python3
"""
Comprehensive Test Script for New AI System Integrations

This script tests all the newly created integration wrappers for the medical research AI system.
It verifies that each integration can be imported, initialized, and provides basic functionality.
"""

import sys
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_monai_integration():
    """Test MONAI integration for medical imaging."""
    logger.info("Testing MONAI Integration...")
    
    try:
        from core.neural.monai_integration import MONAIIntegration
        
        # Initialize integration
        config = {"deterministic_seed": 42, "gpu_available": False}
        monai_integration = MONAIIntegration(config)
        
        # Test brain MRI analysis
        mri_result = monai_integration.analyze_brain_mri("sample_mri.nii.gz", "segmentation")
        logger.info(f"✅ MONAI MRI Analysis: {mri_result['status']}")
        
        # Test PET scan analysis
        pet_result = monai_integration.analyze_pet_scan("sample_pet.nii.gz", "FDG")
        logger.info(f"✅ MONAI PET Analysis: {pet_result['status']}")
        
        # Test Parkinson's detection
        parkinsons_result = monai_integration.detect_parkinsons_features("sample_mri.nii.gz")
        logger.info(f"✅ MONAI Parkinson's Detection: {parkinsons_result['status']}")
        
        # Test Alzheimer's detection
        alzheimers_result = monai_integration.detect_alzheimers_features("sample_mri.nii.gz")
        logger.info(f"✅ MONAI Alzheimer's Detection: {alzheimers_result['status']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ MONAI Integration Test Failed: {e}")
        return False

def test_medclip_integration():
    """Test MedCLIP integration for medical vision-language understanding."""
    logger.info("Testing MedCLIP Integration...")
    
    try:
        from core.neural.medclip_integration import MedCLIPIntegration
        
        # Initialize integration
        config = {"model_type": "MedCLIP", "device": "auto"}
        medclip_integration = MedCLIPIntegration(config)
        
        # Test medical image analysis
        analysis_result = medclip_integration.analyze_medical_image(
            "sample_brain_mri.jpg", 
            "brain tumor present", 
            "brain"
        )
        logger.info(f"✅ MedCLIP Medical Analysis: {analysis_result['status']}")
        
        # Test medical report generation
        report_result = medclip_integration.generate_medical_report(
            "sample_brain_mri.jpg", 
            "brain_mri"
        )
        logger.info(f"✅ MedCLIP Medical Report: {report_result['status']}")
        
        # Test abnormality detection
        abnormality_result = medclip_integration.detect_brain_abnormalities(
            "sample_brain_mri.jpg", 
            "tumor"
        )
        logger.info(f"✅ MedCLIP Abnormality Detection: {abnormality_result['status']}")
        
        # Test image comparison
        comparison_result = medclip_integration.compare_medical_images(
            ["sample_mri1.jpg", "sample_mri2.jpg"], 
            "similarity"
        )
        logger.info(f"✅ MedCLIP Image Comparison: {comparison_result['status']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ MedCLIP Integration Test Failed: {e}")
        return False

def test_biobert_integration():
    """Test BioBERT integration for biomedical text understanding."""
    logger.info("Testing BioBERT Integration...")
    
    try:
        from core.neural.biobert_integration import BioBERTIntegration
        
        # Initialize integration
        config = {"model_name": "dmis-lab/biobert-base-cased-v1.2", "device": "auto"}
        biobert_integration = BioBERTIntegration(config)
        
        # Test entity extraction
        sample_text = "Parkinson's disease is characterized by alpha-synuclein aggregation in the substantia nigra."
        entity_result = biobert_integration.extract_biomedical_entities(sample_text)
        logger.info(f"✅ BioBERT Entity Extraction: {entity_result['status']}")
        
        # Test literature analysis
        literature_result = biobert_integration.analyze_medical_literature(sample_text, "disease")
        logger.info(f"✅ BioBERT Literature Analysis: {literature_result['status']}")
        
        # Test disease extraction
        disease_result = biobert_integration.extract_disease_mentions(sample_text)
        logger.info(f"✅ BioBERT Disease Extraction: {disease_result['status']}")
        
        # Test drug extraction
        drug_text = "Levodopa and carbidopa are commonly used to treat Parkinson's disease."
        drug_result = biobert_integration.extract_drug_mentions(drug_text)
        logger.info(f"✅ BioBERT Drug Extraction: {drug_result['status']}")
        
        # Test biomarker extraction
        biomarker_text = "Alpha-synuclein and tau proteins are potential biomarkers for neurodegeneration."
        biomarker_result = biobert_integration.extract_biomarker_mentions(biomarker_text)
        logger.info(f"✅ BioBERT Biomarker Extraction: {biomarker_result['status']}")
        
        # Test concept search
        search_result = biobert_integration.search_medical_concepts("Parkinson's disease biomarkers")
        logger.info(f"✅ BioBERT Concept Search: {search_result['status']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ BioBERT Integration Test Failed: {e}")
        return False

async def test_langchain_integration():
    """Test LangChain integration for LLM application framework."""
    logger.info("Testing LangChain Integration...")
    
    try:
        from orchestration.langchain_integration import LangChainIntegration
        
        # Initialize integration
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
        logger.info(f"✅ LangChain Literature Analysis: {literature_result['status']}")
        
        # Test hypothesis generation
        hypothesis_result = langchain_integration.generate_research_hypotheses(
            "Current research on Parkinson's disease",
            "Alpha-synuclein aggregation is a key pathological feature"
        )
        logger.info(f"✅ LangChain Hypothesis Generation: {hypothesis_result['status']}")
        
        # Test clinical trial design
        trial_result = langchain_integration.design_clinical_trial(
            "Does alpha-synuclein inhibitor reduce Parkinson's symptoms?",
            "Early-stage Parkinson's patients",
            "Motor symptom improvement"
        )
        logger.info(f"✅ LangChain Trial Design: {trial_result['status']}")
        
        # Test agent response
        agent_result = await langchain_integration.run_medical_research_agent(
            "Analyze the relationship between alpha-synuclein and Parkinson's disease"
        )
        logger.info(f"✅ LangChain Agent Response: {agent_result['status']}")
        
        # Test custom chain creation
        chain_result = langchain_integration.create_reasoning_chain(
            "custom_analysis",
            "Analyze {text} for {focus}",
            ["text", "focus"]
        )
        logger.info(f"✅ LangChain Chain Creation: {chain_result['status']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ LangChain Integration Test Failed: {e}")
        return False

def test_autogen_integration():
    """Test AutoGen integration for multi-agent conversations."""
    logger.info("Testing AutoGen Integration...")
    
    try:
        from orchestration.autogen_integration import AutoGenIntegration
        
        # Initialize integration
        config = {"openai_api_key": None}  # No API key for testing
        autogen_integration = AutoGenIntegration(config)
        
        # Test medical research team creation
        team_result = autogen_integration.create_medical_research_team("test_team")
        logger.info(f"✅ AutoGen Team Creation: {team_result['status'] if 'status' in team_result else 'completed'}")
        
        # Test specialized agent creation
        agent_result = autogen_integration.create_specialized_agent(
            "neurologist",
            "Neurologist specializing in neurodegeneration",
            ["Parkinson's disease", "Alzheimer's", "ALS"]
        )
        logger.info(f"✅ AutoGen Agent Creation: {agent_result['status'] if 'status' in agent_result else 'completed'}")
        
        # Test available teams and agents
        teams = autogen_integration.get_available_teams()
        agents = autogen_integration.get_available_agents()
        logger.info(f"✅ AutoGen Available Teams: {len(teams)}, Available Agents: {len(agents)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ AutoGen Integration Test Failed: {e}")
        return False

def test_existing_integrations():
    """Test existing integrations to ensure they still work."""
    logger.info("Testing Existing Integrations...")
    
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

async def run_comprehensive_tests():
    """Run all integration tests."""
    logger.info("🚀 Starting Comprehensive Integration Tests")
    logger.info("=" * 60)
    
    test_results = {}
    
    # Test new integrations
    test_results["MONAI"] = test_monai_integration()
    test_results["MedCLIP"] = test_medclip_integration()
    test_results["BioBERT"] = test_biobert_integration()
    test_results["LangChain"] = await test_langchain_integration()
    test_results["AutoGen"] = test_autogen_integration()
    
    # Test existing integrations
    test_results["Existing"] = test_existing_integrations()
    
    # Summary
    logger.info("=" * 60)
    logger.info("📊 Integration Test Results Summary")
    logger.info("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for integration, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{integration:12} : {status}")
        if result:
            passed += 1
    
    logger.info("=" * 60)
    logger.info(f"Overall Result: {passed}/{total} integrations passed")
    
    if passed == total:
        logger.info("🎉 All integrations are working correctly!")
        return True
    else:
        logger.warning(f"⚠️  {total - passed} integration(s) need attention")
        return False

def main():
    """Main function to run the comprehensive tests."""
    try:
        # Run the comprehensive tests
        success = asyncio.run(run_comprehensive_tests())
        
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

if __name__ == "__main__":
    main() 
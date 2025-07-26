#!/usr/bin/env python3
"""
Basic Usage Example for PremedPro AI
Demonstrates hybrid neuro-symbolic reasoning with integrated OSS components
"""

import asyncio
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main demonstration function"""
    print("🧠 PremedPro AI - Basic Usage Example")
    print("=" * 50)
    
    # Example 1: Symbolic Reasoning
    print("\n1. Testing Symbolic Reasoning Integration...")
    test_symbolic_reasoning()
    
    # Example 2: Neural Reasoning  
    print("\n2. Testing Neural Reasoning Integration...")
    test_neural_reasoning()
    
    # Example 3: Mathematical Foundation
    print("\n3. Testing Mathematical Foundation...")
    test_mathematical_foundation()
    
    # Example 4: Ethical Audit
    print("\n4. Testing Ethical Audit System...")
    test_ethical_audit()
    
    # Example 5: Hybrid Reasoning (Main Feature)
    print("\n5. Testing Hybrid Reasoning...")
    asyncio.run(test_hybrid_reasoning())
    
    print("\n✅ All tests completed!")
    print("\nNext steps:")
    print("- Explore the individual components in detail")
    print("- Add your own medical domain knowledge")
    print("- Customize ethical constraints in config/ethical_constraints.yaml")
    print("- Start building your medical AI application!")

def test_symbolic_reasoning():
    """Test symbolic reasoning components"""
    try:
        from core.symbolic.custom_logic import create_medical_logic_engine
        
        # Create the medical logic engine
        engine = create_medical_logic_engine()
        
        # Test with a simple medical query
        query = "What are the main chambers of the heart?"
        context = {"user_type": "medical_student", "education_level": "undergraduate"}
        
        result = engine.process_medical_query(query, context)
        
        print(f"   Query: {query}")
        print(f"   Confidence: {result.get('confidence', 0):.2f}")
        print(f"   Status: {result.get('status', 'unknown')}")
        
        if result.get('status') != 'blocked':
            print("   ✅ Symbolic reasoning working correctly")
        else:
            print("   ⚠️  Query blocked by safety rules (expected for some queries)")
            
        # Test system status
        status = engine.get_system_status()
        print(f"   System Ready: {status.get('system_ready', False)}")
        
    except Exception as e:
        print(f"   ❌ Error in symbolic reasoning: {e}")

def test_neural_reasoning():
    """Test neural reasoning components"""
    try:
        from core.neural.custom_neural import create_medical_neural_reasoner
        
        # Create the neural reasoner
        reasoner = create_medical_neural_reasoner()
        
        # Test with a medical text
        input_text = "The patient presents with chest pain and shortness of breath"
        context = {"has_personal_data": False, "educational_context": True}
        
        result = reasoner.process_medical_input(input_text, context)
        
        print(f"   Input: {input_text}")
        print(f"   Reasoning Type: {result.get('reasoning_type', 'unknown')}")
        print(f"   Model Confidence: {result.get('model_confidence', 0):.2f}")
        print(f"   Ethical Score: {result.get('ethical_score', 0):.2f}")
        
        if 'error' not in result:
            print("   ✅ Neural reasoning working correctly")
        else:
            print(f"   ❌ Neural reasoning error: {result['error']}")
            
    except Exception as e:
        print(f"   ❌ Error in neural reasoning: {e}")

def test_mathematical_foundation():
    """Test mathematical foundation components"""
    try:
        from math_foundation.python_wrapper import create_math_foundation
        
        # Create the mathematical foundation
        foundation = create_math_foundation()
        
        # Test uncertainty principle calculation
        result = foundation.calculate_uncertainty_principle(
            knowledge_uncertainty=0.3,
            belief_uncertainty=0.4,
            hbar_analog=1.0
        )
        
        print(f"   Uncertainty Product: {result.get('uncertainty_product', 0):.3f}")
        print(f"   Minimum Bound: {result.get('minimum_bound', 0):.3f}")
        print(f"   Satisfies Principle: {result.get('satisfies_principle', False)}")
        
        # Test quantum entropy calculation
        amplitudes = [1+0j, 0.5+0.5j, 0.3+0.7j]
        uncertainties = [0.1, 0.15, 0.2]
        
        entropy_result = foundation.calculate_quantum_entropy(amplitudes, uncertainties)
        
        print(f"   Von Neumann Entropy: {entropy_result.get('von_neumann_entropy', 0):.3f}")
        print(f"   Total Entropy: {entropy_result.get('total_entropy', 0):.3f}")
        
        # Check system status
        status = foundation.get_system_status()
        julia_available = status.get('julia_available', False)
        
        if julia_available:
            print("   ✅ Julia mathematical foundation working")
        else:
            print("   ⚠️  Using Python fallbacks (Julia not available)")
            
    except Exception as e:
        print(f"   ❌ Error in mathematical foundation: {e}")

def test_ethical_audit():
    """Test ethical audit system"""
    try:
        # Import would fail since we haven't built the Rust component yet
        # from ethical_audit import EthicalAuditSystem
        
        # For now, just check if the Rust code compiles
        import subprocess
        import os
        
        if os.path.exists("ethical_audit/Cargo.toml"):
            result = subprocess.run(
                ["cargo", "check"], 
                cwd="ethical_audit", 
                capture_output=True, 
                text=True
            )
            
            if result.returncode == 0:
                print("   ✅ Ethical audit Rust code compiles successfully")
            else:
                print("   ⚠️  Rust compilation issues (expected without dependencies)")
                print(f"   Details: {result.stderr[:100]}...")
        else:
            print("   ❌ Ethical audit directory not found")
            
    except Exception as e:
        print(f"   ❌ Error testing ethical audit: {e}")

async def test_hybrid_reasoning():
    """Test the main hybrid reasoning system"""
    try:
        from core.hybrid_bridge import create_hybrid_reasoning_engine
        
        # Create the hybrid reasoning engine
        config = {
            "reasoning_mode": "adaptive",
            "neural_config": {
                "input_dim": 512,
                "output_dim": 256,
                "medical_vocab_size": 10000,
                "embedding_dim": 512
            }
        }
        
        engine = create_hybrid_reasoning_engine(config)
        
        # Test medical reasoning
        query = "Explain the pathophysiology of myocardial infarction"
        context = {
            "user_type": "medical_student",
            "education_level": "graduate",
            "has_personal_data": False
        }
        
        print(f"   Query: {query}")
        print("   Processing with hybrid reasoning...")
        
        result = await engine.reason(query, context)
        
        print(f"   Final Confidence: {result.confidence:.2f}")
        print(f"   Ethical Compliance: {result.ethical_compliance}")
        print(f"   Reasoning Path: {' → '.join(result.reasoning_path)}")
        print(f"   Symbolic Contribution: {result.symbolic_contribution:.2f}")
        print(f"   Neural Contribution: {result.neural_contribution:.2f}")
        print(f"   Interpretability Score: {result.interpretability_score:.2f}")
        
        if result.final_answer and not result.final_answer.get('error'):
            print("   ✅ Hybrid reasoning working correctly")
        else:
            print("   ⚠️  Hybrid reasoning completed with limitations")
            
        # Test system status
        status = engine.get_system_status()
        print(f"   Default Reasoning Mode: {status.get('default_reasoning_mode', 'unknown')}")
        
    except Exception as e:
        print(f"   ❌ Error in hybrid reasoning: {e}")

def demo_integrated_components():
    """Demonstrate integrated OSS components"""
    print("\n🔗 Integrated Open Source Components:")
    print("-" * 40)
    
    # Check SymbolicAI integration
    try:
        import sys
        import os
        sys.path.append(os.path.join('core', 'neural', 'symbolicai'))
        from symai import Symbol
        
        # Create a symbolic AI symbol
        sym = Symbol("The heart pumps blood through the body")
        print(f"   ✅ SymbolicAI: Symbol created - '{str(sym)[:50]}...'")
        
    except Exception as e:
        print(f"   ⚠️  SymbolicAI: {e}")
    
    # Check TorchLogic (basic check)
    try:
        torchlogic_path = os.path.join('core', 'neural', 'torchlogic')
        if os.path.exists(torchlogic_path):
            print(f"   ✅ TorchLogic: Repository available at {torchlogic_path}")
        else:
            print("   ❌ TorchLogic: Repository not found")
    except Exception as e:
        print(f"   ⚠️  TorchLogic: {e}")
    
    # Check OpenSSA (basic check)
    try:
        openssa_path = os.path.join('orchestration', 'openssa')
        if os.path.exists(openssa_path):
            print(f"   ✅ OpenSSA: Repository available at {openssa_path}")
        else:
            print("   ❌ OpenSSA: Repository not found")
    except Exception as e:
        print(f"   ⚠️  OpenSSA: {e}")
    
    # Check other submodules
    submodules = [
        ('NSTK', 'core/symbolic/nstk'),
        ('Nucleoid', 'core/symbolic/nucleoid'), 
        ('PEIRCE', 'core/symbolic/peirce')
    ]
    
    for name, path in submodules:
        if os.path.exists(path):
            print(f"   ✅ {name}: Repository available at {path}")
        else:
            print(f"   ❌ {name}: Repository not found at {path}")

if __name__ == "__main__":
    # Add the current directory to Python path for imports
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
    
    # First show integrated components
    demo_integrated_components()
    
    # Then run main demo
    main() 
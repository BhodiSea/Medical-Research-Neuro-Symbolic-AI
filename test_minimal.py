#!/usr/bin/env python3
"""Minimal test to check what's working"""

import warnings
warnings.filterwarnings('ignore')

def test_components():
    """Test each component individually"""
    
    # Test 1: Symbolic reasoning
    try:
        from core.symbolic.custom_logic import create_medical_logic_engine
        engine = create_medical_logic_engine()
        result = engine.process_medical_query("chest pain", {"has_personal_data": False})
        method = result.get('method', 'unknown')
        print(f"✅ Symbolic: {method}")
        symbolic_ok = method == 'knowledge_graph_enhanced_logic'
    except Exception as e:
        print(f"❌ Symbolic error: {e}")
        symbolic_ok = False
    
    # Test 2: Neural reasoning
    try:
        from core.neural.custom_neural import create_medical_neural_reasoner
        reasoner = create_medical_neural_reasoner()
        result = reasoner.process_medical_input("chest pain", {})
        confidence = result.get('model_confidence', 0.0)
        print(f"✅ Neural: confidence={confidence:.3f}")
        neural_ok = confidence > 0.0
    except Exception as e:
        print(f"❌ Neural error: {e}")
        neural_ok = False
    
    # Test 3: Hybrid bridge import
    try:
        from core.hybrid_bridge import create_hybrid_reasoning_engine
        print("✅ Hybrid bridge import: success")
        hybrid_import_ok = True
    except Exception as e:
        print(f"❌ Hybrid import error: {e}")
        hybrid_import_ok = False
    
    # Test 4: Hybrid bridge creation
    if hybrid_import_ok:
        try:
            engine = create_hybrid_reasoning_engine()
            print("✅ Hybrid bridge creation: success")
            hybrid_create_ok = True
        except Exception as e:
            print(f"❌ Hybrid creation error: {e}")
            hybrid_create_ok = False
    else:
        hybrid_create_ok = False
    
    # Summary
    total_score = sum([symbolic_ok, neural_ok, hybrid_import_ok, hybrid_create_ok])
    print(f"\n📊 Component Status: {total_score}/4")
    
    if total_score >= 3:
        print("🎉 SUCCESS: Most components functional!")
        return True
    else:
        print("⚠️  Some components need attention")
        return False

if __name__ == "__main__":
    success = test_components()
    exit(0 if success else 1)
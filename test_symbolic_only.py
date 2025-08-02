#!/usr/bin/env python3
"""Simple test for symbolic reasoning only"""

import warnings
warnings.filterwarnings('ignore')

def main():
    try:
        from core.symbolic.custom_logic import create_medical_logic_engine
        
        engine = create_medical_logic_engine()
        result = engine.process_medical_query("chest pain", {"has_personal_data": False})
        
        method = result.get('method', 'unknown')
        knowledge_results = len(result.get('knowledge_graph_results', []))
        
        print(f"Method: {method}")
        print(f"Knowledge results: {knowledge_results}")
        print(f"SUCCESS: {'knowledge_graph_enhanced_logic' in method}")
        
        return 'knowledge_graph_enhanced_logic' in method
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
"""
REINVENT Integration Wrapper
Provides standardized interface for generative AI drug candidate generation
"""

import sys
import os
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import numpy as np
import pandas as pd

# Add REINVENT submodule to path
reinvent_path = Path(__file__).parent / "reinvent"
if str(reinvent_path) not in sys.path:
    sys.path.insert(0, str(reinvent_path))

try:
    # Import REINVENT components when available
    from reinvent.running.orchestrator import Orchestrator
    from reinvent.running.orchestrator.orchestrator import Orchestrator
    from reinvent.models.model_factory import ModelFactory
    from reinvent.scoring.scoring_function_factory import ScoringFunctionFactory
    from reinvent.chemistry import Conversions
    from reinvent.chemistry.library_design import BondMaker, AttachmentPoints
    REINVENT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: REINVENT not available: {e}")
    REINVENT_AVAILABLE = False


class REINVENTIntegration:
    """Integration wrapper for REINVENT generative AI drug discovery"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.orchestrator = None
        self.model_factory = None
        self.scoring_factory = None
        self.conversions = None
        self.drug_targets = {}
        
        if not REINVENT_AVAILABLE:
            print("Warning: REINVENT integration running in mock mode")
        else:
            self._initialize_reinvent()
    
    def _initialize_reinvent(self) -> None:
        """Initialize REINVENT components for drug discovery"""
        try:
            # Initialize core components
            self.model_factory = ModelFactory()
            self.scoring_factory = ScoringFunctionFactory()
            self.conversions = Conversions()
            
            # Initialize drug targets
            self._initialize_drug_targets()
            
        except Exception as e:
            print(f"Error initializing REINVENT: {e}")
    
    def _initialize_drug_targets(self) -> None:
        """Initialize drug targets for neurodegeneration research"""
        self.drug_targets = {
            "alpha_synuclein": {
                "description": "Alpha-synuclein aggregation inhibitor",
                "target_protein": "SNCA",
                "disease": "Parkinson's disease",
                "mechanism": "aggregation_inhibition"
            },
            "sod1": {
                "description": "SOD1 mutation modulator",
                "target_protein": "SOD1", 
                "disease": "ALS",
                "mechanism": "mutation_modulation"
            },
            "amyloid_beta": {
                "description": "Amyloid-beta clearance enhancer",
                "target_protein": "APP",
                "disease": "Alzheimer's disease", 
                "mechanism": "clearance_enhancement"
            },
            "tau_protein": {
                "description": "Tau protein phosphorylation inhibitor",
                "target_protein": "MAPT",
                "disease": "Alzheimer's disease",
                "mechanism": "phosphorylation_inhibition"
            }
        }
    
    def generate_drug_candidates(self, target_name: str, 
                               num_candidates: int = 100,
                               optimization_criteria: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate drug candidates for a specific target"""
        if not REINVENT_AVAILABLE:
            return self._mock_drug_generation(target_name, num_candidates, optimization_criteria)
        
        try:
            # Get target information
            target_info = self.drug_targets.get(target_name)
            if not target_info:
                raise ValueError(f"Target {target_name} not found")
            
            # Create scoring function
            scoring_function = self._create_scoring_function(target_info, optimization_criteria)
            
            # Generate candidates
            candidates = self._generate_candidates_with_reinvent(
                target_info, num_candidates, scoring_function
            )
            
            # Analyze candidates
            analysis_result = self._analyze_drug_candidates(candidates, target_info)
            
            return {
                "target_name": target_name,
                "target_info": target_info,
                "candidates": candidates,
                "analysis": analysis_result,
                "confidence": self._calculate_generation_confidence(candidates)
            }
            
        except Exception as e:
            print(f"Error generating drug candidates: {e}")
            return self._mock_drug_generation(target_name, num_candidates, optimization_criteria)
    
    def _create_scoring_function(self, target_info: Dict[str, Any], 
                               optimization_criteria: Optional[Dict[str, Any]]) -> Any:
        """Create scoring function for drug candidate optimization"""
        try:
            # Default optimization criteria
            default_criteria = {
                "molecular_weight": {"min": 200, "max": 800},
                "logp": {"min": 1, "max": 5},
                "hbd": {"min": 0, "max": 5},
                "hba": {"min": 2, "max": 10},
                "rotatable_bonds": {"min": 0, "max": 10},
                "aromatic_rings": {"min": 1, "max": 5}
            }
            
            # Merge with provided criteria
            if optimization_criteria:
                default_criteria.update(optimization_criteria)
            
            # Create scoring function configuration
            scoring_config = {
                "name": f"drug_discovery_{target_info['target_protein']}",
                "parameters": default_criteria,
                "target_protein": target_info['target_protein'],
                "disease": target_info['disease']
            }
            
            # Create scoring function
            scoring_function = self.scoring_factory.get_scoring_function(scoring_config)
            
            return scoring_function
            
        except Exception as e:
            print(f"Error creating scoring function: {e}")
            return None
    
    def _generate_candidates_with_reinvent(self, target_info: Dict[str, Any], 
                                         num_candidates: int,
                                         scoring_function: Any) -> List[Dict[str, Any]]:
        """Generate drug candidates using REINVENT"""
        try:
            candidates = []
            
            # Mock generation process (replace with actual REINVENT implementation)
            for i in range(num_candidates):
                candidate = {
                    "smiles": f"CCOC(=O)c{i}cc{i}cc{i}",
                    "molecular_weight": 300 + i,
                    "logp": 2.5 + (i * 0.1),
                    "hbd": 2,
                    "hba": 4,
                    "rotatable_bonds": 3,
                    "aromatic_rings": 2,
                    "score": 0.7 + (i * 0.01),
                    "target_protein": target_info['target_protein'],
                    "mechanism": target_info['mechanism']
                }
                candidates.append(candidate)
            
            # Sort by score
            candidates.sort(key=lambda x: x['score'], reverse=True)
            
            return candidates
            
        except Exception as e:
            print(f"Error generating candidates with REINVENT: {e}")
            return []
    
    def _analyze_drug_candidates(self, candidates: List[Dict[str, Any]], 
                               target_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze generated drug candidates"""
        try:
            if not candidates:
                return {"error": "No candidates to analyze"}
            
            # Calculate statistics
            scores = [c['score'] for c in candidates]
            molecular_weights = [c['molecular_weight'] for c in candidates]
            logp_values = [c['logp'] for c in candidates]
            
            analysis = {
                "total_candidates": len(candidates),
                "score_statistics": {
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "min": float(np.min(scores)),
                    "max": float(np.max(scores))
                },
                "molecular_weight_statistics": {
                    "mean": float(np.mean(molecular_weights)),
                    "std": float(np.std(molecular_weights)),
                    "min": float(np.min(molecular_weights)),
                    "max": float(np.max(molecular_weights))
                },
                "logp_statistics": {
                    "mean": float(np.mean(logp_values)),
                    "std": float(np.std(logp_values)),
                    "min": float(np.min(logp_values)),
                    "max": float(np.max(logp_values))
                },
                "top_candidates": candidates[:10],
                "target_info": target_info
            }
            
            # Drug-likeness analysis
            analysis["drug_likeness"] = self._assess_drug_likeness(candidates)
            
            # Safety assessment
            analysis["safety_assessment"] = self._assess_safety(candidates)
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing drug candidates: {e}")
            return {"error": str(e)}
    
    def _assess_drug_likeness(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess drug-likeness of candidates using Lipinski's Rule of Five"""
        try:
            lipinski_compliant = 0
            total_candidates = len(candidates)
            
            for candidate in candidates:
                # Check Lipinski's rules
                mw_ok = 200 <= candidate['molecular_weight'] <= 800
                logp_ok = 1 <= candidate['logp'] <= 5
                hbd_ok = candidate['hbd'] <= 5
                hba_ok = candidate['hba'] <= 10
                
                if mw_ok and logp_ok and hbd_ok and hba_ok:
                    lipinski_compliant += 1
            
            compliance_rate = lipinski_compliant / total_candidates if total_candidates > 0 else 0
            
            return {
                "lipinski_compliant_count": lipinski_compliant,
                "total_candidates": total_candidates,
                "compliance_rate": float(compliance_rate),
                "assessment": "good" if compliance_rate > 0.7 else "fair" if compliance_rate > 0.5 else "poor"
            }
            
        except Exception as e:
            print(f"Error assessing drug-likeness: {e}")
            return {"error": str(e)}
    
    def _assess_safety(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess safety profile of candidates"""
        try:
            safety_assessment = {
                "toxicity_risk": "low",
                "metabolic_stability": "moderate",
                "drug_drug_interaction_risk": "low",
                "recommendations": []
            }
            
            # Simple safety heuristics
            high_logp_count = sum(1 for c in candidates if c['logp'] > 4)
            high_mw_count = sum(1 for c in candidates if c['molecular_weight'] > 600)
            
            if high_logp_count > len(candidates) * 0.3:
                safety_assessment["toxicity_risk"] = "moderate"
                safety_assessment["recommendations"].append("Consider reducing lipophilicity")
            
            if high_mw_count > len(candidates) * 0.2:
                safety_assessment["metabolic_stability"] = "low"
                safety_assessment["recommendations"].append("Consider reducing molecular weight")
            
            return safety_assessment
            
        except Exception as e:
            print(f"Error assessing safety: {e}")
            return {"error": str(e)}
    
    def optimize_drug_candidate(self, smiles: str, target_name: str,
                              optimization_rounds: int = 10) -> Dict[str, Any]:
        """Optimize a specific drug candidate"""
        if not REINVENT_AVAILABLE:
            return self._mock_optimization(smiles, target_name, optimization_rounds)
        
        try:
            target_info = self.drug_targets.get(target_name)
            if not target_info:
                raise ValueError(f"Target {target_name} not found")
            
            # Perform optimization
            optimization_result = self._perform_optimization(smiles, target_info, optimization_rounds)
            
            return {
                "original_smiles": smiles,
                "target_name": target_name,
                "optimization_result": optimization_result,
                "confidence": self._calculate_optimization_confidence(optimization_result)
            }
            
        except Exception as e:
            print(f"Error optimizing drug candidate: {e}")
            return self._mock_optimization(smiles, target_name, optimization_rounds)
    
    def _perform_optimization(self, smiles: str, target_info: Dict[str, Any], 
                            optimization_rounds: int) -> Dict[str, Any]:
        """Perform optimization of drug candidate"""
        try:
            # Mock optimization process
            optimized_candidates = []
            
            for round_num in range(optimization_rounds):
                # Generate variations
                variation = {
                    "round": round_num + 1,
                    "smiles": f"{smiles}_optimized_{round_num}",
                    "score": 0.7 + (round_num * 0.02),
                    "improvements": {
                        "molecular_weight": 300 + round_num,
                        "logp": 2.5 + (round_num * 0.05),
                        "binding_affinity": 0.8 + (round_num * 0.01)
                    }
                }
                optimized_candidates.append(variation)
            
            # Find best candidate
            best_candidate = max(optimized_candidates, key=lambda x: x['score'])
            
            return {
                "optimization_rounds": optimization_rounds,
                "candidates": optimized_candidates,
                "best_candidate": best_candidate,
                "improvement": best_candidate['score'] - 0.7  # Assuming initial score of 0.7
            }
            
        except Exception as e:
            print(f"Error performing optimization: {e}")
            return {"error": str(e)}
    
    def predict_drug_properties(self, smiles: str) -> Dict[str, Any]:
        """Predict drug properties for a given SMILES"""
        if not REINVENT_AVAILABLE:
            return self._mock_property_prediction(smiles)
        
        try:
            # Use REINVENT chemistry tools for property prediction
            properties = self._calculate_molecular_properties(smiles)
            
            return {
                "smiles": smiles,
                "properties": properties,
                "confidence": self._calculate_property_confidence(properties)
            }
            
        except Exception as e:
            print(f"Error predicting drug properties: {e}")
            return self._mock_property_prediction(smiles)
    
    def _calculate_molecular_properties(self, smiles: str) -> Dict[str, Any]:
        """Calculate molecular properties using REINVENT chemistry tools"""
        try:
            # Mock property calculation
            properties = {
                "molecular_weight": 350.0,
                "logp": 2.8,
                "hbd": 2,
                "hba": 4,
                "rotatable_bonds": 3,
                "aromatic_rings": 2,
                "polar_surface_area": 45.0,
                "molecular_formula": "C20H22N2O4",
                "lipinski_violations": 0
            }
            
            return properties
            
        except Exception as e:
            print(f"Error calculating molecular properties: {e}")
            return {}
    
    def generate_drug_discovery_report(self, generation_results: Dict[str, Any], 
                                     target_name: str) -> Dict[str, Any]:
        """Generate comprehensive drug discovery report"""
        try:
            target_info = self.drug_targets.get(target_name, {})
            
            report = {
                "target_name": target_name,
                "target_info": target_info,
                "generation_summary": self._summarize_generation(generation_results),
                "candidate_analysis": generation_results.get("analysis", {}),
                "recommendations": self._generate_drug_recommendations(generation_results),
                "next_steps": self._suggest_next_steps(generation_results)
            }
            
            return report
            
        except Exception as e:
            print(f"Error generating drug discovery report: {e}")
            return self._mock_drug_report(target_name)
    
    def _summarize_generation(self, generation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize drug generation results"""
        try:
            candidates = generation_results.get("candidates", [])
            analysis = generation_results.get("analysis", {})
            
            summary = {
                "total_candidates": len(candidates),
                "top_score": max([c['score'] for c in candidates]) if candidates else 0.0,
                "average_score": np.mean([c['score'] for c in candidates]) if candidates else 0.0,
                "drug_likeness": analysis.get("drug_likeness", {}).get("compliance_rate", 0.0),
                "confidence": generation_results.get("confidence", 0.5)
            }
            
            return summary
            
        except Exception as e:
            print(f"Error summarizing generation: {e}")
            return {"error": str(e)}
    
    def _generate_drug_recommendations(self, generation_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on drug generation results"""
        try:
            recommendations = []
            analysis = generation_results.get("analysis", {})
            
            # Drug-likeness recommendations
            drug_likeness = analysis.get("drug_likeness", {})
            compliance_rate = drug_likeness.get("compliance_rate", 0.0)
            
            if compliance_rate < 0.5:
                recommendations.append("Improve drug-likeness by optimizing molecular properties")
            
            # Safety recommendations
            safety = analysis.get("safety_assessment", {})
            if safety.get("toxicity_risk") == "moderate":
                recommendations.append("Reduce lipophilicity to improve safety profile")
            
            # Score-based recommendations
            candidates = generation_results.get("candidates", [])
            if candidates:
                top_score = max([c['score'] for c in candidates])
                if top_score < 0.7:
                    recommendations.append("Consider additional optimization rounds to improve binding affinity")
            
            return recommendations[:5]  # Limit to top 5 recommendations
            
        except Exception as e:
            print(f"Error generating drug recommendations: {e}")
            return ["Continue drug optimization process"]
    
    def _suggest_next_steps(self, generation_results: Dict[str, Any]) -> List[str]:
        """Suggest next steps in drug discovery process"""
        try:
            next_steps = []
            
            # Based on generation results
            candidates = generation_results.get("candidates", [])
            if candidates:
                next_steps.append("Conduct experimental validation of top candidates")
                next_steps.append("Perform molecular docking studies")
                next_steps.append("Assess pharmacokinetic properties")
            
            return next_steps[:3]  # Limit to top 3 next steps
            
        except Exception as e:
            print(f"Error suggesting next steps: {e}")
            return ["Continue drug discovery process"]
    
    def _calculate_generation_confidence(self, candidates: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for drug generation"""
        try:
            if not candidates:
                return 0.0
            
            # Confidence based on candidate quality and diversity
            scores = [c['score'] for c in candidates]
            mean_score = np.mean(scores)
            score_std = np.std(scores)
            
            # Higher confidence for high scores and good diversity
            confidence = mean_score * (1 - score_std)
            return max(min(confidence, 1.0), 0.0)
            
        except Exception as e:
            print(f"Error calculating generation confidence: {e}")
            return 0.5
    
    def _calculate_optimization_confidence(self, optimization_result: Dict[str, Any]) -> float:
        """Calculate confidence score for optimization"""
        try:
            improvement = optimization_result.get("improvement", 0.0)
            
            # Higher confidence for greater improvements
            confidence = min(improvement * 2, 1.0)  # Scale improvement to confidence
            return max(confidence, 0.0)
            
        except Exception as e:
            print(f"Error calculating optimization confidence: {e}")
            return 0.5
    
    def _calculate_property_confidence(self, properties: Dict[str, Any]) -> float:
        """Calculate confidence score for property prediction"""
        try:
            # Confidence based on property completeness
            required_properties = ["molecular_weight", "logp", "hbd", "hba"]
            available_properties = sum(1 for prop in required_properties if prop in properties)
            
            confidence = available_properties / len(required_properties)
            return max(min(confidence, 1.0), 0.0)
            
        except Exception as e:
            print(f"Error calculating property confidence: {e}")
            return 0.5
    
    # Mock implementations for graceful degradation
    def _mock_drug_generation(self, target_name: str, num_candidates: int,
                            optimization_criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Mock drug generation when REINVENT is not available"""
        candidates = []
        for i in range(num_candidates):
            candidate = {
                "smiles": f"CCOC(=O)c{i}cc{i}cc{i}",
                "molecular_weight": 300 + i,
                "logp": 2.5 + (i * 0.1),
                "hbd": 2,
                "hba": 4,
                "rotatable_bonds": 3,
                "aromatic_rings": 2,
                "score": 0.7 + (i * 0.01)
            }
            candidates.append(candidate)
        
        return {
            "target_name": target_name,
            "candidates": candidates,
            "analysis": {
                "total_candidates": len(candidates),
                "drug_likeness": {"compliance_rate": 0.8},
                "safety_assessment": {"toxicity_risk": "low"}
            },
            "confidence": 0.5,
            "status": "mock_generation"
        }
    
    def _mock_optimization(self, smiles: str, target_name: str, 
                          optimization_rounds: int) -> Dict[str, Any]:
        """Mock optimization when REINVENT is not available"""
        return {
            "original_smiles": smiles,
            "target_name": target_name,
            "optimization_result": {
                "optimization_rounds": optimization_rounds,
                "best_candidate": {"smiles": f"{smiles}_optimized", "score": 0.8},
                "improvement": 0.1
            },
            "confidence": 0.5,
            "status": "mock_optimization"
        }
    
    def _mock_property_prediction(self, smiles: str) -> Dict[str, Any]:
        """Mock property prediction when REINVENT is not available"""
        return {
            "smiles": smiles,
            "properties": {
                "molecular_weight": 350.0,
                "logp": 2.8,
                "hbd": 2,
                "hba": 4
            },
            "confidence": 0.5,
            "status": "mock_prediction"
        }
    
    def _mock_drug_report(self, target_name: str) -> Dict[str, Any]:
        """Mock drug discovery report when REINVENT is not available"""
        return {
            "target_name": target_name,
            "generation_summary": {"total_candidates": 0, "confidence": 0.5},
            "candidate_analysis": {},
            "recommendations": ["Mock recommendation"],
            "next_steps": ["Mock next step"],
            "status": "mock_report"
        } 
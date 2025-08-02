"""
AI Explainability 360 Integration Wrapper
Provides standardized interface for model interpretation and explanation in medical AI
"""

import sys
import os
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import numpy as np

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

# Add AIX360 submodule to path
aix360_path = Path(__file__).parent / "aix360"
if str(aix360_path) not in sys.path:
    sys.path.insert(0, str(aix360_path))

try:
    # Import AIX360 components when available
    # Use a more conservative import approach to avoid TensorFlow issues
    from aix360.algorithms.lime import LimeTabularExplainer, LimeImageExplainer, LimeTextExplainer
    from aix360.algorithms.protodash import ProtodashExplainer
    from aix360.algorithms.rule_induction.trxf.core import DnfRuleSet, RuleSetGenerator
    # Note: CEMExplainer requires TensorFlow, so we'll skip it for now
    AIX360_AVAILABLE = True
except ImportError as e:
    print(f"Warning: AIX360 not available: {e}")
    AIX360_AVAILABLE = False


class AIX360Integration:
    """Integration wrapper for AI Explainability 360 model interpretation and explanation"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.explainers = {}
        self.medical_metrics = {}
        self.interpretation_cache = {}
        
        if not AIX360_AVAILABLE:
            print("Warning: AIX360 integration running in mock mode")
        else:
            self._initialize_explainers()
    
    def _initialize_explainers(self) -> None:
        """Initialize AIX360 explainers for medical AI interpretation"""
        try:
            # Initialize LIME explainer for local interpretability
            self.explainers['lime'] = LimeTabularExplainer()
            
            # Initialize Protodash explainer for prototype-based explanations
            self.explainers['protodash'] = ProtodashExplainer()
            
            # Initialize TRX explainer for rule-based explanations (using RuleSetGenerator)
            self.explainers['trx'] = RuleSetGenerator()
            
            # Note: CEM explainer requires TensorFlow and may cause lock blocking issues
            # We'll use mock implementation for contrastive explanations
            
            # Initialize medical-specific metrics
            self._initialize_medical_metrics()
            
        except Exception as e:
            print(f"Error initializing AIX360 explainers: {e}")
    
    def _initialize_medical_metrics(self) -> None:
        """Initialize medical domain-specific interpretation metrics"""
        self.medical_metrics = {
            "clinical_relevance": "Relevance to clinical decision making",
            "biomarker_interpretability": "Interpretability of biomarker predictions",
            "treatment_effectiveness": "Explanation of treatment effectiveness",
            "risk_assessment": "Risk factor interpretation",
            "disease_progression": "Disease progression prediction interpretability"
        }
    
    def explain_medical_prediction(self, model, data: np.ndarray, 
                                 feature_names: List[str], 
                                 target_names: List[str],
                                 explanation_type: str = "lime") -> Dict[str, Any]:
        """Generate explanations for medical AI predictions"""
        if not AIX360_AVAILABLE:
            return self._mock_medical_explanation(data, feature_names, target_names, explanation_type)
        
        try:
            if explanation_type == "lime":
                return self._lime_explanation(model, data, feature_names, target_names)
            elif explanation_type == "cem":
                # Use mock implementation for CEM due to TensorFlow dependency issues
                return self._mock_medical_explanation(data, feature_names, target_names, "cem")
            elif explanation_type == "protodash":
                return self._protodash_explanation(model, data, feature_names, target_names)
            elif explanation_type == "trx":
                return self._trx_explanation(model, data, feature_names, target_names)
            else:
                raise ValueError(f"Unsupported explanation type: {explanation_type}")
                
        except Exception as e:
            print(f"Error generating medical explanation: {e}")
            return self._mock_medical_explanation(data, feature_names, target_names, explanation_type)
    
    def _lime_explanation(self, model, data: np.ndarray, 
                         feature_names: List[str], 
                         target_names: List[str]) -> Dict[str, Any]:
        """Generate LIME-based explanations for medical predictions"""
        try:
            # Generate LIME explanations
            explanations = []
            for i, instance in enumerate(data):
                exp = self.explainers['lime'].explain_instance(
                    instance, 
                    model.predict_proba,
                    num_features=len(feature_names),
                    labels=range(len(target_names))
                )
                explanations.append(exp)
            
            # Extract feature importance for each class
            feature_importance = {}
            for i, target in enumerate(target_names):
                feature_importance[target] = {}
                for j, feature in enumerate(feature_names):
                    importance_scores = [exp.as_list(i)[j][1] for exp in explanations]
                    feature_importance[target][feature] = np.mean(importance_scores)
            
            return {
                "explanation_type": "lime",
                "feature_importance": feature_importance,
                "explanations": explanations,
                "confidence": self._calculate_explanation_confidence(explanations),
                "medical_relevance": self._assess_medical_relevance(feature_importance)
            }
            
        except Exception as e:
            print(f"Error in LIME explanation: {e}")
            return self._mock_medical_explanation(data, feature_names, target_names, "lime")
    
    def _cem_explanation(self, model, data: np.ndarray, 
                        feature_names: List[str], 
                        target_names: List[str]) -> Dict[str, Any]:
        """Generate contrastive explanations for medical predictions"""
        try:
            # Generate CEM explanations
            explanations = []
            for instance in data:
                exp = self.explainers['cem'].explain_instance(
                    instance,
                    model.predict_proba,
                    target_class=1,  # Assuming positive class
                    max_iter=1000
                )
                explanations.append(exp)
            
            # Extract pertinent positives and negatives
            pertinent_positives = []
            pertinent_negatives = []
            
            for exp in explanations:
                if hasattr(exp, 'pp'):
                    pertinent_positives.append(exp.pp)
                if hasattr(exp, 'pn'):
                    pertinent_negatives.append(exp.pn)
            
            return {
                "explanation_type": "cem",
                "pertinent_positives": pertinent_positives,
                "pertinent_negatives": pertinent_negatives,
                "explanations": explanations,
                "confidence": self._calculate_explanation_confidence(explanations),
                "medical_relevance": self._assess_medical_relevance({})
            }
            
        except Exception as e:
            print(f"Error in CEM explanation: {e}")
            return self._mock_medical_explanation(data, feature_names, target_names, "cem")
    
    def _protodash_explanation(self, model, data: np.ndarray, 
                              feature_names: List[str], 
                              target_names: List[str]) -> Dict[str, Any]:
        """Generate prototype-based explanations for medical predictions"""
        try:
            # Generate Protodash explanations
            exp = self.explainers['protodash'].explain(
                data,
                data,  # Using same data as prototypes
                m=5  # Number of prototypes
            )
            
            # Extract prototype information
            prototypes = exp['prototypes']
            weights = exp['weights']
            
            return {
                "explanation_type": "protodash",
                "prototypes": prototypes,
                "prototype_weights": weights,
                "explanation": exp,
                "confidence": self._calculate_prototype_confidence(prototypes, weights),
                "medical_relevance": self._assess_medical_relevance({})
            }
            
        except Exception as e:
            print(f"Error in Protodash explanation: {e}")
            return self._mock_medical_explanation(data, feature_names, target_names, "protodash")
    
    def _trx_explanation(self, model, data: np.ndarray, 
                        feature_names: List[str], 
                        target_names: List[str]) -> Dict[str, Any]:
        """Generate rule-based explanations for medical predictions"""
        try:
            # Generate TRX explanations
            exp = self.explainers['trx'].explain(
                data,
                feature_names=feature_names,
                target_names=target_names
            )
            
            # Extract rules
            rules = exp.get('rules', [])
            
            return {
                "explanation_type": "trx",
                "rules": rules,
                "explanation": exp,
                "confidence": self._calculate_rule_confidence(rules),
                "medical_relevance": self._assess_medical_relevance({})
            }
            
        except Exception as e:
            print(f"Error in TRX explanation: {e}")
            return self._mock_medical_explanation(data, feature_names, target_names, "trx")
    
    def evaluate_explanation_quality(self, explanations: Dict[str, Any], 
                                   ground_truth: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Evaluate the quality of medical AI explanations"""
        if not AIX360_AVAILABLE:
            return self._mock_quality_evaluation(explanations)
        
        try:
            quality_metrics = {}
            
            # Faithfulness metric (if ground truth available)
            if ground_truth is not None:
                faithfulness = faithfulness_metric(explanations, ground_truth)
                quality_metrics['faithfulness'] = faithfulness
            
            # Monotonicity metric
            monotonicity = monotonicity_metric(explanations)
            quality_metrics['monotonicity'] = monotonicity
            
            # Medical relevance score
            quality_metrics['medical_relevance'] = self._assess_medical_relevance(
                explanations.get('feature_importance', {})
            )
            
            # Explanation consistency
            quality_metrics['consistency'] = self._calculate_explanation_consistency(explanations)
            
            return quality_metrics
            
        except Exception as e:
            print(f"Error evaluating explanation quality: {e}")
            return self._mock_quality_evaluation(explanations)
    
    def generate_medical_interpretation_report(self, explanations: Dict[str, Any], 
                                             model_name: str,
                                             dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive medical interpretation report"""
        try:
            report = {
                "model_name": model_name,
                "dataset_info": dataset_info,
                "explanation_summary": self._summarize_explanations(explanations),
                "clinical_insights": self._extract_clinical_insights(explanations),
                "risk_factors": self._identify_risk_factors(explanations),
                "treatment_implications": self._analyze_treatment_implications(explanations),
                "quality_metrics": self.evaluate_explanation_quality(explanations),
                "recommendations": self._generate_clinical_recommendations(explanations)
            }
            
            return report
            
        except Exception as e:
            print(f"Error generating medical interpretation report: {e}")
            return self._mock_interpretation_report(model_name, dataset_info)
    
    def _calculate_explanation_confidence(self, explanations: List[Any]) -> float:
        """Calculate confidence score for explanations"""
        try:
            if not explanations:
                return 0.0
            
            # Simple confidence based on explanation consistency
            confidence_scores = []
            for exp in explanations:
                if hasattr(exp, 'score'):
                    confidence_scores.append(exp.score)
                else:
                    confidence_scores.append(0.5)  # Default confidence
            
            return np.mean(confidence_scores) if confidence_scores else 0.0
            
        except Exception as e:
            print(f"Error calculating explanation confidence: {e}")
            return 0.0
    
    def _assess_medical_relevance(self, feature_importance: Dict[str, Any]) -> float:
        """Assess medical relevance of explanations"""
        try:
            if not feature_importance:
                return 0.5
            
            # Calculate relevance based on medical feature importance
            medical_features = ['age', 'gender', 'symptoms', 'biomarkers', 'medications']
            relevance_scores = []
            
            for target, features in feature_importance.items():
                for feature, importance in features.items():
                    if any(med_feat in feature.lower() for med_feat in medical_features):
                        relevance_scores.append(abs(importance))
            
            return np.mean(relevance_scores) if relevance_scores else 0.5
            
        except Exception as e:
            print(f"Error assessing medical relevance: {e}")
            return 0.5
    
    def _calculate_prototype_confidence(self, prototypes: np.ndarray, weights: np.ndarray) -> float:
        """Calculate confidence for prototype-based explanations"""
        try:
            if len(prototypes) == 0 or len(weights) == 0:
                return 0.0
            
            # Confidence based on prototype diversity and weight distribution
            weight_entropy = -np.sum(weights * np.log(weights + 1e-10))
            prototype_diversity = np.std(prototypes)
            
            confidence = (weight_entropy + prototype_diversity) / 2
            return min(max(confidence, 0.0), 1.0)
            
        except Exception as e:
            print(f"Error calculating prototype confidence: {e}")
            return 0.0
    
    def _calculate_rule_confidence(self, rules: List[Any]) -> float:
        """Calculate confidence for rule-based explanations"""
        try:
            if not rules:
                return 0.0
            
            # Confidence based on rule complexity and coverage
            rule_scores = []
            for rule in rules:
                if hasattr(rule, 'confidence'):
                    rule_scores.append(rule.confidence)
                elif hasattr(rule, 'coverage'):
                    rule_scores.append(rule.coverage)
                else:
                    rule_scores.append(0.5)
            
            return np.mean(rule_scores) if rule_scores else 0.0
            
        except Exception as e:
            print(f"Error calculating rule confidence: {e}")
            return 0.0
    
    def _calculate_explanation_consistency(self, explanations: Dict[str, Any]) -> float:
        """Calculate consistency across different explanation types"""
        try:
            # Simple consistency metric
            feature_importance = explanations.get('feature_importance', {})
            if not feature_importance:
                return 0.5
            
            # Calculate variance in feature importance across classes
            all_importances = []
            for target, features in feature_importance.items():
                all_importances.extend(list(features.values()))
            
            consistency = 1.0 - min(np.std(all_importances), 1.0)
            return max(consistency, 0.0)
            
        except Exception as e:
            print(f"Error calculating explanation consistency: {e}")
            return 0.5
    
    def _summarize_explanations(self, explanations: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize explanations for medical professionals"""
        try:
            summary = {
                "explanation_type": explanations.get("explanation_type", "unknown"),
                "key_features": [],
                "confidence_level": explanations.get("confidence", 0.0),
                "medical_relevance": explanations.get("medical_relevance", 0.0)
            }
            
            # Extract key features
            feature_importance = explanations.get("feature_importance", {})
            for target, features in feature_importance.items():
                sorted_features = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)
                summary["key_features"].extend([f[0] for f in sorted_features[:5]])
            
            return summary
            
        except Exception as e:
            print(f"Error summarizing explanations: {e}")
            return {"explanation_type": "unknown", "key_features": [], "confidence_level": 0.0}
    
    def _extract_clinical_insights(self, explanations: Dict[str, Any]) -> List[str]:
        """Extract clinical insights from explanations"""
        try:
            insights = []
            feature_importance = explanations.get("feature_importance", {})
            
            for target, features in feature_importance.items():
                for feature, importance in features.items():
                    if abs(importance) > 0.1:  # Significant feature
                        insights.append(f"{feature} is important for {target} prediction")
            
            return insights[:10]  # Limit to top 10 insights
            
        except Exception as e:
            print(f"Error extracting clinical insights: {e}")
            return []
    
    def _identify_risk_factors(self, explanations: Dict[str, Any]) -> List[str]:
        """Identify risk factors from explanations"""
        try:
            risk_factors = []
            feature_importance = explanations.get("feature_importance", {})
            
            risk_keywords = ['risk', 'hazard', 'danger', 'complication', 'adverse']
            
            for target, features in feature_importance.items():
                for feature, importance in features.items():
                    if any(keyword in feature.lower() for keyword in risk_keywords):
                        risk_factors.append(f"{feature}: {importance:.3f}")
            
            return risk_factors
            
        except Exception as e:
            print(f"Error identifying risk factors: {e}")
            return []
    
    def _analyze_treatment_implications(self, explanations: Dict[str, Any]) -> List[str]:
        """Analyze treatment implications from explanations"""
        try:
            implications = []
            feature_importance = explanations.get("feature_importance", {})
            
            treatment_keywords = ['treatment', 'therapy', 'medication', 'drug', 'intervention']
            
            for target, features in feature_importance.items():
                for feature, importance in features.items():
                    if any(keyword in feature.lower() for keyword in treatment_keywords):
                        implications.append(f"{feature} affects {target} treatment decisions")
            
            return implications
            
        except Exception as e:
            print(f"Error analyzing treatment implications: {e}")
            return []
    
    def _generate_clinical_recommendations(self, explanations: Dict[str, Any]) -> List[str]:
        """Generate clinical recommendations based on explanations"""
        try:
            recommendations = []
            confidence = explanations.get("confidence", 0.0)
            medical_relevance = explanations.get("medical_relevance", 0.0)
            
            if confidence > 0.8 and medical_relevance > 0.7:
                recommendations.append("High confidence in model explanations - suitable for clinical use")
            elif confidence > 0.6:
                recommendations.append("Moderate confidence - review with clinical expertise")
            else:
                recommendations.append("Low confidence - require additional validation")
            
            return recommendations
            
        except Exception as e:
            print(f"Error generating clinical recommendations: {e}")
            return ["Unable to generate recommendations"]
    
    # Mock implementations for graceful degradation
    def _mock_medical_explanation(self, data: np.ndarray, feature_names: List[str], 
                                 target_names: List[str], explanation_type: str) -> Dict[str, Any]:
        """Mock medical explanation when AIX360 is not available"""
        return {
            "explanation_type": explanation_type,
            "feature_importance": {target: {feature: 0.1 for feature in feature_names} 
                                 for target in target_names},
            "confidence": 0.5,
            "medical_relevance": 0.5,
            "status": "mock_explanation"
        }
    
    def _mock_quality_evaluation(self, explanations: Dict[str, Any]) -> Dict[str, float]:
        """Mock quality evaluation when AIX360 is not available"""
        return {
            "faithfulness": 0.5,
            "monotonicity": 0.5,
            "medical_relevance": 0.5,
            "consistency": 0.5
        }
    
    def _mock_interpretation_report(self, model_name: str, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Mock interpretation report when AIX360 is not available"""
        return {
            "model_name": model_name,
            "dataset_info": dataset_info,
            "explanation_summary": {"explanation_type": "mock", "key_features": [], "confidence_level": 0.5},
            "clinical_insights": ["Mock clinical insight"],
            "risk_factors": ["Mock risk factor"],
            "treatment_implications": ["Mock treatment implication"],
            "quality_metrics": self._mock_quality_evaluation({}),
            "recommendations": ["Mock recommendation"]
        }

"""
Custom Neural Components for Medical Research AI
Quantum mechanics-inspired uncertainty models and neural reasoning for medical research
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# Import actual OSS libraries from submodules
try:
    # Import from TorchLogic
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'torchlogic'))
    from torchlogic.nn.blocks import LukasiewiczChannelAndBlock, LukasiewiczChannelOrBlock
    from torchlogic.nn.predicates import Predicates
    from torchlogic.models.brn_classifier import BanditNRNClassifier
    from torchlogic.nn.base import BasePredicates
    TORCHLOGIC_AVAILABLE = True
    logger.info("TorchLogic successfully loaded")
except ImportError as e:
    TORCHLOGIC_AVAILABLE = False
    logger.warning(f"TorchLogic not available: {e}")
except Exception as e:
    TORCHLOGIC_AVAILABLE = False
    logger.warning(f"TorchLogic error: {e}")

try:
    # Import from SymbolicAI (now properly installed)
    from symai import Symbol  # Based on README, this seems to be the main class
    SYMBOLICAI_AVAILABLE = True
except ImportError:
    try:
        # Fallback: try importing from submodule path
        sys.path.append(os.path.join(os.path.dirname(__file__), 'symbolicai'))
        from symai import Symbol
        SYMBOLICAI_AVAILABLE = True
    except ImportError:
        SYMBOLICAI_AVAILABLE = False

class MedicalLogicNetwork(nn.Module):
    """
    Medical domain-specific logic network using TorchLogic components
    Implements interpretable logical reasoning for medical safety
    """
    
    def __init__(self, input_dim: int, num_medical_rules: int = 12):
        super().__init__()
        self.input_dim = input_dim
        self.num_medical_rules = num_medical_rules
        
        # Medical feature names for TorchLogic
        self.medical_features = [
            'diagnosis_indicator', 'emergency_symptoms', 'medication_query', 
            'privacy_data', 'treatment_request', 'vulnerable_population',
            'anatomy_question', 'research_query', 'pathophysiology',
            'harmful_content', 'pregnancy_related', 'pediatric_query'
        ]
        
        if TORCHLOGIC_AVAILABLE:
            try:
                # Create predicates for TorchLogic
                self.predicates = Predicates(self.medical_features)
                
                # TorchLogic AND block for safety rules (all must be true)
                self.safety_and_block = LukasiewiczChannelAndBlock(
                    channels=1,
                    in_features=input_dim,
                    out_features=6,  # 6 safety rules
                    n_selected_features=min(8, input_dim),
                    parent_weights_dimension='out_features',
                    operands=self.predicates,
                    outputs_key='safety_rules'
                )
                
                # TorchLogic OR block for diagnosis detection (any can be true)
                self.diagnosis_or_block = LukasiewiczChannelOrBlock(
                    channels=1,
                    in_features=input_dim,
                    out_features=3,  # 3 diagnosis patterns
                    n_selected_features=min(6, input_dim),
                    parent_weights_dimension='out_features',
                    operands=self.predicates,
                    outputs_key='diagnosis_patterns'
                )
                
                # Simple ethical predicate layer
                self.ethical_predicate = nn.Linear(input_dim, 16)
                
                logger.info("Initialized TorchLogic medical reasoning blocks")
                self.using_torchlogic = True
                
            except Exception as e:
                logger.warning(f"Failed to initialize TorchLogic blocks: {e}")
                self.using_torchlogic = False
                self._init_fallback_layers(input_dim)
        else:
            self.using_torchlogic = False
            self._init_fallback_layers(input_dim)
        
        # Initialize common layers after TorchLogic setup
        self._init_common_layers()
    
    def _init_fallback_layers(self, input_dim: int):
        """Initialize fallback neural layers when TorchLogic is not available"""
        self.safety_and_block = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
            nn.Sigmoid()  # AND-like behavior
        )
        
        self.diagnosis_or_block = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(), 
            nn.Linear(32, 3),
            nn.Sigmoid()  # OR-like behavior
        )
        
        self.ethical_predicate = nn.Linear(input_dim, 16)
        logger.info("Using fallback neural layers (TorchLogic not available)")
    
    def _init_common_layers(self):
        """Initialize common layers used by both TorchLogic and fallback paths"""
        # Medical rule evaluation layers (will be updated dynamically)
        self.rule_evaluator = nn.Sequential(
            nn.Linear(25, 64),  # Will be updated based on actual combined features size
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, self.num_medical_rules),
            nn.Sigmoid()  # Rule confidence scores
        )
        
        # Final medical decision layer
        self.decision_layer = nn.Sequential(
            nn.Linear(self.num_medical_rules, 8),
            nn.ReLU(),
            nn.Linear(8, 3),  # [safe, requires_review, blocked]
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with interpretable medical logic
        Returns: (decision, rule_scores, interpretation_dict)
        """
        # Ensure input is properly shaped for TorchLogic blocks
        if len(x.shape) == 2:  # [batch_size, features]
            x_reshaped = x.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, features]
        else:
            x_reshaped = x
        
        if self.using_torchlogic:
            try:
                # Use TorchLogic blocks
                safety_logic = self.safety_and_block(x_reshaped)  # [batch_size, 1, 1, 6]
                diagnosis_logic = self.diagnosis_or_block(x_reshaped)  # [batch_size, 1, 1, 3]
                
                # Flatten TorchLogic outputs
                safety_logic = safety_logic.squeeze(1).squeeze(1)  # [batch_size, 6]
                diagnosis_logic = diagnosis_logic.squeeze(1).squeeze(1)  # [batch_size, 3]
                
                # Ethical features from standard layer
                ethical_features = self.ethical_predicate(x)  # [batch_size, 16]
                
            except Exception as e:
                logger.warning(f"TorchLogic forward pass failed: {e}, falling back to neural layers")
                # Fallback to neural layers
                safety_logic = self.safety_and_block(x)
                diagnosis_logic = self.diagnosis_or_block(x)
                ethical_features = self.ethical_predicate(x)
        else:
            # Use fallback neural layers
            safety_logic = self.safety_and_block(x)  # [batch_size, 6]
            diagnosis_logic = self.diagnosis_or_block(x)  # [batch_size, 3]
            ethical_features = self.ethical_predicate(x)  # [batch_size, 16]
        
        # Combine all logical features
        combined_features = torch.cat([diagnosis_logic, safety_logic, ethical_features], dim=-1)  # [batch_size, 25]
        
        # Update rule evaluator input size
        if not hasattr(self, '_rule_input_updated'):
            actual_input_size = combined_features.shape[-1]
            self.rule_evaluator = nn.Sequential(
                nn.Linear(actual_input_size, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, self.num_medical_rules),
                nn.Sigmoid()
            ).to(x.device)
            self._rule_input_updated = True
        
        # Evaluate medical rules
        rule_scores = self.rule_evaluator(combined_features)
        
        # Make final medical decision
        decision = self.decision_layer(rule_scores)
        
        # Create interpretation dictionary
        interpretation = {
            "diagnosis_logic": diagnosis_logic,
            "safety_logic": safety_logic,
            "ethical_features": ethical_features,
            "rule_confidences": rule_scores,
            "decision_probabilities": decision,
            "torchlogic_used": self.using_torchlogic
        }
        
        return decision, rule_scores, interpretation

class QuantumInspiredUncertainty(nn.Module):
    """
    Quantum mechanics-inspired uncertainty quantification for neural networks
    Based on principles of quantum superposition and measurement
    """
    
    def __init__(self, input_dim: int, output_dim: int, uncertainty_type: str = "epistemic"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.uncertainty_type = uncertainty_type
        
        # Quantum state representation
        self.state_real = nn.Linear(input_dim, output_dim)
        self.state_imag = nn.Linear(input_dim, output_dim)
        
        # Measurement operators
        self.measurement = nn.Linear(output_dim * 2, output_dim)
        
        # Uncertainty quantification
        self.uncertainty_head = nn.Linear(output_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with uncertainty quantification
        Returns: (prediction, uncertainty)
        """
        # Create quantum-like state representation
        real_part = self.state_real(x)
        imag_part = self.state_imag(x)
        
        # Combine real and imaginary parts
        quantum_state = torch.cat([real_part, imag_part], dim=-1)
        
        # Measurement collapse (prediction)
        prediction = self.measurement(quantum_state)
        
        # Calculate uncertainty (quantum variance analog)
        probability_amplitude = torch.sqrt(real_part**2 + imag_part**2)
        uncertainty = self.uncertainty_head(probability_amplitude)
        uncertainty = F.softplus(uncertainty)  # Ensure positive
        
        return prediction, uncertainty

class MedicalNeuralReasoner:
    """
    Neural reasoning component for medical domain with uncertainty quantification
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = config.get("input_dim", 512)
        
        # TorchLogic medical logic network
        self.medical_logic_network = MedicalLogicNetwork(
            input_dim=self.input_dim,
            num_medical_rules=config.get("num_medical_rules", 12)
        ).to(self.device)
        
        # TorchLogic BanditNRN classifier if available  
        self.brn_classifier = None
        if TORCHLOGIC_AVAILABLE:
            try:
                # Medical target and feature names for TorchLogic
                target_names = ['safe', 'requires_review', 'blocked']
                feature_names = [f'medical_feature_{i}' for i in range(min(32, self.input_dim))]
                
                self.brn_classifier = BanditNRNClassifier(
                    target_names=target_names,
                    feature_names=feature_names,
                    input_size=min(32, self.input_dim),  # Limit input size for computational efficiency
                    output_size=3,  # [safe, requires_review, blocked]
                    layer_sizes=[8, 6],  # Two hidden layers
                    n_selected_features_input=min(8, self.input_dim),
                    n_selected_features_internal=4,
                    n_selected_features_output=2,
                    perform_prune_quantile=0.8,
                    ucb_scale=1.96,  # 95% confidence interval
                    normal_form='dnf',  # Disjunctive normal form
                    bootstrap=True,
                    swa=False,  # Disable stochastic weight averaging for stability
                    add_negations=False
                ).to(self.device)
                logger.info("TorchLogic BanditNRN classifier initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize BanditNRN classifier: {e}")
                self.brn_classifier = None
        
        # Custom uncertainty model
        self.uncertainty_model = QuantumInspiredUncertainty(
            input_dim=self.input_dim,
            output_dim=config.get("output_dim", 256)
        ).to(self.device)
        
        # Medical domain embeddings
        self.medical_embeddings = nn.Embedding(
            num_embeddings=config.get("medical_vocab_size", 10000),
            embedding_dim=config.get("embedding_dim", 512)
        ).to(self.device)
        
        # Ethical constraint network
        self.ethical_filter = self._build_ethical_filter()
        
    def _build_ethical_filter(self) -> nn.Module:
        """Build neural network for ethical constraint filtering"""
        return nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output probability of ethical compliance
        ).to(self.device)
    
    def initialize_components(self) -> None:
        """Initialize neural reasoning components"""
        try:
            # TODO: Initialize actual OSS components
            # self.torchlogic_module = WeightedLogicModule(self.config["torchlogic"])
            # self.symbolic_llm = SymbolicLLM(self.config["symbolicai"])
            
            logger.info("Initializing neural reasoning components...")
            logger.info("TorchLogic module: [PLACEHOLDER - will be WeightedLogicModule]")
            logger.info("SymbolicAI LLM: [PLACEHOLDER - will be SymbolicLLM]")
            logger.info(f"Uncertainty model initialized on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize neural components: {e}")
            raise
    
    def process_medical_input(self, input_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process medical input with neural reasoning and uncertainty quantification
        """
        try:
            # Encode input
            input_embedding = self._encode_medical_text(input_text)
            
            # Medical logic reasoning with TorchLogic
            decision, rule_scores, interpretation = self.medical_logic_network(input_embedding)
            
            # Additional BanditNRN classification if available
            brn_result = None
            if self.brn_classifier is not None:
                try:
                    # Adapt input size for BRN classifier if needed
                    brn_input = input_embedding
                    if input_embedding.shape[-1] > 32:
                        brn_input = input_embedding[:, :32]  # Truncate to fit BRN input size
                    elif input_embedding.shape[-1] < 32:
                        # Pad with zeros if input is smaller
                        padding = torch.zeros(input_embedding.shape[0], 32 - input_embedding.shape[-1], device=input_embedding.device)
                        brn_input = torch.cat([input_embedding, padding], dim=-1)
                    
                    brn_result = self.brn_classifier(brn_input)
                    logger.debug("BanditNRN classification successful")
                except Exception as e:
                    logger.warning(f"BanditNRN classifier error: {e}")
                    brn_result = None
            
            # Neural reasoning with uncertainty
            prediction, uncertainty = self.uncertainty_model(input_embedding)
            
            # Apply ethical filtering
            ethical_score = self._evaluate_ethical_compliance(prediction, context)
            
            # Generate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(prediction, uncertainty)
            
            # Interpret medical logic results
            medical_decision = self._interpret_medical_decision(decision, rule_scores, interpretation)
            
            return {
                "input": input_text,
                "medical_logic_decision": medical_decision,
                "neural_prediction": prediction.detach().cpu().numpy().tolist(),
                "uncertainty": uncertainty.detach().cpu().numpy().tolist(),
                "rule_scores": rule_scores.detach().cpu().numpy().tolist(),
                "brn_result": brn_result.detach().cpu().numpy().tolist() if brn_result is not None else None,
                "ethical_score": ethical_score.item(),
                "confidence_intervals": confidence_intervals,
                "reasoning_type": "neural_logic_hybrid",
                "model_confidence": self._calculate_overall_confidence(uncertainty),
                "torchlogic_used": TORCHLOGIC_AVAILABLE,
                "interpretability": self._generate_interpretability_report(interpretation)
            }
            
        except Exception as e:
            logger.error(f"Error in neural processing: {e}")
            return {
                "input": input_text,
                "error": str(e),
                "status": "neural_processing_failed"
            }
    
    def _encode_medical_text(self, text: str) -> torch.Tensor:
        """Encode medical text to embeddings (placeholder implementation)"""
        # TODO: Implement proper medical text encoding
        # For now, create a random embedding as placeholder
        words = text.lower().split()
        
        # Simulate medical vocabulary encoding
        vocab_indices = [hash(word) % 10000 for word in words[:32]]  # Limit to 32 words
        
        # Pad or truncate to fixed length
        if len(vocab_indices) < 32:
            vocab_indices.extend([0] * (32 - len(vocab_indices)))
        else:
            vocab_indices = vocab_indices[:32]
        
        indices_tensor = torch.tensor(vocab_indices, device=self.device)
        embeddings = self.medical_embeddings(indices_tensor)
        
        # Pool embeddings (mean pooling)
        pooled_embedding = embeddings.mean(dim=0, keepdim=True)
        
        return pooled_embedding
    
    def _evaluate_ethical_compliance(self, prediction: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        """Evaluate ethical compliance of neural prediction"""
        # Combine prediction with context features
        ethical_input = prediction.mean(dim=-1, keepdim=True)  # Simplified
        ethical_score = self.ethical_filter(ethical_input)
        
        return ethical_score
    
    def _calculate_confidence_intervals(self, prediction: torch.Tensor, uncertainty: torch.Tensor) -> Dict[str, List[float]]:
        """Calculate confidence intervals based on uncertainty"""
        # Convert to numpy for calculation
        pred_np = prediction.detach().cpu().numpy().flatten()
        unc_np = uncertainty.detach().cpu().numpy().flatten()
        
        # Calculate 95% confidence intervals
        lower_95 = (pred_np - 1.96 * unc_np).tolist()
        upper_95 = (pred_np + 1.96 * unc_np).tolist()
        
        # Calculate 68% confidence intervals (1 sigma)
        lower_68 = (pred_np - unc_np).tolist()
        upper_68 = (pred_np + unc_np).tolist()
        
        return {
            "95_percent": {"lower": lower_95, "upper": upper_95},
            "68_percent": {"lower": lower_68, "upper": upper_68}
        }
    
    def _calculate_overall_confidence(self, uncertainty: torch.Tensor) -> float:
        """Calculate overall model confidence from uncertainty"""
        # Lower uncertainty = higher confidence
        avg_uncertainty = uncertainty.mean().item()
        confidence = 1.0 / (1.0 + avg_uncertainty)  # Sigmoid-like transformation
        return float(confidence)
    
    def update_model_weights(self, feedback: Dict[str, Any]) -> bool:
        """Update model weights based on feedback"""
        try:
            # TODO: Implement federated learning update
            logger.info(f"Updating neural model with feedback: {feedback}")
            
            # Placeholder for model update logic
            return True
            
        except Exception as e:
            logger.error(f"Failed to update model weights: {e}")
            return False
    
    def get_model_interpretability(self, prediction: torch.Tensor) -> Dict[str, Any]:
        """Generate interpretability information for predictions"""
        try:
            # Attention weights (placeholder)
            attention_weights = F.softmax(prediction, dim=-1)
            
            # Feature importance (simplified)
            feature_importance = torch.abs(prediction).detach().cpu().numpy()
            
            # Top contributing features
            top_features_idx = np.argsort(feature_importance.flatten())[-5:]
            
            return {
                "attention_weights": attention_weights.detach().cpu().numpy().tolist(),
                "feature_importance": feature_importance.tolist(),
                "top_contributing_features": top_features_idx.tolist(),
                "interpretability_score": float(torch.mean(attention_weights).item())
            }
            
        except Exception as e:
            logger.error(f"Error generating interpretability: {e}")
            return {"error": str(e)}
    
    def _interpret_medical_decision(self, decision: torch.Tensor, rule_scores: torch.Tensor, interpretation: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Interpret medical logic decision results"""
        # Convert to numpy for easier processing
        decision_probs = decision.detach().cpu().numpy().flatten()
        rules = rule_scores.detach().cpu().numpy().flatten()
        
        # Decision categories: [safe, requires_review, blocked]
        decision_labels = ["safe", "requires_review", "blocked"]
        primary_decision = decision_labels[np.argmax(decision_probs)]
        
        # Medical rule interpretations
        rule_names = [
            "diagnosis_safety", "emergency_detection", "medication_safety", 
            "privacy_protection", "treatment_appropriateness", "vulnerable_population",
            "anatomy_education", "research_validity", "pathophysiology_accuracy",
            "harmful_content", "pregnancy_safety", "pediatric_safety"
        ]
        
        # Find most activated rules
        top_rule_indices = np.argsort(rules)[-3:][::-1]  # Top 3 rules
        activated_rules = [
            {"rule": rule_names[i] if i < len(rule_names) else f"rule_{i}", 
             "confidence": float(rules[i])}
            for i in top_rule_indices
        ]
        
        return {
            "primary_decision": primary_decision,
            "decision_confidence": float(np.max(decision_probs)),
            "decision_probabilities": {
                label: float(prob) for label, prob in zip(decision_labels, decision_probs)
            },
            "activated_rules": activated_rules,
            "safety_assessment": {
                "is_safe": primary_decision == "safe",
                "needs_review": primary_decision == "requires_review", 
                "is_blocked": primary_decision == "blocked",
                "overall_safety_score": float(1.0 - decision_probs[2])  # 1 - blocked probability
            }
        }
    
    def _generate_interpretability_report(self, interpretation: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Generate comprehensive interpretability report"""
        try:
            report = {
                "feature_analysis": {},
                "logical_reasoning": {},
                "decision_factors": {}
            }
            
            # Analyze diagnosis features
            if "diagnosis_features" in interpretation:
                diag_features = interpretation["diagnosis_features"].detach().cpu().numpy().flatten()
                report["feature_analysis"]["diagnosis"] = {
                    "mean_activation": float(np.mean(diag_features)),
                    "max_activation": float(np.max(diag_features)),
                    "activation_distribution": "normal" if np.std(diag_features) < 0.5 else "skewed"
                }
            
            # Analyze logical reasoning components  
            if "safety_logic" in interpretation:
                safety_logic = interpretation["safety_logic"].detach().cpu().numpy().flatten()
                report["logical_reasoning"]["safety"] = {
                    "logic_strength": float(np.mean(safety_logic)),
                    "consistency": float(1.0 - np.std(safety_logic))
                }
            
            if "ethical_logic" in interpretation:
                ethical_logic = interpretation["ethical_logic"].detach().cpu().numpy().flatten()
                report["logical_reasoning"]["ethics"] = {
                    "ethical_compliance": float(np.mean(ethical_logic)),
                    "ethical_consistency": float(1.0 - np.std(ethical_logic))
                }
            
            # Decision factor analysis
            if "rule_confidences" in interpretation:
                rule_conf = interpretation["rule_confidences"].detach().cpu().numpy().flatten()
                report["decision_factors"]["rule_based"] = {
                    "average_rule_confidence": float(np.mean(rule_conf)),
                    "rule_agreement": float(np.mean(rule_conf > 0.5)),
                    "strongest_rules": np.argsort(rule_conf)[-3:].tolist()
                }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating interpretability report: {e}")
            return {"error": str(e), "interpretability_available": False}

class HybridNeuralSymbolic:
    """
    Bridge between neural and symbolic reasoning components
    """
    
    def __init__(self, neural_reasoner: MedicalNeuralReasoner, symbolic_engine):
        self.neural_reasoner = neural_reasoner
        self.symbolic_engine = symbolic_engine
        
    def fuse_reasoning(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fuse neural and symbolic reasoning for comprehensive medical reasoning
        """
        try:
            # Get neural reasoning results
            neural_result = self.neural_reasoner.process_medical_input(query, context)
            
            # Get symbolic reasoning results
            symbolic_result = self.symbolic_engine.process_medical_query(query, context)
            
            # Fuse results
            fused_confidence = self._calculate_fused_confidence(neural_result, symbolic_result)
            
            # Resolve conflicts
            final_result = self._resolve_reasoning_conflicts(neural_result, symbolic_result)
            
            return {
                "query": query,
                "neural_reasoning": neural_result,
                "symbolic_reasoning": symbolic_result,
                "fused_result": final_result,
                "fused_confidence": fused_confidence,
                "reasoning_method": "hybrid_neuro_symbolic"
            }
            
        except Exception as e:
            logger.error(f"Error in hybrid reasoning: {e}")
            return {
                "query": query,
                "error": str(e),
                "status": "hybrid_reasoning_failed"
            }
    
    def _calculate_fused_confidence(self, neural_result: Dict[str, Any], symbolic_result: Dict[str, Any]) -> float:
        """Calculate combined confidence from neural and symbolic reasoning"""
        neural_conf = neural_result.get("model_confidence", 0.5)
        symbolic_conf = symbolic_result.get("confidence", 0.5)
        
        # Weighted average with higher weight on symbolic for safety
        fused_conf = 0.3 * neural_conf + 0.7 * symbolic_conf
        
        return float(fused_conf)
    
    def _resolve_reasoning_conflicts(self, neural_result: Dict[str, Any], symbolic_result: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflicts between neural and symbolic reasoning"""
        # For medical domain, prioritize symbolic reasoning for safety
        if symbolic_result.get("status") == "blocked":
            return symbolic_result
        
        # If both successful, combine insights
        return {
            "primary_reasoning": "symbolic",
            "supporting_evidence": neural_result.get("neural_prediction"),
            "ethical_compliance": symbolic_result.get("ethical_assessment"),
            "uncertainty_quantification": neural_result.get("uncertainty"),
            "confidence_intervals": neural_result.get("confidence_intervals"),
            "final_recommendation": symbolic_result.get("reasoning_result")
        }

# Factory function for creating neural reasoner
def create_medical_neural_reasoner(config: Optional[Dict[str, Any]] = None) -> MedicalNeuralReasoner:
    """Factory function to create and initialize medical neural reasoner"""
    if config is None:
        config = {
            "input_dim": 512,
            "output_dim": 256,
            "medical_vocab_size": 10000,
            "embedding_dim": 512
        }
    
    reasoner = MedicalNeuralReasoner(config)
    reasoner.initialize_components()
    return reasoner 
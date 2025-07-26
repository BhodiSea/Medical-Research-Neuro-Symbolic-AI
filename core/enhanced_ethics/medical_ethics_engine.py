#!/usr/bin/env python3
"""
Enhanced Medical Ethics Engine
Advanced ethical reasoning for medical AI decisions with context-aware evaluation
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import yaml
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class EthicalSeverity(Enum):
    """Severity levels for ethical violations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class MedicalContext(Enum):
    """Medical context types for ethical evaluation"""
    EDUCATION = "education"
    RESEARCH = "research"
    CLINICAL_SUPPORT = "clinical_support"
    EMERGENCY = "emergency"
    GENERAL_INFO = "general_info"

@dataclass
class EthicalViolation:
    """Represents an ethical violation or concern"""
    rule_id: str
    severity: EthicalSeverity
    description: str
    context: str
    recommendation: str
    confidence: float

@dataclass
class EthicalDecision:
    """Represents an ethical decision with reasoning"""
    decision_id: str
    approved: bool
    confidence: float
    reasoning: List[str]
    violations: List[EthicalViolation]
    constraints_applied: List[str]
    alternatives_suggested: List[str]

class MedicalEthicsRules:
    """Comprehensive medical ethics rules and principles"""
    
    def __init__(self):
        self.core_principles = {
            "beneficence": {
                "description": "Do good - act in the patient's best interest",
                "weight": 1.0,
                "rules": [
                    "maximize_benefit",
                    "evidence_based_recommendations",
                    "appropriate_care_level"
                ]
            },
            "non_maleficence": {
                "description": "Do no harm - avoid causing harm",
                "weight": 1.0,
                "rules": [
                    "no_harmful_advice",
                    "avoid_misdiagnosis_risk",
                    "prevent_medical_errors"
                ]
            },
            "autonomy": {
                "description": "Respect patient autonomy and decision-making",
                "weight": 0.9,
                "rules": [
                    "informed_consent_required",
                    "respect_patient_choices",
                    "no_coercive_recommendations"
                ]
            },
            "justice": {
                "description": "Fair distribution of benefits and risks",
                "weight": 0.9,
                "rules": [
                    "equitable_access",
                    "no_discrimination",
                    "resource_allocation_fairness"
                ]
            }
        }
        
        self.medical_specific_rules = {
            "scope_limitations": {
                "educational_only": "Responses must emphasize educational nature",
                "no_personal_diagnosis": "Cannot provide personal medical diagnosis",
                "no_emergency_care": "Cannot provide emergency medical care",
                "professional_consultation": "Must recommend professional consultation"
            },
            "accuracy_requirements": {
                "evidence_based": "Information must be evidence-based",
                "uncertainty_disclosure": "Must disclose uncertainty and limitations",
                "source_attribution": "Must provide credible sources",
                "update_recommendations": "Must recommend verifying current information"
            },
            "privacy_protection": {
                "no_personal_data": "Cannot request personal health information",
                "confidentiality": "Maintain confidentiality of any shared information",
                "data_minimization": "Minimize data collection and retention"
            }
        }
        
        self.context_specific_rules = {
            MedicalContext.EDUCATION: {
                "level_appropriate": "Content must match educational level",
                "learning_objectives": "Must support clear learning objectives",
                "assessment_safe": "Assessment questions must be educationally sound"
            },
            MedicalContext.CLINICAL_SUPPORT: {
                "supervision_required": "Must emphasize need for clinical supervision",
                "decision_support_only": "Provide support, not replacement for clinical judgment",
                "institutional_protocols": "Must defer to institutional protocols"
            },
            MedicalContext.RESEARCH: {
                "methodology_sound": "Research suggestions must be methodologically sound",
                "ethics_approval": "Must recommend ethics committee approval",
                "participant_protection": "Must emphasize participant protection"
            }
        }

class MedicalEthicsEvaluator:
    """Core medical ethics evaluation engine"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.rules = MedicalEthicsRules()
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.violation_threshold = 0.7  # Threshold for blocking actions
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load ethics configuration"""
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                self.logger.warning(f"Could not load config from {config_path}: {e}")
        
        # Default configuration
        return {
            "strict_mode": True,
            "educational_emphasis": True,
            "require_disclaimers": True,
            "block_personal_advice": True
        }
    
    def evaluate_medical_query(self, query: str, context: MedicalContext, 
                              user_type: str, additional_context: Dict[str, Any] = None) -> EthicalDecision:
        """Evaluate the ethics of a medical query"""
        self.logger.info(f"Evaluating medical query ethics for context: {context.value}")
        
        violations = []
        reasoning = []
        constraints_applied = []
        
        # Check for dangerous patterns
        dangerous_violations = self._check_dangerous_patterns(query, context)
        violations.extend(dangerous_violations)
        
        # Check scope limitations
        scope_violations = self._check_scope_limitations(query, context, user_type)
        violations.extend(scope_violations)
        
        # Check context-specific rules
        context_violations = self._check_context_rules(query, context, additional_context or {})
        violations.extend(context_violations)
        
        # Check privacy concerns
        privacy_violations = self._check_privacy_rules(query, additional_context or {})
        violations.extend(privacy_violations)
        
        # Determine if query should be approved
        critical_violations = [v for v in violations if v.severity == EthicalSeverity.CRITICAL]
        high_violations = [v for v in violations if v.severity == EthicalSeverity.HIGH]
        
        if critical_violations:
            approved = False
            reasoning.append("Critical ethical violations detected - query blocked")
        elif len(high_violations) >= 2:
            approved = False
            reasoning.append("Multiple high-severity ethical concerns - query blocked")
        else:
            approved = True
            reasoning.append("Query meets ethical standards with constraints")
            
            # Apply appropriate constraints
            constraints_applied = self._apply_constraints(violations, context, user_type)
        
        # Calculate confidence
        confidence = self._calculate_ethical_confidence(violations, context)
        
        # Generate alternatives if blocked
        alternatives = []
        if not approved:
            alternatives = self._suggest_alternatives(query, violations, context)
        
        return EthicalDecision(
            decision_id=f"eth_{hash(query) % 10000}",
            approved=approved,
            confidence=confidence,
            reasoning=reasoning,
            violations=violations,
            constraints_applied=constraints_applied,
            alternatives_suggested=alternatives
        )
    
    def _check_dangerous_patterns(self, query: str, context: MedicalContext) -> List[EthicalViolation]:
        """Check for dangerous patterns in the query"""
        violations = []
        query_lower = query.lower()
        
        # Emergency/urgent language
        emergency_patterns = [
            "emergency", "urgent", "immediately", "dying", "severe pain",
            "can't breathe", "chest pain", "having a heart attack"
        ]
        
        for pattern in emergency_patterns:
            if pattern in query_lower:
                violations.append(EthicalViolation(
                    rule_id="emergency_detected",
                    severity=EthicalSeverity.CRITICAL,
                    description=f"Emergency language detected: '{pattern}'",
                    context="Medical emergency requires immediate professional care",
                    recommendation="Direct user to emergency services immediately",
                    confidence=0.9
                ))
        
        # Personal diagnosis requests
        diagnosis_patterns = [
            "diagnose me", "what do I have", "am I sick", "do I have",
            "should I take", "my symptoms", "I am experiencing"
        ]
        
        for pattern in diagnosis_patterns:
            if pattern in query_lower:
                violations.append(EthicalViolation(
                    rule_id="personal_diagnosis_request",
                    severity=EthicalSeverity.HIGH,
                    description=f"Personal diagnosis request detected: '{pattern}'",
                    context="Cannot provide personal medical diagnosis",
                    recommendation="Redirect to educational information and healthcare provider",
                    confidence=0.8
                ))
        
        return violations
    
    def _check_scope_limitations(self, query: str, context: MedicalContext, user_type: str) -> List[EthicalViolation]:
        """Check scope limitations based on context and user type"""
        violations = []
        
        # Check if clinical advice is being requested in educational context
        if context == MedicalContext.EDUCATION:
            clinical_patterns = ["treatment for", "medication for", "should treat"]
            query_lower = query.lower()
            
            for pattern in clinical_patterns:
                if pattern in query_lower:
                    violations.append(EthicalViolation(
                        rule_id="clinical_advice_in_education",
                        severity=EthicalSeverity.MEDIUM,
                        description=f"Clinical advice requested in educational context",
                        context="Educational context should focus on learning, not clinical decisions",
                        recommendation="Emphasize educational nature and recommend clinical consultation",
                        confidence=0.7
                    ))
        
        # Check user type appropriateness
        if user_type == "medical_student" and "patient care" in query.lower():
            violations.append(EthicalViolation(
                rule_id="inappropriate_user_scope",
                severity=EthicalSeverity.MEDIUM,
                description="Medical student requesting patient care guidance",
                context="Medical students should not make independent patient care decisions",
                recommendation="Emphasize supervision requirement and educational nature",
                confidence=0.8
            ))
        
        return violations
    
    def _check_context_rules(self, query: str, context: MedicalContext, 
                           additional_context: Dict[str, Any]) -> List[EthicalViolation]:
        """Check context-specific ethical rules"""
        violations = []
        
        context_rules = self.rules.context_specific_rules.get(context, {})
        
        # Educational context checks
        if context == MedicalContext.EDUCATION:
            student_level = additional_context.get("student_level", "unknown")
            if student_level == "unknown":
                violations.append(EthicalViolation(
                    rule_id="missing_educational_level",
                    severity=EthicalSeverity.LOW,
                    description="Educational level not specified",
                    context="Cannot ensure level-appropriate content",
                    recommendation="Request clarification of educational level",
                    confidence=0.6
                ))
        
        # Clinical support context checks
        if context == MedicalContext.CLINICAL_SUPPORT:
            if not additional_context.get("supervision_acknowledged", False):
                violations.append(EthicalViolation(
                    rule_id="supervision_not_acknowledged",
                    severity=EthicalSeverity.HIGH,
                    description="Clinical supervision not acknowledged",
                    context="Clinical support requires supervision acknowledgment",
                    recommendation="Require explicit supervision acknowledgment",
                    confidence=0.9
                ))
        
        return violations
    
    def _check_privacy_rules(self, query: str, additional_context: Dict[str, Any]) -> List[EthicalViolation]:
        """Check privacy protection rules"""
        violations = []
        
        # Check for personal identifiers
        personal_patterns = [
            "my name is", "I am", "patient named", "SSN", "date of birth",
            "address", "phone number"
        ]
        
        query_lower = query.lower()
        for pattern in personal_patterns:
            if pattern in query_lower:
                violations.append(EthicalViolation(
                    rule_id="personal_identifiers_detected",
                    severity=EthicalSeverity.MEDIUM,
                    description=f"Personal identifier detected: '{pattern}'",
                    context="Personal identifiers should not be shared",
                    recommendation="Remove personal identifiers and use anonymized examples",
                    confidence=0.7
                ))
        
        return violations
    
    def _apply_constraints(self, violations: List[EthicalViolation], 
                          context: MedicalContext, user_type: str) -> List[str]:
        """Apply appropriate constraints based on violations and context"""
        constraints = []
        
        # Always apply educational disclaimer
        constraints.append("educational_disclaimer")
        
        # Apply context-specific constraints
        if context == MedicalContext.EDUCATION:
            constraints.append("emphasize_learning_objectives")
            constraints.append("recommend_authoritative_sources")
        
        if context == MedicalContext.CLINICAL_SUPPORT:
            constraints.append("require_supervision_reminder")
            constraints.append("defer_to_institutional_protocols")
        
        # Apply violation-specific constraints
        for violation in violations:
            if violation.rule_id == "personal_diagnosis_request":
                constraints.append("block_diagnosis_language")
                constraints.append("redirect_to_healthcare_provider")
            
            if violation.rule_id == "emergency_detected":
                constraints.append("emergency_services_redirect")
        
        # User type specific constraints
        if user_type == "medical_student":
            constraints.append("emphasize_educational_nature")
            constraints.append("recommend_instructor_consultation")
        
        return list(set(constraints))  # Remove duplicates
    
    def _calculate_ethical_confidence(self, violations: List[EthicalViolation], 
                                    context: MedicalContext) -> float:
        """Calculate confidence in ethical evaluation"""
        if not violations:
            return 0.95
        
        # Calculate severity-weighted score
        severity_weights = {
            EthicalSeverity.LOW: 0.1,
            EthicalSeverity.MEDIUM: 0.3,
            EthicalSeverity.HIGH: 0.7,
            EthicalSeverity.CRITICAL: 1.0
        }
        
        total_severity = sum(severity_weights[v.severity] for v in violations)
        max_possible = len(violations) * severity_weights[EthicalSeverity.CRITICAL]
        
        # Confidence decreases with severity
        confidence = max(0.1, 1.0 - (total_severity / max_possible))
        
        return round(confidence, 2)
    
    def _suggest_alternatives(self, query: str, violations: List[EthicalViolation], 
                            context: MedicalContext) -> List[str]:
        """Suggest alternative approaches for blocked queries"""
        alternatives = []
        
        # Generic alternatives
        alternatives.append("Rephrase as an educational question about medical concepts")
        alternatives.append("Consult with a qualified healthcare provider")
        
        # Specific alternatives based on violations
        for violation in violations:
            if violation.rule_id == "emergency_detected":
                alternatives.append("Contact emergency services immediately (911)")
                alternatives.append("Visit the nearest emergency department")
            
            elif violation.rule_id == "personal_diagnosis_request":
                alternatives.append("Ask about general information on symptoms or conditions")
                alternatives.append("Schedule an appointment with your healthcare provider")
            
            elif violation.rule_id == "clinical_advice_in_education":
                alternatives.append("Focus on understanding the underlying medical principles")
                alternatives.append("Discuss clinical applications with supervising physicians")
        
        return alternatives
    
    def get_ethics_summary(self) -> Dict[str, Any]:
        """Get summary of ethics engine configuration and status"""
        return {
            "core_principles": list(self.rules.core_principles.keys()),
            "medical_rules": list(self.rules.medical_specific_rules.keys()),
            "supported_contexts": [ctx.value for ctx in MedicalContext],
            "violation_threshold": self.violation_threshold,
            "config": self.config,
            "ready": True
        }

def create_medical_ethics_engine(config_path: Optional[str] = None) -> MedicalEthicsEvaluator:
    """Factory function to create medical ethics engine"""
    logger.info("Creating medical ethics engine...")
    engine = MedicalEthicsEvaluator(config_path)
    logger.info("Medical ethics engine created successfully")
    return engine

# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create ethics engine
    ethics_engine = create_medical_ethics_engine()
    
    print("🛡️ Medical Ethics Engine Demo")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        {
            "query": "Explain the pathophysiology of myocardial infarction",
            "context": MedicalContext.EDUCATION,
            "user_type": "medical_student",
            "additional_context": {"student_level": "year_2"}
        },
        {
            "query": "I am having chest pain, what should I do?",
            "context": MedicalContext.GENERAL_INFO,
            "user_type": "general_public",
            "additional_context": {}
        },
        {
            "query": "What medication should I prescribe for hypertension?",
            "context": MedicalContext.CLINICAL_SUPPORT,
            "user_type": "resident",
            "additional_context": {"supervision_acknowledged": False}
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: '{case['query'][:50]}...'")
        print(f"   Context: {case['context'].value}, User: {case['user_type']}")
        
        decision = ethics_engine.evaluate_medical_query(
            case["query"],
            case["context"],
            case["user_type"],
            case["additional_context"]
        )
        
        print(f"   ✅ Approved: {decision.approved}")
        print(f"   🎯 Confidence: {decision.confidence}")
        print(f"   📋 Violations: {len(decision.violations)}")
        
        if decision.violations:
            for violation in decision.violations:
                print(f"      - {violation.severity.value}: {violation.description}")
        
        if decision.constraints_applied:
            print(f"   🔒 Constraints: {', '.join(decision.constraints_applied[:3])}")
        
        if decision.alternatives_suggested and not decision.approved:
            print(f"   💡 Alternatives: {len(decision.alternatives_suggested)} suggested")
    
    print(f"\n📊 Ethics Engine Summary:")
    summary = ethics_engine.get_ethics_summary()
    print(f"   Core Principles: {', '.join(summary['core_principles'])}")
    print(f"   Supported Contexts: {len(summary['supported_contexts'])}")
    print(f"   Ready: {summary['ready']}") 
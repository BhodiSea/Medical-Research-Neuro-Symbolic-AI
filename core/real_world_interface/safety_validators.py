"""
Real-World Safety Validators
Ensure compliance with ethics and privacy when accessing real data
"""

import re
import hashlib
import json
from typing import Dict, List, Any, Optional, Set
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class DataSensitivityLevel(Enum):
    """Data sensitivity classification"""
    PUBLIC = "public"
    RESEARCH = "research"
    CLINICAL = "clinical"
    PERSONAL = "personal"
    RESTRICTED = "restricted"

class ValidationResult(Enum):
    """Validation result types"""
    APPROVED = "approved"
    REJECTED = "rejected"
    REQUIRES_APPROVAL = "requires_approval"
    MODIFIED = "modified"

@dataclass
class ValidationResponse:
    """Response from validation check"""
    result: ValidationResult
    message: str
    modified_query: Optional[str] = None
    sensitivity_level: Optional[DataSensitivityLevel] = None
    risk_factors: Optional[List[str]] = None

class BaseValidator(ABC):
    """Base class for all validators"""
    
    @abstractmethod
    def validate(self, query: str, data_source: str, context: Dict[str, Any]) -> ValidationResponse:
        """Validate a query against safety criteria"""
        pass

class PrivacyValidator(BaseValidator):
    """Validate queries for privacy compliance"""
    
    def __init__(self):
        # Personal identifiers patterns
        self.personal_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{3}-\d{3}-\d{4}\b',  # Phone
            r'\bpatient\s+id\b',
            r'\bmedical\s+record\s+number\b',
            r'\bhealth\s+insurance\b'
        ]
        
        # Sensitive medical terms
        self.sensitive_terms = {
            'personal_identifiers',
            'patient_name',
            'medical_record',
            'insurance_number',
            'date_of_birth',
            'address',
            'phone_number'
        }
        
    def validate(self, query: str, data_source: str, context: Dict[str, Any]) -> ValidationResponse:
        """Validate query for privacy compliance"""
        query_lower = query.lower()
        risk_factors = []
        
        # Check for personal identifier patterns
        for pattern in self.personal_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                risk_factors.append(f"Contains personal identifier pattern: {pattern}")
        
        # Check for sensitive terms
        for term in self.sensitive_terms:
            if term in query_lower:
                risk_factors.append(f"Contains sensitive term: {term}")
        
        # Check for personal pronouns indicating personal data
        personal_pronouns = ['my', 'i have', 'i am', 'my patient']
        for pronoun in personal_pronouns:
            if pronoun in query_lower:
                risk_factors.append(f"Contains personal reference: {pronoun}")
        
        # Determine validation result
        if risk_factors:
            if len(risk_factors) > 2 or any('identifier' in rf for rf in risk_factors):
                return ValidationResponse(
                    result=ValidationResult.REJECTED,
                    message="Query contains personal identifiers or excessive sensitive information",
                    sensitivity_level=DataSensitivityLevel.PERSONAL,
                    risk_factors=risk_factors
                )
            else:
                # Modify query to remove sensitive terms
                modified_query = self._anonymize_query(query)
                return ValidationResponse(
                    result=ValidationResult.MODIFIED,
                    message="Query modified to remove sensitive information",
                    modified_query=modified_query,
                    sensitivity_level=DataSensitivityLevel.RESEARCH,
                    risk_factors=risk_factors
                )
        
        return ValidationResponse(
            result=ValidationResult.APPROVED,
            message="Query approved for privacy compliance",
            sensitivity_level=DataSensitivityLevel.PUBLIC
        )
    
    def _anonymize_query(self, query: str) -> str:
        """Anonymize query by removing/replacing sensitive terms"""
        anonymized = query
        
        # Replace personal pronouns with generic terms
        anonymized = re.sub(r'\bmy\b', 'the', anonymized, flags=re.IGNORECASE)
        anonymized = re.sub(r'\bi have\b', 'patient has', anonymized, flags=re.IGNORECASE)
        anonymized = re.sub(r'\bi am\b', 'patient is', anonymized, flags=re.IGNORECASE)
        
        # Remove personal identifiers
        for pattern in self.personal_patterns:
            anonymized = re.sub(pattern, '[REDACTED]', anonymized, flags=re.IGNORECASE)
        
        return anonymized

class EthicalConstraintValidator(BaseValidator):
    """Validate queries against ethical constraints"""
    
    def __init__(self, ethical_constraints: Dict[str, Any]):
        self.constraints = ethical_constraints
        
        # Load forbidden topics
        self.forbidden_topics = set(
            ethical_constraints.get('forbidden_topics', [])
        )
        
        # Load restricted research areas
        self.restricted_areas = set(
            ethical_constraints.get('restricted_research_areas', [])
        )
        
        # Harmful query patterns
        self.harmful_patterns = [
            r'diagnose\s+me',
            r'should\s+i\s+take',
            r'what\s+medication',
            r'am\s+i\s+having',
            r'emergency',
            r'urgent\s+medical'
        ]
    
    def validate(self, query: str, data_source: str, context: Dict[str, Any]) -> ValidationResponse:
        """Validate query against ethical constraints"""
        query_lower = query.lower()
        risk_factors = []
        
        # Check for forbidden topics
        for topic in self.forbidden_topics:
            if topic.lower() in query_lower:
                risk_factors.append(f"Contains forbidden topic: {topic}")
        
        # Check for restricted research areas
        for area in self.restricted_areas:
            if area.lower() in query_lower:
                risk_factors.append(f"Involves restricted research area: {area}")
        
        # Check for harmful patterns
        for pattern in self.harmful_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                risk_factors.append(f"Contains harmful pattern: {pattern}")
        
        # Check data source restrictions
        if not self._is_approved_data_source(data_source):
            risk_factors.append(f"Unapproved data source: {data_source}")
        
        # Determine validation result
        if risk_factors:
            severity = self._assess_risk_severity(risk_factors)
            
            if severity == "high":
                return ValidationResponse(
                    result=ValidationResult.REJECTED,
                    message="Query violates ethical constraints",
                    sensitivity_level=DataSensitivityLevel.RESTRICTED,
                    risk_factors=risk_factors
                )
            elif severity == "medium":
                return ValidationResponse(
                    result=ValidationResult.REQUIRES_APPROVAL,
                    message="Query requires human approval due to ethical concerns",
                    sensitivity_level=DataSensitivityLevel.CLINICAL,
                    risk_factors=risk_factors
                )
        
        return ValidationResponse(
            result=ValidationResult.APPROVED,
            message="Query meets ethical constraints",
            sensitivity_level=DataSensitivityLevel.RESEARCH
        )
    
    def _is_approved_data_source(self, data_source: str) -> bool:
        """Check if data source is approved"""
        approved_sources = {
            'pubmed', 'pubchem', 'ncbi', 'clinicaltrials.gov', 
            'pdb', 'uniprot', 'ensembl', 'omim'
        }
        return data_source.lower() in approved_sources
    
    def _assess_risk_severity(self, risk_factors: List[str]) -> str:
        """Assess overall risk severity"""
        high_risk_keywords = ['forbidden', 'harmful', 'emergency', 'urgent']
        medium_risk_keywords = ['restricted', 'unapproved']
        
        high_risk_count = sum(
            1 for rf in risk_factors 
            if any(keyword in rf.lower() for keyword in high_risk_keywords)
        )
        
        if high_risk_count > 0:
            return "high"
        
        medium_risk_count = sum(
            1 for rf in risk_factors 
            if any(keyword in rf.lower() for keyword in medium_risk_keywords)
        )
        
        if medium_risk_count > 1:
            return "medium"
        
        return "low"

class DataProvenanceValidator(BaseValidator):
    """Validate data source legitimacy and provenance"""
    
    def __init__(self):
        # Approved data sources with their characteristics
        self.approved_sources = {
            'pubmed': {
                'domain': 'ncbi.nlm.nih.gov',
                'api_endpoint': 'eutils.ncbi.nlm.nih.gov',
                'data_type': 'literature',
                'access_level': 'public'
            },
            'pubchem': {
                'domain': 'pubchem.ncbi.nlm.nih.gov',
                'api_endpoint': 'pubchem.ncbi.nlm.nih.gov/rest/pug',
                'data_type': 'chemical',
                'access_level': 'public'
            },
            'clinicaltrials': {
                'domain': 'clinicaltrials.gov',
                'api_endpoint': 'clinicaltrials.gov/api',
                'data_type': 'clinical_trials',
                'access_level': 'public'
            },
            'pdb': {
                'domain': 'rcsb.org',
                'api_endpoint': 'search.rcsb.org',
                'data_type': 'protein_structure',
                'access_level': 'public'
            }
        }
        
        # Suspicious patterns that might indicate illegitimate sources
        self.suspicious_patterns = [
            r'internal[-_]database',
            r'private[-_]api',
            r'medical[-_]records',
            r'patient[-_]data',
            r'hospital[-_]system'
        ]
    
    def validate(self, query: str, data_source: str, context: Dict[str, Any]) -> ValidationResponse:
        """Validate data source legitimacy"""
        risk_factors = []
        
        # Check if data source is approved
        if data_source.lower() not in self.approved_sources:
            risk_factors.append(f"Unapproved data source: {data_source}")
        
        # Check for suspicious patterns in query or context
        combined_text = f"{query} {json.dumps(context)}"
        for pattern in self.suspicious_patterns:
            if re.search(pattern, combined_text, re.IGNORECASE):
                risk_factors.append(f"Suspicious pattern detected: {pattern}")
        
        # Check for potential data source spoofing
        if self._detect_source_spoofing(data_source, context):
            risk_factors.append("Potential data source spoofing detected")
        
        # Determine validation result
        if risk_factors:
            return ValidationResponse(
                result=ValidationResult.REJECTED,
                message="Data source validation failed",
                sensitivity_level=DataSensitivityLevel.RESTRICTED,
                risk_factors=risk_factors
            )
        
        # Get approved source info
        source_info = self.approved_sources.get(data_source.lower(), {})
        sensitivity = DataSensitivityLevel.PUBLIC if source_info.get('access_level') == 'public' else DataSensitivityLevel.RESEARCH
        
        return ValidationResponse(
            result=ValidationResult.APPROVED,
            message=f"Data source {data_source} validated successfully",
            sensitivity_level=sensitivity
        )
    
    def _detect_source_spoofing(self, data_source: str, context: Dict[str, Any]) -> bool:
        """Detect potential data source spoofing"""
        # Check if context contains conflicting information about the source
        context_sources = []
        
        for key, value in context.items():
            if 'source' in key.lower() or 'api' in key.lower():
                if isinstance(value, str):
                    context_sources.append(value.lower())
        
        # If context mentions different sources, it might be spoofing
        if context_sources and data_source.lower() not in ' '.join(context_sources):
            return True
        
        return False

class DifferentialPrivacyValidator(BaseValidator):
    """Apply differential privacy validation to queries"""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon  # Privacy budget
        self.delta = delta      # Failure probability
        self.query_log = {}     # Track queries for privacy budget
        
    def validate(self, query: str, data_source: str, context: Dict[str, Any]) -> ValidationResponse:
        """Validate query for differential privacy compliance"""
        query_hash = self._hash_query(query)
        
        # Check privacy budget
        current_usage = self.query_log.get(query_hash, 0)
        if current_usage >= self.epsilon:
            return ValidationResponse(
                result=ValidationResult.REJECTED,
                message="Privacy budget exceeded for similar queries",
                sensitivity_level=DataSensitivityLevel.PERSONAL,
                risk_factors=["Privacy budget exhausted"]
            )
        
        # Check if query can be made differentially private
        if not self._can_apply_differential_privacy(query):
            return ValidationResponse(
                result=ValidationResult.REQUIRES_APPROVAL,
                message="Query requires review for differential privacy compliance",
                sensitivity_level=DataSensitivityLevel.CLINICAL,
                risk_factors=["Cannot apply differential privacy automatically"]
            )
        
        # Apply differential privacy transformation
        private_query = self._apply_noise(query)
        
        # Update privacy budget
        self.query_log[query_hash] = current_usage + 0.1
        
        return ValidationResponse(
            result=ValidationResult.MODIFIED,
            message="Query modified for differential privacy compliance",
            modified_query=private_query,
            sensitivity_level=DataSensitivityLevel.RESEARCH
        )
    
    def _hash_query(self, query: str) -> str:
        """Hash query for privacy budget tracking"""
        normalized = re.sub(r'\s+', ' ', query.lower().strip())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]
    
    def _can_apply_differential_privacy(self, query: str) -> bool:
        """Check if differential privacy can be applied to query"""
        # Queries that aggregate data can typically use differential privacy
        aggregation_terms = ['count', 'average', 'mean', 'sum', 'total', 'statistics']
        
        query_lower = query.lower()
        return any(term in query_lower for term in aggregation_terms)
    
    def _apply_noise(self, query: str) -> str:
        """Apply noise for differential privacy (simplified)"""
        # In real implementation, would apply Laplace or Gaussian noise
        # This is a simplified version that adds randomization instructions
        
        if 'count' in query.lower():
            return f"{query} (apply Laplace noise with scale 1/{self.epsilon})"
        elif any(term in query.lower() for term in ['average', 'mean']):
            return f"{query} (apply Gaussian noise for differential privacy)"
        else:
            return f"{query} (apply k-anonymity with k=5)"

class RealWorldSafetyValidator:
    """Main safety validator that orchestrates all validation checks"""
    
    def __init__(self, ethical_constraints: Optional[Dict[str, Any]] = None):
        # Initialize sub-validators
        self.privacy_validator = PrivacyValidator()
        self.ethical_validator = EthicalConstraintValidator(
            ethical_constraints or self._get_default_constraints()
        )
        self.provenance_validator = DataProvenanceValidator()
        self.differential_privacy_validator = DifferentialPrivacyValidator()
        
        # Validation order (most restrictive first)
        self.validators = [
            self.provenance_validator,
            self.privacy_validator,
            self.ethical_validator,
            self.differential_privacy_validator
        ]
    
    def validate_external_query(self, query: str, data_source: str, context: Optional[Dict[str, Any]] = None) -> ValidationResponse:
        """Comprehensive validation of external data query"""
        context = context or {}
        
        # Run all validators in sequence
        for validator in self.validators:
            result = validator.validate(query, data_source, context)
            
            # If any validator rejects, return immediately
            if result.result == ValidationResult.REJECTED:
                logger.warning(f"Query rejected by {validator.__class__.__name__}: {result.message}")
                return result
            
            # If query was modified, use the modified version for subsequent validators
            if result.result == ValidationResult.MODIFIED and result.modified_query:
                query = result.modified_query
        
        # If we get here, all validators passed
        return ValidationResponse(
            result=ValidationResult.APPROVED,
            message="Query passed all safety validations",
            modified_query=query if query != context.get('original_query', query) else None,
            sensitivity_level=DataSensitivityLevel.PUBLIC
        )
    
    def anonymize_real_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply anonymization to real medical data"""
        anonymized = data.copy()
        
        # Remove or hash personal identifiers
        sensitive_fields = {
            'patient_id', 'medical_record_number', 'ssn', 'email', 'phone',
            'name', 'address', 'birth_date', 'insurance_number'
        }
        
        for field in sensitive_fields:
            if field in anonymized:
                if isinstance(anonymized[field], str):
                    # Hash sensitive string data
                    anonymized[field] = hashlib.sha256(str(anonymized[field]).encode()).hexdigest()[:12]
                else:
                    # Remove non-string sensitive data
                    del anonymized[field]
        
        # Add anonymization metadata
        anonymized['_anonymization_applied'] = True
        anonymized['_anonymization_timestamp'] = hash(str(data))
        
        return anonymized
    
    def check_data_provenance(self, data_source: str) -> bool:
        """Check if data source is legitimate and approved"""
        result = self.provenance_validator.validate("", data_source, {})
        return result.result in [ValidationResult.APPROVED, ValidationResult.MODIFIED]
    
    def assess_privacy_risk(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess privacy risk of a query"""
        privacy_result = self.privacy_validator.validate(query, "", context)
        
        return {
            'risk_level': privacy_result.sensitivity_level.value if privacy_result.sensitivity_level else 'unknown',
            'risk_factors': privacy_result.risk_factors or [],
            'requires_approval': privacy_result.result == ValidationResult.REQUIRES_APPROVAL,
            'can_proceed': privacy_result.result != ValidationResult.REJECTED
        }
    
    def _get_default_constraints(self) -> Dict[str, Any]:
        """Get default ethical constraints"""
        return {
            'forbidden_topics': [
                'personal_medical_advice',
                'emergency_diagnosis',
                'prescription_recommendations'
            ],
            'restricted_research_areas': [
                'human_experimentation',
                'genetic_discrimination',
                'unauthorized_clinical_trials'
            ]
        }
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation system status"""
        return {
            'validators_active': len(self.validators),
            'privacy_budget_remaining': self.differential_privacy_validator.epsilon,
            'approved_data_sources': list(self.provenance_validator.approved_sources.keys()),
            'validation_system_status': 'operational'
        }

# Factory function
def create_real_world_safety_validator(ethical_constraints: Optional[Dict[str, Any]] = None) -> RealWorldSafetyValidator:
    """Factory function to create RealWorldSafetyValidator"""
    return RealWorldSafetyValidator(ethical_constraints)
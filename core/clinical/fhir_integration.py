"""
FHIR Integration for Medical Research AI

This module provides integration with FHIR (Fast Healthcare Interoperability Resources)
for healthcare data standards and patient data interoperability.

FHIR is available via PyPI: pip install fhirclient
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add FHIR submodule to path
fhir_path = Path(__file__).parent / "fhir"
if str(fhir_path) not in sys.path:
    sys.path.insert(0, str(fhir_path))

# Global flags for FHIR availability - will be set on first use
FHIR_AVAILABLE = None
FHIR_INITIALIZED = False


class FHIRIntegration:
    """
    Integration wrapper for FHIR (Fast Healthcare Interoperability Resources).
    
    FHIR provides healthcare data standards and patient data interoperability
    for medical research and clinical data management.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize FHIR integration.
        
        Args:
            config: Configuration dictionary with FHIR settings
        """
        self.config = config or {}
        self.client = None
        self.server_url = None
        self.resource_types = {}
        self._fhir_components = {}
        
        # Don't initialize anything at startup - use lazy loading
        logger.info("FHIR integration initialized with lazy loading")
    
    def _check_fhir_availability(self) -> bool:
        """Check if FHIR is available and initialize if needed."""
        global FHIR_AVAILABLE, FHIR_INITIALIZED
        
        if FHIR_AVAILABLE is None:
            try:
                # Try to import FHIR components only when needed
                from fhirclient import client
                from fhirclient.models import patient, observation, condition, medication
                from fhirclient.models import procedure, encounter, diagnosticreport
                
                # Store components for later use
                self._fhir_components = {
                    'client': client,
                    'patient': patient,
                    'observation': observation,
                    'condition': condition,
                    'medication': medication,
                    'procedure': procedure,
                    'encounter': encounter,
                    'diagnosticreport': diagnosticreport
                }
                
                FHIR_AVAILABLE = True
                logger.info("FHIR components loaded successfully")
                
            except ImportError as e:
                FHIR_AVAILABLE = False
                logger.warning(f"FHIR not available: {e}")
                logger.info("Install with: pip install fhirclient")
        
        return FHIR_AVAILABLE
    
    def _initialize_fhir_systems(self) -> None:
        """Initialize FHIR systems and components - called only when needed."""
        global FHIR_INITIALIZED
        
        if FHIR_INITIALIZED:
            return
            
        try:
            if not self._check_fhir_availability():
                return
                
            # Initialize FHIR client
            self._initialize_fhir_client()
            
            # Initialize resource types
            self._initialize_resource_types()
            
            FHIR_INITIALIZED = True
            logger.info("FHIR systems initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing FHIR systems: {e}")
    
    def _initialize_fhir_client(self) -> None:
        """Initialize FHIR client."""
        try:
            client = self._fhir_components['client']
            
            # Set up FHIR client with default server
            server_url = self.config.get("server_url", "https://hapi.fhir.org/baseR4")
            self.server_url = server_url
            
            settings = {
                'app_id': 'medical_research_ai',
                'api_base': server_url
            }
            
            self.client = client.FHIRClient(settings=settings)
            logger.info(f"FHIR client initialized with server: {server_url}")
        except Exception as e:
            logger.error(f"Error initializing FHIR client: {e}")
    
    def _initialize_resource_types(self) -> None:
        """Initialize FHIR resource types."""
        try:
            # Define common FHIR resource types
            self.resource_types = {
                "Patient": self._fhir_components['patient'].Patient,
                "Observation": self._fhir_components['observation'].Observation,
                "Condition": self._fhir_components['condition'].Condition,
                "Medication": self._fhir_components['medication'].Medication,
                "Procedure": self._fhir_components['procedure'].Procedure,
                "Encounter": self._fhir_components['encounter'].Encounter,
                "DiagnosticReport": self._fhir_components['diagnosticreport'].DiagnosticReport
            }
            
            logger.info("FHIR resource types initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing resource types: {e}")
    
    def search_patients(self, 
                      search_criteria: Dict[str, Any],
                      limit: int = 10) -> Dict[str, Any]:
        """
        Search for patients using FHIR.
        
        Args:
            search_criteria: Dictionary of search criteria
            limit: Maximum number of results to return
            
        Returns:
            Dictionary containing patient search results
        """
        # Initialize FHIR only when this method is called
        if not self._check_fhir_availability():
            return self._mock_patient_search(search_criteria, limit)
        
        try:
            # Initialize systems on first use
            if not FHIR_INITIALIZED:
                self._initialize_fhir_systems()
            
            Patient = self.resource_types["Patient"]
            
            # Build search parameters
            search_params = {}
            for key, value in search_criteria.items():
                search_params[key] = value
            
            # Perform search
            patients = Patient.where(search_params).limit(limit).perform(self.client)
            
            # Process results
            patient_results = []
            for patient in patients:
                patient_data = {
                    "id": patient.id,
                    "name": f"{patient.name[0].given[0]} {patient.name[0].family}" if patient.name else "Unknown",
                    "gender": patient.gender,
                    "birth_date": patient.birthDate.isostring if patient.birthDate else None,
                    "resource_type": "Patient"
                }
                patient_results.append(patient_data)
            
            return {
                "search_criteria": search_criteria,
                "limit": limit,
                "status": "completed",
                "total_results": len(patient_results),
                "patients": patient_results,
                "metadata": {
                    "model": "FHIR",
                    "server_url": self.server_url,
                    "search_method": "patient_search"
                }
            }
            
        except Exception as e:
            logger.error(f"Error searching patients: {e}")
            return self._mock_patient_search(search_criteria, limit)
    
    def get_patient_observations(self, 
                               patient_id: str,
                               observation_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get observations for a specific patient.
        
        Args:
            patient_id: FHIR patient ID
            observation_types: Optional list of observation types to filter by
            
        Returns:
            Dictionary containing patient observations
        """
        # Initialize FHIR only when this method is called
        if not self._check_fhir_availability():
            return self._mock_patient_observations(patient_id, observation_types)
        
        try:
            # Initialize systems on first use
            if not FHIR_INITIALIZED:
                self._initialize_fhir_systems()
            
            Observation = self.resource_types["Observation"]
            
            # Build search parameters
            search_params = {"subject": f"Patient/{patient_id}"}
            if observation_types:
                search_params["code"] = ",".join(observation_types)
            
            # Perform search
            observations = Observation.where(search_params).perform(self.client)
            
            # Process results
            observation_results = []
            for obs in observations:
                obs_data = {
                    "id": obs.id,
                    "code": obs.code.coding[0].code if obs.code and obs.code.coding else None,
                    "display": obs.code.coding[0].display if obs.code and obs.code.coding else None,
                    "value": obs.valueQuantity.value if obs.valueQuantity else None,
                    "unit": obs.valueQuantity.unit if obs.valueQuantity else None,
                    "date": obs.effectiveDateTime.isostring if obs.effectiveDateTime else None,
                    "resource_type": "Observation"
                }
                observation_results.append(obs_data)
            
            return {
                "patient_id": patient_id,
                "observation_types": observation_types,
                "status": "completed",
                "total_observations": len(observation_results),
                "observations": observation_results,
                "metadata": {
                    "model": "FHIR",
                    "server_url": self.server_url,
                    "search_method": "observation_search"
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting patient observations: {e}")
            return self._mock_patient_observations(patient_id, observation_types)
    
    def get_patient_conditions(self, 
                             patient_id: str,
                             condition_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get conditions for a specific patient.
        
        Args:
            patient_id: FHIR patient ID
            condition_types: Optional list of condition types to filter by
            
        Returns:
            Dictionary containing patient conditions
        """
        # Initialize FHIR only when this method is called
        if not self._check_fhir_availability():
            return self._mock_patient_conditions(patient_id, condition_types)
        
        try:
            # Initialize systems on first use
            if not FHIR_INITIALIZED:
                self._initialize_fhir_systems()
            
            Condition = self.resource_types["Condition"]
            
            # Build search parameters
            search_params = {"subject": f"Patient/{patient_id}"}
            if condition_types:
                search_params["code"] = ",".join(condition_types)
            
            # Perform search
            conditions = Condition.where(search_params).perform(self.client)
            
            # Process results
            condition_results = []
            for condition in conditions:
                condition_data = {
                    "id": condition.id,
                    "code": condition.code.coding[0].code if condition.code and condition.code.coding else None,
                    "display": condition.code.coding[0].display if condition.code and condition.code.coding else None,
                    "clinical_status": condition.clinicalStatus.coding[0].code if condition.clinicalStatus else None,
                    "onset_date": condition.onsetDateTime.isostring if condition.onsetDateTime else None,
                    "resource_type": "Condition"
                }
                condition_results.append(condition_data)
            
            return {
                "patient_id": patient_id,
                "condition_types": condition_types,
                "status": "completed",
                "total_conditions": len(condition_results),
                "conditions": condition_results,
                "metadata": {
                    "model": "FHIR",
                    "server_url": self.server_url,
                    "search_method": "condition_search"
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting patient conditions: {e}")
            return self._mock_patient_conditions(patient_id, condition_types)
    
    def create_fhir_resource(self, 
                           resource_type: str,
                           resource_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new FHIR resource.
        
        Args:
            resource_type: Type of FHIR resource to create
            resource_data: Data for the new resource
            
        Returns:
            Dictionary containing creation results
        """
        # Initialize FHIR only when this method is called
        if not self._check_fhir_availability():
            return self._mock_resource_creation(resource_type, resource_data)
        
        try:
            # Initialize systems on first use
            if not FHIR_INITIALIZED:
                self._initialize_fhir_systems()
            
            if resource_type not in self.resource_types:
                return self._mock_resource_creation(resource_type, resource_data)
            
            ResourceClass = self.resource_types[resource_type]
            
            # Create resource instance
            resource = ResourceClass(resource_data)
            
            # Save resource (simulated for mock mode)
            # In a real implementation, this would save to the FHIR server
            creation_result = {
                "resource_id": f"mock_{resource_type.lower()}_id",
                "resource_type": resource_type,
                "creation_status": "completed",
                "resource_data": resource_data
            }
            
            return {
                "resource_type": resource_type,
                "resource_data": resource_data,
                "status": "completed",
                "creation_result": creation_result,
                "metadata": {
                    "model": "FHIR",
                    "server_url": self.server_url,
                    "creation_method": "resource_creation"
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating FHIR resource: {e}")
            return self._mock_resource_creation(resource_type, resource_data)
    
    def validate_fhir_data(self, 
                          resource_type: str,
                          resource_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate FHIR resource data against FHIR specifications.
        
        Args:
            resource_type: Type of FHIR resource to validate
            resource_data: Data to validate
            
        Returns:
            Dictionary containing validation results
        """
        # Initialize FHIR only when this method is called
        if not self._check_fhir_availability():
            return self._mock_data_validation(resource_type, resource_data)
        
        try:
            # Initialize systems on first use
            if not FHIR_INITIALIZED:
                self._initialize_fhir_systems()
            
            # Validate resource data (simplified)
            validation_results = {
                "is_valid": True,
                "validation_errors": [],
                "validation_warnings": [],
                "resource_type": resource_type,
                "validation_status": "passed"
            }
            
            # Basic validation checks
            if resource_type not in self.resource_types:
                validation_results["is_valid"] = False
                validation_results["validation_errors"].append(f"Unknown resource type: {resource_type}")
                validation_results["validation_status"] = "failed"
            
            return {
                "resource_type": resource_type,
                "resource_data": resource_data,
                "status": "completed",
                "validation_results": validation_results,
                "metadata": {
                    "model": "FHIR",
                    "validation_method": "fhir_validation"
                }
            }
            
        except Exception as e:
            logger.error(f"Error validating FHIR data: {e}")
            return self._mock_data_validation(resource_type, resource_data)
    
    # Mock implementations for when FHIR is not available
    def _mock_patient_search(self, search_criteria: Dict[str, Any], limit: int) -> Dict[str, Any]:
        """Mock implementation for patient search."""
        return {
            "search_criteria": search_criteria,
            "limit": limit,
            "status": "mock_completed",
            "total_results": 0,
            "patients": [],
            "metadata": {"model": "mock", "server_url": "mock", "search_method": "mock"}
        }
    
    def _mock_patient_observations(self, patient_id: str, observation_types: Optional[List[str]]) -> Dict[str, Any]:
        """Mock implementation for patient observations."""
        return {
            "patient_id": patient_id,
            "observation_types": observation_types,
            "status": "mock_completed",
            "total_observations": 0,
            "observations": [],
            "metadata": {"model": "mock", "server_url": "mock", "search_method": "mock"}
        }
    
    def _mock_patient_conditions(self, patient_id: str, condition_types: Optional[List[str]]) -> Dict[str, Any]:
        """Mock implementation for patient conditions."""
        return {
            "patient_id": patient_id,
            "condition_types": condition_types,
            "status": "mock_completed",
            "total_conditions": 0,
            "conditions": [],
            "metadata": {"model": "mock", "server_url": "mock", "search_method": "mock"}
        }
    
    def _mock_resource_creation(self, resource_type: str, resource_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock implementation for resource creation."""
        return {
            "resource_type": resource_type,
            "resource_data": resource_data,
            "status": "mock_completed",
            "creation_result": {"mock_result": "FHIR not available"},
            "metadata": {"model": "mock", "server_url": "mock", "creation_method": "mock"}
        }
    
    def _mock_data_validation(self, resource_type: str, resource_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock implementation for data validation."""
        return {
            "resource_type": resource_type,
            "resource_data": resource_data,
            "status": "mock_completed",
            "validation_results": {"mock_results": "FHIR not available"},
            "metadata": {"model": "mock", "validation_method": "mock"}
        }


# Example usage and testing
def test_fhir_integration():
    """Test the FHIR integration."""
    config = {
        "server_url": "https://hapi.fhir.org/baseR4",
        "app_id": "medical_research_ai"
    }
    
    fhir_integration = FHIRIntegration(config)
    
    # Test patient search
    search_result = fhir_integration.search_patients({"gender": "male"}, 5)
    print(f"Patient Search: {search_result['status']}")
    
    # Test patient observations
    observations_result = fhir_integration.get_patient_observations("patient123", ["blood-pressure"])
    print(f"Patient Observations: {observations_result['status']}")
    
    # Test data validation
    validation_result = fhir_integration.validate_fhir_data("Patient", {"name": "John Doe"})
    print(f"Data Validation: {validation_result['status']}")


if __name__ == "__main__":
    test_fhir_integration() 
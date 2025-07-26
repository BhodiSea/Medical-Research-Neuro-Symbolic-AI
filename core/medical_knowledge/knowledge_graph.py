#!/usr/bin/env python3
"""
Medical Knowledge Graph System
Integrates with Nucleoid for medical domain knowledge representation
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class MedicalEntity:
    """Represents a medical entity in the knowledge graph"""
    id: str
    name: str
    type: str  # anatomy, condition, symptom, treatment, drug, etc.
    properties: Dict[str, Any]
    relationships: List[Dict[str, str]]  # {relation_type, target_id, confidence}

@dataclass
class MedicalRelationship:
    """Represents a relationship between medical entities"""
    source_id: str
    target_id: str
    relation_type: str  # causes, treats, located_in, symptom_of, etc.
    confidence: float
    evidence: List[str]  # Supporting evidence or references

class MedicalOntology:
    """Medical domain ontology and classification system"""
    
    def __init__(self):
        self.entity_types = {
            "anatomy": {
                "heart", "brain", "liver", "kidney", "lung", "stomach",
                "blood_vessel", "nerve", "muscle", "bone"
            },
            "condition": {
                "myocardial_infarction", "hypertension", "diabetes", "pneumonia",
                "stroke", "heart_failure", "asthma", "copd"
            },
            "symptom": {
                "chest_pain", "shortness_of_breath", "fatigue", "nausea",
                "headache", "fever", "cough", "dizziness"
            },
            "treatment": {
                "surgery", "medication", "therapy", "lifestyle_change",
                "monitoring", "procedure"
            },
            "drug": {
                "aspirin", "metformin", "lisinopril", "atorvastatin",
                "insulin", "albuterol", "warfarin"
            }
        }
        
        self.relation_types = {
            "anatomical": ["located_in", "part_of", "connected_to"],
            "pathological": ["causes", "leads_to", "risk_factor_for"],
            "clinical": ["symptom_of", "sign_of", "indicates"],
            "therapeutic": ["treats", "prevents", "contraindicates"],
            "pharmacological": ["acts_on", "metabolized_by", "interacts_with"]
        }

class MedicalKnowledgeGraph:
    """Core medical knowledge graph implementation"""
    
    def __init__(self):
        self.entities: Dict[str, MedicalEntity] = {}
        self.relationships: List[MedicalRelationship] = []
        self.ontology = MedicalOntology()
        self.logger = logging.getLogger(__name__)
        self._initialize_base_knowledge()
    
    def _initialize_base_knowledge(self):
        """Initialize with fundamental medical knowledge"""
        self.logger.info("Initializing medical knowledge graph with base knowledge...")
        
        # Core cardiovascular system
        self._add_cardiovascular_knowledge()
        # Core respiratory system  
        self._add_respiratory_knowledge()
        # Common conditions and treatments
        self._add_common_conditions()
        
        self.logger.info(f"Knowledge graph initialized with {len(self.entities)} entities and {len(self.relationships)} relationships")
    
    def _add_cardiovascular_knowledge(self):
        """Add cardiovascular system knowledge"""
        # Heart anatomy
        heart = MedicalEntity(
            id="heart",
            name="Heart",
            type="anatomy", 
            properties={
                "system": "cardiovascular",
                "chambers": 4,
                "function": "pump_blood",
                "weight_grams": "250-350"
            },
            relationships=[]
        )
        self.add_entity(heart)
        
        # Myocardial infarction
        mi = MedicalEntity(
            id="myocardial_infarction",
            name="Myocardial Infarction",
            type="condition",
            properties={
                "severity": "high",
                "emergency": True,
                "mortality_risk": "high",
                "pathophysiology": "coronary_artery_occlusion"
            },
            relationships=[]
        )
        self.add_entity(mi)
        
        # Chest pain symptom
        chest_pain = MedicalEntity(
            id="chest_pain",
            name="Chest Pain", 
            type="symptom",
            properties={
                "location": "chest",
                "quality": ["sharp", "crushing", "burning"],
                "severity_scale": "1-10"
            },
            relationships=[]
        )
        self.add_entity(chest_pain)
        
        # Relationships
        self.add_relationship("myocardial_infarction", "heart", "affects", 0.95, ["clinical_evidence"])
        self.add_relationship("myocardial_infarction", "chest_pain", "causes", 0.85, ["symptom_studies"])
    
    def _add_respiratory_knowledge(self):
        """Add respiratory system knowledge"""
        # Lungs
        lungs = MedicalEntity(
            id="lungs",
            name="Lungs",
            type="anatomy",
            properties={
                "system": "respiratory", 
                "lobes": 5,
                "function": "gas_exchange",
                "capacity_ml": "6000"
            },
            relationships=[]
        )
        self.add_entity(lungs)
        
        # Shortness of breath
        sob = MedicalEntity(
            id="shortness_of_breath",
            name="Shortness of Breath",
            type="symptom",
            properties={
                "medical_term": "dyspnea",
                "severity": ["mild", "moderate", "severe"],
                "triggers": ["exertion", "rest", "positional"]
            },
            relationships=[]
        )
        self.add_entity(sob)
    
    def _add_common_conditions(self):
        """Add common medical conditions and treatments"""
        # Aspirin
        aspirin = MedicalEntity(
            id="aspirin",
            name="Aspirin",
            type="drug",
            properties={
                "class": "nsaid",
                "mechanism": "cox_inhibition", 
                "indications": ["pain", "inflammation", "cardioprotection"],
                "dose_mg": "81-325"
            },
            relationships=[]
        )
        self.add_entity(aspirin)
        
        # Treatment relationship
        self.add_relationship("aspirin", "myocardial_infarction", "treats", 0.80, ["clinical_trials"])
    
    def add_entity(self, entity: MedicalEntity):
        """Add an entity to the knowledge graph"""
        self.entities[entity.id] = entity
        self.logger.debug(f"Added entity: {entity.name} ({entity.type})")
    
    def add_relationship(self, source_id: str, target_id: str, relation_type: str, 
                        confidence: float, evidence: List[str]):
        """Add a relationship between entities"""
        if source_id not in self.entities or target_id not in self.entities:
            self.logger.warning(f"Cannot add relationship: entity not found")
            return
        
        relationship = MedicalRelationship(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            confidence=confidence,
            evidence=evidence
        )
        self.relationships.append(relationship)
        
        # Update entity relationships
        self.entities[source_id].relationships.append({
            "relation_type": relation_type,
            "target_id": target_id,
            "confidence": confidence
        })
        
        self.logger.debug(f"Added relationship: {source_id} --{relation_type}--> {target_id}")
    
    def query_entity(self, entity_id: str) -> Optional[MedicalEntity]:
        """Query for a specific entity"""
        return self.entities.get(entity_id)
    
    def query_relationships(self, entity_id: str, relation_type: Optional[str] = None) -> List[MedicalRelationship]:
        """Query relationships for an entity"""
        relationships = [r for r in self.relationships if r.source_id == entity_id or r.target_id == entity_id]
        
        if relation_type:
            relationships = [r for r in relationships if r.relation_type == relation_type]
        
        return relationships
    
    def find_related_entities(self, entity_id: str, max_confidence: float = 1.0, 
                            min_confidence: float = 0.0) -> List[Tuple[MedicalEntity, float]]:
        """Find entities related to a given entity"""
        related = []
        
        for relationship in self.relationships:
            if relationship.source_id == entity_id:
                target_entity = self.entities.get(relationship.target_id)
                if target_entity and min_confidence <= relationship.confidence <= max_confidence:
                    related.append((target_entity, relationship.confidence))
            elif relationship.target_id == entity_id:
                source_entity = self.entities.get(relationship.source_id)
                if source_entity and min_confidence <= relationship.confidence <= max_confidence:
                    related.append((source_entity, relationship.confidence))
        
        # Sort by confidence
        related.sort(key=lambda x: x[1], reverse=True)
        return related
    
    def semantic_search(self, query: str, entity_types: Optional[List[str]] = None) -> List[MedicalEntity]:
        """Perform semantic search on the knowledge graph"""
        query_lower = query.lower()
        results = []
        
        for entity in self.entities.values():
            if entity_types and entity.type not in entity_types:
                continue
            
            # Simple text matching - could be enhanced with embeddings
            if (query_lower in entity.name.lower() or 
                any(query_lower in str(prop).lower() for prop in entity.properties.values())):
                results.append(entity)
        
        return results
    
    def get_differential_diagnosis(self, symptoms: List[str]) -> List[Tuple[MedicalEntity, float]]:
        """Get possible diagnoses based on symptoms"""
        symptom_entities = []
        
        # Find symptom entities
        for symptom in symptoms:
            entities = self.semantic_search(symptom, ["symptom"])
            symptom_entities.extend(entities)
        
        # Find conditions associated with these symptoms
        condition_scores = {}
        
        for symptom_entity in symptom_entities:
            related = self.find_related_entities(symptom_entity.id)
            for entity, confidence in related:
                if entity.type == "condition":
                    if entity.id not in condition_scores:
                        condition_scores[entity.id] = 0
                    condition_scores[entity.id] += confidence
        
        # Convert to results
        results = []
        for condition_id, score in condition_scores.items():
            condition = self.entities[condition_id]
            normalized_score = min(score / len(symptom_entities), 1.0)
            results.append((condition, normalized_score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def get_treatment_options(self, condition_id: str) -> List[Tuple[MedicalEntity, float]]:
        """Get treatment options for a condition"""
        treatments = []
        
        for relationship in self.relationships:
            if (relationship.target_id == condition_id and 
                relationship.relation_type == "treats"):
                treatment = self.entities.get(relationship.source_id)
                if treatment:
                    treatments.append((treatment, relationship.confidence))
        
        treatments.sort(key=lambda x: x[1], reverse=True)
        return treatments
    
    def export_knowledge(self) -> Dict[str, Any]:
        """Export knowledge graph to dictionary"""
        return {
            "entities": {eid: asdict(entity) for eid, entity in self.entities.items()},
            "relationships": [asdict(rel) for rel in self.relationships],
            "statistics": {
                "total_entities": len(self.entities),
                "total_relationships": len(self.relationships),
                "entity_types": {etype: len([e for e in self.entities.values() if e.type == etype]) 
                               for etype in self.ontology.entity_types.keys()}
            }
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get knowledge graph system status"""
        return {
            "initialized": True,
            "total_entities": len(self.entities),
            "total_relationships": len(self.relationships),
            "ontology_loaded": True,
            "ready_for_queries": True
        }

def create_medical_knowledge_graph() -> MedicalKnowledgeGraph:
    """Factory function to create and initialize medical knowledge graph"""
    logger.info("Creating medical knowledge graph...")
    kg = MedicalKnowledgeGraph()
    logger.info("Medical knowledge graph created successfully")
    return kg

# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create knowledge graph
    kg = create_medical_knowledge_graph()
    
    # Test queries
    print("🔬 Medical Knowledge Graph Demo")
    print("=" * 40)
    
    # Query for chest pain
    print("\n1. Query for chest pain:")
    chest_pain = kg.query_entity("chest_pain")
    if chest_pain:
        print(f"   Found: {chest_pain.name} ({chest_pain.type})")
        print(f"   Properties: {chest_pain.properties}")
    
    # Find related entities
    print("\n2. Entities related to myocardial infarction:")
    related = kg.find_related_entities("myocardial_infarction")
    for entity, confidence in related:
        print(f"   {entity.name} (confidence: {confidence:.2f})")
    
    # Differential diagnosis
    print("\n3. Differential diagnosis for ['chest pain', 'shortness of breath']:")
    diagnoses = kg.get_differential_diagnosis(["chest pain", "shortness of breath"])
    for condition, score in diagnoses:
        print(f"   {condition.name}: {score:.2f}")
    
    # Treatment options
    print("\n4. Treatment options for myocardial infarction:")
    treatments = kg.get_treatment_options("myocardial_infarction")
    for treatment, confidence in treatments:
        print(f"   {treatment.name}: {confidence:.2f}")
    
    print(f"\n📊 System Status: {kg.get_system_status()}") 
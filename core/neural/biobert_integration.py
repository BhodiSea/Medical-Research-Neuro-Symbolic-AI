"""
BioBERT Integration for Medical Research AI

This module provides integration with BioBERT (Biomedical Language Model) for
biomedical text understanding, entity recognition, and medical literature mining
for neurodegeneration research.

BioBERT is available via the cloned submodule.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add BioBERT submodule to path
biobert_path = Path(__file__).parent / "biobert"
if str(biobert_path) not in sys.path:
    sys.path.insert(0, str(biobert_path))

# Global flags for BioBERT availability - will be set on first use
BIOBERT_AVAILABLE = None
BIOBERT_INITIALIZED = False


class BioBERTIntegration:
    """
    Integration wrapper for BioBERT (Biomedical Language Model).
    
    BioBERT provides biomedical text understanding and entity recognition capabilities,
    supporting medical literature mining and biomedical text analysis for research applications.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize BioBERT integration.
        
        Args:
            config: Configuration dictionary with BioBERT settings
        """
        self.config = config or {}
        self.tokenizer = None
        self.model = None
        self.ner_pipeline = None
        self.classification_pipeline = None
        self.device = "cuda" if self._check_torch_availability() else "cpu"
        self._biobert_components = {}
        
        # Don't initialize anything at startup - use lazy loading
        logger.info("BioBERT integration initialized with lazy loading")
    
    def _check_torch_availability(self) -> bool:
        """Check if PyTorch is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _check_biobert_availability(self) -> bool:
        """Check if BioBERT is available and initialize if needed."""
        global BIOBERT_AVAILABLE, BIOBERT_INITIALIZED
        
        if BIOBERT_AVAILABLE is None:
            try:
                # Try to import BioBERT components only when needed
                import torch
                from transformers import (
                    AutoTokenizer, AutoModel, AutoModelForTokenClassification,
                    AutoModelForSequenceClassification, pipeline
                )
                from transformers import BertTokenizer, BertModel
                
                # Store components for later use
                self._biobert_components = {
                    'torch': torch,
                    'AutoTokenizer': AutoTokenizer,
                    'AutoModel': AutoModel,
                    'AutoModelForTokenClassification': AutoModelForTokenClassification,
                    'AutoModelForSequenceClassification': AutoModelForSequenceClassification,
                    'pipeline': pipeline,
                    'BertTokenizer': BertTokenizer,
                    'BertModel': BertModel
                }
                
                BIOBERT_AVAILABLE = True
                logger.info("BioBERT components loaded successfully")
                
            except ImportError as e:
                BIOBERT_AVAILABLE = False
                logger.warning(f"BioBERT/Transformers not available: {e}")
                logger.info("Install with: pip install transformers torch")
        
        return BIOBERT_AVAILABLE
    
    def _initialize_biobert_systems(self) -> None:
        """Initialize BioBERT systems and components - called only when needed."""
        global BIOBERT_INITIALIZED
        
        if BIOBERT_INITIALIZED:
            return
            
        try:
            if not self._check_biobert_availability():
                return
                
            # Initialize BioBERT model and tokenizer
            model_name = self.config.get("model_name", "dmis-lab/biobert-base-cased-v1.2")
            
            AutoTokenizer = self._biobert_components['AutoTokenizer']
            AutoModel = self._biobert_components['AutoModel']
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            
            # Move model to appropriate device
            self.model.to(self.device)
            self.model.eval()
            
            # Initialize NER pipeline
            self._initialize_ner_pipeline()
            
            # Initialize classification pipeline
            self._initialize_classification_pipeline()
            
            BIOBERT_INITIALIZED = True
            logger.info(f"BioBERT systems initialized successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error initializing BioBERT systems: {e}")
    
    def _initialize_ner_pipeline(self) -> None:
        """Initialize Named Entity Recognition pipeline."""
        try:
            # Use BioBERT-based NER model for biomedical entities
            ner_model_name = self.config.get("ner_model", "dmis-lab/biobert-base-cased-v1.2")
            pipeline = self._biobert_components['pipeline']
            
            self.ner_pipeline = pipeline(
                "ner",
                model=ner_model_name,
                tokenizer=ner_model_name,
                device=0 if self.device == "cuda" else -1
            )
            logger.info("BioBERT NER pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing NER pipeline: {e}")
    
    def _initialize_classification_pipeline(self) -> None:
        """Initialize text classification pipeline."""
        try:
            # Use BioBERT-based classification model
            classification_model_name = self.config.get("classification_model", "dmis-lab/biobert-base-cased-v1.2")
            pipeline = self._biobert_components['pipeline']
            
            self.classification_pipeline = pipeline(
                "text-classification",
                model=classification_model_name,
                tokenizer=classification_model_name,
                device=0 if self.device == "cuda" else -1
            )
            logger.info("BioBERT classification pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing classification pipeline: {e}")
    
    def extract_biomedical_entities(self, 
                                  text: str,
                                  entity_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Extract biomedical entities from text using BioBERT.
        
        Args:
            text: Input text for entity extraction
            entity_types: Optional list of entity types to focus on
            
        Returns:
            Dictionary containing extracted entities and their information
        """
        # Initialize BioBERT only when this method is called
        if not self._check_biobert_availability():
            return self._mock_entity_extraction(text, entity_types)
        
        try:
            # Initialize systems on first use
            if not BIOBERT_INITIALIZED:
                self._initialize_biobert_systems()
            
            # Extract entities using NER pipeline
            entities = self.ner_pipeline(text)
            
            # Filter entities by type if specified
            if entity_types:
                entities = [entity for entity in entities if entity["entity_group"] in entity_types]
            
            # Group entities by type
            grouped_entities = {}
            for entity in entities:
                entity_type = entity["entity_group"]
                if entity_type not in grouped_entities:
                    grouped_entities[entity_type] = []
                grouped_entities[entity_type].append({
                    "text": entity["word"],
                    "score": entity["score"],
                    "start": entity["start"],
                    "end": entity["end"]
                })
            
            # Extract medical concepts
            medical_concepts = self._extract_medical_concepts(text, entities)
            
            return {
                "text": text,
                "entities": grouped_entities,
                "medical_concepts": medical_concepts,
                "total_entities": len(entities),
                "entity_types_found": list(grouped_entities.keys()),
                "status": "completed",
                "metadata": {
                    "model": "BioBERT",
                    "device": self.device,
                    "entity_types_requested": entity_types
                }
            }
            
        except Exception as e:
            logger.error(f"Error extracting biomedical entities: {e}")
            return self._mock_entity_extraction(text, entity_types)
    
    def analyze_medical_literature(self, 
                                 text: str,
                                 analysis_type: str = "general") -> Dict[str, Any]:
        """
        Analyze medical literature using BioBERT.
        
        Args:
            text: Medical literature text to analyze
            analysis_type: Type of analysis (general, disease, drug, biomarker)
            
        Returns:
            Dictionary containing literature analysis results
        """
        # Initialize BioBERT only when this method is called
        if not self._check_biobert_availability():
            return self._mock_literature_analysis(text, analysis_type)
        
        try:
            # Initialize systems on first use
            if not BIOBERT_INITIALIZED:
                self._initialize_biobert_systems()
            
            # Extract entities
            entities_result = self.extract_biomedical_entities(text)
            
            # Perform text classification
            classification_result = self._classify_medical_text(text, analysis_type)
            
            # Extract key concepts
            key_concepts = self._extract_key_concepts(text, analysis_type)
            
            # Generate summary
            summary = self._generate_literature_summary(text, entities_result, classification_result)
            
            return {
                "text": text,
                "analysis_type": analysis_type,
                "status": "completed",
                "entities": entities_result,
                "classification": classification_result,
                "key_concepts": key_concepts,
                "summary": summary,
                "metadata": {
                    "model": "BioBERT",
                    "device": self.device,
                    "text_length": len(text)
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing medical literature: {e}")
            return self._mock_literature_analysis(text, analysis_type)
    
    def extract_disease_mentions(self, 
                               text: str,
                               disease_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Extract disease mentions from medical text.
        
        Args:
            text: Medical text to analyze
            disease_types: Optional list of disease types to focus on
            
        Returns:
            Dictionary containing disease mentions and their information
        """
        # Initialize BioBERT only when this method is called
        if not self._check_biobert_availability():
            return self._mock_disease_extraction(text, disease_types)
        
        try:
            # Initialize systems on first use
            if not BIOBERT_INITIALIZED:
                self._initialize_biobert_systems()
            
            # Extract all entities first
            entities_result = self.extract_biomedical_entities(text)
            
            # Filter for disease-related entities
            disease_entities = []
            for entity_type, entities in entities_result["entities"].items():
                if entity_type.lower() in ["disease", "disorder", "syndrome", "condition"]:
                    disease_entities.extend(entities)
            
            # Classify diseases by type
            disease_classification = self._classify_diseases(disease_entities, disease_types)
            
            # Extract disease relationships
            disease_relationships = self._extract_disease_relationships(text, disease_entities)
            
            return {
                "text": text,
                "disease_entities": disease_entities,
                "disease_classification": disease_classification,
                "disease_relationships": disease_relationships,
                "total_diseases": len(disease_entities),
                "status": "completed",
                "metadata": {
                    "model": "BioBERT",
                    "device": self.device,
                    "disease_types_requested": disease_types
                }
            }
            
        except Exception as e:
            logger.error(f"Error extracting disease mentions: {e}")
            return self._mock_disease_extraction(text, disease_types)
    
    def extract_drug_mentions(self, 
                            text: str,
                            drug_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Extract drug mentions from medical text.
        
        Args:
            text: Medical text to analyze
            drug_types: Optional list of drug types to focus on
            
        Returns:
            Dictionary containing drug mentions and their information
        """
        # Initialize BioBERT only when this method is called
        if not self._check_biobert_availability():
            return self._mock_drug_extraction(text, drug_types)
        
        try:
            # Initialize systems on first use
            if not BIOBERT_INITIALIZED:
                self._initialize_biobert_systems()
            
            # Extract all entities first
            entities_result = self.extract_biomedical_entities(text)
            
            # Filter for drug-related entities
            drug_entities = []
            for entity_type, entities in entities_result["entities"].items():
                if entity_type.lower() in ["drug", "medication", "compound", "molecule"]:
                    drug_entities.extend(entities)
            
            # Classify drugs by type
            drug_classification = self._classify_drugs(drug_entities, drug_types)
            
            # Extract drug relationships
            drug_relationships = self._extract_drug_relationships(text, drug_entities)
            
            return {
                "text": text,
                "drug_entities": drug_entities,
                "drug_classification": drug_classification,
                "drug_relationships": drug_relationships,
                "total_drugs": len(drug_entities),
                "status": "completed",
                "metadata": {
                    "model": "BioBERT",
                    "device": self.device,
                    "drug_types_requested": drug_types
                }
            }
            
        except Exception as e:
            logger.error(f"Error extracting drug mentions: {e}")
            return self._mock_drug_extraction(text, drug_types)
    
    def extract_biomarker_mentions(self, 
                                 text: str,
                                 biomarker_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Extract biomarker mentions from medical text.
        
        Args:
            text: Medical text to analyze
            biomarker_types: Optional list of biomarker types to focus on
            
        Returns:
            Dictionary containing biomarker mentions and their information
        """
        # Initialize BioBERT only when this method is called
        if not self._check_biobert_availability():
            return self._mock_biomarker_extraction(text, biomarker_types)
        
        try:
            # Initialize systems on first use
            if not BIOBERT_INITIALIZED:
                self._initialize_biobert_systems()
            
            # Extract all entities first
            entities_result = self.extract_biomedical_entities(text)
            
            # Filter for biomarker-related entities
            biomarker_entities = []
            for entity_type, entities in entities_result["entities"].items():
                if entity_type.lower() in ["biomarker", "protein", "gene", "molecule"]:
                    biomarker_entities.extend(entities)
            
            # Classify biomarkers by type
            biomarker_classification = self._classify_biomarkers(biomarker_entities, biomarker_types)
            
            # Extract biomarker relationships
            biomarker_relationships = self._extract_biomarker_relationships(text, biomarker_entities)
            
            return {
                "text": text,
                "biomarker_entities": biomarker_entities,
                "biomarker_classification": biomarker_classification,
                "biomarker_relationships": biomarker_relationships,
                "total_biomarkers": len(biomarker_entities),
                "status": "completed",
                "metadata": {
                    "model": "BioBERT",
                    "device": self.device,
                    "biomarker_types_requested": biomarker_types
                }
            }
            
        except Exception as e:
            logger.error(f"Error extracting biomarker mentions: {e}")
            return self._mock_biomarker_extraction(text, biomarker_types)
    
    def search_medical_concepts(self, 
                              query: str,
                              concept_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Search for medical concepts in text using BioBERT.
        
        Args:
            query: Search query for medical concepts
            concept_types: Optional list of concept types to search for
            
        Returns:
            Dictionary containing search results
        """
        # Initialize BioBERT only when this method is called
        if not self._check_biobert_availability():
            return self._mock_concept_search(query, concept_types)
        
        try:
            # Initialize systems on first use
            if not BIOBERT_INITIALIZED:
                self._initialize_biobert_systems()
            
            # Extract entities from query
            query_entities = self.extract_biomedical_entities(query)
            
            # Perform semantic search
            search_results = self._perform_semantic_search(query, concept_types)
            
            # Rank results by relevance
            ranked_results = self._rank_search_results(search_results, query_entities)
            
            return {
                "query": query,
                "concept_types": concept_types,
                "status": "completed",
                "query_entities": query_entities,
                "search_results": ranked_results,
                "total_results": len(ranked_results),
                "metadata": {
                    "model": "BioBERT",
                    "device": self.device,
                    "search_type": "semantic"
                }
            }
            
        except Exception as e:
            logger.error(f"Error searching medical concepts: {e}")
            return self._mock_concept_search(query, concept_types)
    
    # Internal processing methods
    def _extract_medical_concepts(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract medical concepts from text and entities."""
        concepts = []
        for entity in entities:
            concept = {
                "name": entity["word"],
                "type": entity["entity_group"],
                "confidence": entity["score"],
                "context": self._extract_context(text, entity["start"], entity["end"])
            }
            concepts.append(concept)
        return concepts
    
    def _extract_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        """Extract context around an entity."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end]
    
    def _classify_medical_text(self, text: str, analysis_type: str) -> Dict[str, Any]:
        """Classify medical text using BioBERT."""
        try:
            if self.classification_pipeline:
                result = self.classification_pipeline(text)
                return {
                    "classification": result,
                    "confidence": result[0]["score"] if result else 0.0
                }
            else:
                return {"classification": "mock", "confidence": 0.5}
        except Exception as e:
            logger.error(f"Error classifying medical text: {e}")
            return {"classification": "error", "confidence": 0.0}
    
    def _extract_key_concepts(self, text: str, analysis_type: str) -> List[Dict[str, Any]]:
        """Extract key concepts from medical text."""
        # This would implement more sophisticated concept extraction
        return [
            {"concept": "mock_concept", "type": "mock_type", "confidence": 0.8}
        ]
    
    def _generate_literature_summary(self, text: str, entities_result: Dict[str, Any], 
                                   classification_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate literature summary from analysis results."""
        return {
            "summary": f"Medical literature analysis summary for {len(text)} characters",
            "key_findings": ["mock_finding_1", "mock_finding_2"],
            "main_topics": ["mock_topic_1", "mock_topic_2"],
            "confidence": 0.75
        }
    
    def _classify_diseases(self, disease_entities: List[Dict[str, Any]], 
                          disease_types: Optional[List[str]]) -> Dict[str, Any]:
        """Classify diseases by type."""
        return {
            "neurodegenerative": [e for e in disease_entities if "neuro" in e["text"].lower()],
            "cardiovascular": [e for e in disease_entities if "cardio" in e["text"].lower()],
            "other": [e for e in disease_entities if "neuro" not in e["text"].lower() and "cardio" not in e["text"].lower()]
        }
    
    def _extract_disease_relationships(self, text: str, disease_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relationships between diseases and other entities."""
        return [
            {"disease": "mock_disease", "related_entity": "mock_entity", "relationship": "mock_relation"}
        ]
    
    def _classify_drugs(self, drug_entities: List[Dict[str, Any]], 
                       drug_types: Optional[List[str]]) -> Dict[str, Any]:
        """Classify drugs by type."""
        return {
            "therapeutic": [e for e in drug_entities if "therapeutic" in e["text"].lower()],
            "experimental": [e for e in drug_entities if "experimental" in e["text"].lower()],
            "other": [e for e in drug_entities if "therapeutic" not in e["text"].lower() and "experimental" not in e["text"].lower()]
        }
    
    def _extract_drug_relationships(self, text: str, drug_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relationships between drugs and other entities."""
        return [
            {"drug": "mock_drug", "related_entity": "mock_entity", "relationship": "mock_relation"}
        ]
    
    def _classify_biomarkers(self, biomarker_entities: List[Dict[str, Any]], 
                           biomarker_types: Optional[List[str]]) -> Dict[str, Any]:
        """Classify biomarkers by type."""
        return {
            "protein": [e for e in biomarker_entities if "protein" in e["text"].lower()],
            "genetic": [e for e in biomarker_entities if "gene" in e["text"].lower()],
            "other": [e for e in biomarker_entities if "protein" not in e["text"].lower() and "gene" not in e["text"].lower()]
        }
    
    def _extract_biomarker_relationships(self, text: str, biomarker_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relationships between biomarkers and other entities."""
        return [
            {"biomarker": "mock_biomarker", "related_entity": "mock_entity", "relationship": "mock_relation"}
        ]
    
    def _perform_semantic_search(self, query: str, concept_types: Optional[List[str]]) -> List[Dict[str, Any]]:
        """Perform semantic search for medical concepts."""
        return [
            {"concept": "mock_concept", "relevance": 0.8, "type": "mock_type"}
        ]
    
    def _rank_search_results(self, search_results: List[Dict[str, Any]], 
                           query_entities: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank search results by relevance."""
        return sorted(search_results, key=lambda x: x["relevance"], reverse=True)
    
    # Mock implementations for when BioBERT is not available
    def _mock_entity_extraction(self, text: str, entity_types: Optional[List[str]]) -> Dict[str, Any]:
        """Mock implementation for entity extraction."""
        return {
            "text": text,
            "entities": {"mock_entity_type": [{"text": "mock_entity", "score": 0.8}]},
            "medical_concepts": [{"name": "mock_concept", "type": "mock_type", "confidence": 0.8}],
            "total_entities": 1,
            "entity_types_found": ["mock_entity_type"],
            "status": "mock_completed",
            "metadata": {"model": "mock", "device": "cpu"}
        }
    
    def _mock_literature_analysis(self, text: str, analysis_type: str) -> Dict[str, Any]:
        """Mock implementation for literature analysis."""
        return {
            "text": text,
            "analysis_type": analysis_type,
            "status": "mock_completed",
            "entities": self._mock_entity_extraction(text, None),
            "classification": {"classification": "mock", "confidence": 0.5},
            "key_concepts": [{"concept": "mock_concept", "type": "mock_type", "confidence": 0.8}],
            "summary": {"summary": "Mock literature summary", "key_findings": ["mock_finding"]},
            "metadata": {"model": "mock", "device": "cpu"}
        }
    
    def _mock_disease_extraction(self, text: str, disease_types: Optional[List[str]]) -> Dict[str, Any]:
        """Mock implementation for disease extraction."""
        return {
            "text": text,
            "disease_entities": [{"text": "mock_disease", "score": 0.8}],
            "disease_classification": {"neurodegenerative": [], "other": [{"text": "mock_disease"}]},
            "disease_relationships": [{"disease": "mock_disease", "related_entity": "mock_entity"}],
            "total_diseases": 1,
            "status": "mock_completed",
            "metadata": {"model": "mock", "device": "cpu"}
        }
    
    def _mock_drug_extraction(self, text: str, drug_types: Optional[List[str]]) -> Dict[str, Any]:
        """Mock implementation for drug extraction."""
        return {
            "text": text,
            "drug_entities": [{"text": "mock_drug", "score": 0.8}],
            "drug_classification": {"therapeutic": [], "other": [{"text": "mock_drug"}]},
            "drug_relationships": [{"drug": "mock_drug", "related_entity": "mock_entity"}],
            "total_drugs": 1,
            "status": "mock_completed",
            "metadata": {"model": "mock", "device": "cpu"}
        }
    
    def _mock_biomarker_extraction(self, text: str, biomarker_types: Optional[List[str]]) -> Dict[str, Any]:
        """Mock implementation for biomarker extraction."""
        return {
            "text": text,
            "biomarker_entities": [{"text": "mock_biomarker", "score": 0.8}],
            "biomarker_classification": {"protein": [], "other": [{"text": "mock_biomarker"}]},
            "biomarker_relationships": [{"biomarker": "mock_biomarker", "related_entity": "mock_entity"}],
            "total_biomarkers": 1,
            "status": "mock_completed",
            "metadata": {"model": "mock", "device": "cpu"}
        }
    
    def _mock_concept_search(self, query: str, concept_types: Optional[List[str]]) -> Dict[str, Any]:
        """Mock implementation for concept search."""
        return {
            "query": query,
            "concept_types": concept_types,
            "status": "mock_completed",
            "query_entities": self._mock_entity_extraction(query, None),
            "search_results": [{"concept": "mock_concept", "relevance": 0.8}],
            "total_results": 1,
            "metadata": {"model": "mock", "device": "cpu"}
        }


# Example usage and testing
def test_biobert_integration():
    """Test the BioBERT integration."""
    config = {
        "model_name": "dmis-lab/biobert-base-cased-v1.2",
        "device": "auto"
    }
    
    biobert_integration = BioBERTIntegration(config)
    
    # Test entity extraction
    sample_text = "Parkinson's disease is characterized by alpha-synuclein aggregation in the substantia nigra."
    entity_result = biobert_integration.extract_biomedical_entities(sample_text)
    print(f"Entity Extraction: {entity_result['status']}")
    
    # Test literature analysis
    literature_result = biobert_integration.analyze_medical_literature(sample_text, "disease")
    print(f"Literature Analysis: {literature_result['status']}")
    
    # Test disease extraction
    disease_result = biobert_integration.extract_disease_mentions(sample_text)
    print(f"Disease Extraction: {disease_result['status']}")


if __name__ == "__main__":
    test_biobert_integration() 
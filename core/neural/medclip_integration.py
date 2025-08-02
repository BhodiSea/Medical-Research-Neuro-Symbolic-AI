"""
MedCLIP Integration for Medical Research AI

This module provides integration with MedCLIP (Medical Vision-Language Model) for
medical image-text understanding and reasoning, supporting medical report generation
and image interpretation for neurodegeneration research.

MedCLIP is available via the cloned submodule.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add MedCLIP submodule to path
medclip_path = Path(__file__).parent / "medclip"
if str(medclip_path) not in sys.path:
    sys.path.insert(0, str(medclip_path))

# Try to import MedCLIP components
try:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import clip
    
    # Try to import MedCLIP-specific components
    try:
        from medclip import MedCLIPModel, MedCLIPProcessor
        MEDCLIP_AVAILABLE = True
    except ImportError:
        # Fallback to standard CLIP if MedCLIP not available
        MEDCLIP_AVAILABLE = False
        logger.warning("MedCLIP not available, falling back to standard CLIP")
        
except ImportError as e:
    logger.warning(f"MedCLIP/CLIP not available: {e}")
    logger.info("Install with: pip install torch torchvision clip")
    MEDCLIP_AVAILABLE = False


class MedCLIPIntegration:
    """
    Integration wrapper for MedCLIP (Medical Vision-Language Model).
    
    MedCLIP provides medical image-text understanding and reasoning capabilities,
    supporting medical report generation and image interpretation for research applications.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize MedCLIP integration.
        
        Args:
            config: Configuration dictionary with MedCLIP settings
        """
        self.config = config or {}
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if not MEDCLIP_AVAILABLE:
            logger.warning("MedCLIP integration running in mock mode")
        else:
            self._initialize_medclip_systems()
    
    def _initialize_medclip_systems(self) -> None:
        """Initialize MedCLIP systems and components."""
        try:
            # Initialize MedCLIP model and processor
            self.model = MedCLIPModel.from_pretrained("microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
            self.processor = MedCLIPProcessor.from_pretrained("microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
            
            # Move model to appropriate device
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"MedCLIP systems initialized successfully on {self.device}")
        except Exception as e:
            logger.error(f"Error initializing MedCLIP systems: {e}")
            # Fallback to standard CLIP
            self._initialize_standard_clip()
    
    def _initialize_standard_clip(self) -> None:
        """Initialize standard CLIP as fallback."""
        try:
            self.model, self.processor = clip.load("ViT-B/32", device=self.device)
            logger.info(f"Standard CLIP initialized as fallback on {self.device}")
        except Exception as e:
            logger.error(f"Error initializing standard CLIP: {e}")
    
    def analyze_medical_image(self, 
                            image_path: str,
                            query_text: Optional[str] = None,
                            analysis_type: str = "general") -> Dict[str, Any]:
        """
        Analyze medical image using MedCLIP vision-language understanding.
        
        Args:
            image_path: Path to the medical image file
            query_text: Optional text query for specific analysis
            analysis_type: Type of analysis (general, brain, chest, pathology)
            
        Returns:
            Dictionary containing analysis results
        """
        if not MEDCLIP_AVAILABLE:
            return self._mock_medical_analysis(image_path, query_text, analysis_type)
        
        try:
            # Load and preprocess the image
            image = Image.open(image_path).convert('RGB')
            
            # Prepare text queries based on analysis type
            if query_text:
                text_queries = [query_text]
            else:
                text_queries = self._get_default_queries(analysis_type)
            
            # Process image and text
            image_inputs = self.processor(images=image, return_tensors="pt", padding=True)
            text_inputs = self.processor(text=text_queries, return_tensors="pt", padding=True)
            
            # Move inputs to device
            image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
            
            # Get model predictions
            with torch.no_grad():
                image_features = self.model.get_image_features(**image_inputs)
                text_features = self.model.get_text_features(**text_inputs)
                
                # Calculate similarity scores
                similarity_scores = F.cosine_similarity(image_features, text_features)
                
                # Get predictions
                predictions = torch.softmax(similarity_scores, dim=-1)
            
            # Process results
            results = self._process_analysis_results(text_queries, predictions, analysis_type)
            
            return {
                "image_path": image_path,
                "query_text": query_text,
                "analysis_type": analysis_type,
                "status": "completed",
                "results": results,
                "metadata": {
                    "model": "MedCLIP" if "MedCLIP" in str(type(self.model)) else "CLIP",
                    "device": self.device,
                    "similarity_scores": similarity_scores.cpu().numpy().tolist()
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing medical image: {e}")
            return self._mock_medical_analysis(image_path, query_text, analysis_type)
    
    def generate_medical_report(self, 
                              image_path: str,
                              report_type: str = "brain_mri") -> Dict[str, Any]:
        """
        Generate medical report from image using MedCLIP.
        
        Args:
            image_path: Path to the medical image
            report_type: Type of report to generate (brain_mri, chest_xray, pathology)
            
        Returns:
            Dictionary containing generated report
        """
        if not MEDCLIP_AVAILABLE:
            return self._mock_medical_report(image_path, report_type)
        
        try:
            # Get relevant queries for report generation
            report_queries = self._get_report_queries(report_type)
            
            # Analyze image with multiple queries
            analysis_results = []
            for query in report_queries:
                result = self.analyze_medical_image(image_path, query, report_type)
                analysis_results.append(result)
            
            # Generate comprehensive report
            report = self._synthesize_medical_report(analysis_results, report_type)
            
            return {
                "image_path": image_path,
                "report_type": report_type,
                "status": "completed",
                "report": report,
                "analysis_results": analysis_results,
                "metadata": {
                    "model": "MedCLIP" if "MedCLIP" in str(type(self.model)) else "CLIP",
                    "queries_analyzed": len(report_queries)
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating medical report: {e}")
            return self._mock_medical_report(image_path, report_type)
    
    def compare_medical_images(self, 
                             image_paths: List[str],
                             comparison_type: str = "similarity") -> Dict[str, Any]:
        """
        Compare multiple medical images using MedCLIP.
        
        Args:
            image_paths: List of image file paths
            comparison_type: Type of comparison (similarity, differences, progression)
            
        Returns:
            Dictionary containing comparison results
        """
        if not MEDCLIP_AVAILABLE:
            return self._mock_image_comparison(image_paths, comparison_type)
        
        try:
            # Extract features for all images
            image_features = []
            for image_path in image_paths:
                image = Image.open(image_path).convert('RGB')
                image_inputs = self.processor(images=image, return_tensors="pt", padding=True)
                image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
                
                with torch.no_grad():
                    features = self.model.get_image_features(**image_inputs)
                    image_features.append(features.cpu())
            
            # Calculate similarity matrix
            similarity_matrix = self._calculate_similarity_matrix(image_features)
            
            # Analyze comparison results
            comparison_results = self._analyze_comparison_results(
                image_paths, similarity_matrix, comparison_type
            )
            
            return {
                "image_paths": image_paths,
                "comparison_type": comparison_type,
                "status": "completed",
                "similarity_matrix": similarity_matrix.tolist(),
                "comparison_results": comparison_results,
                "metadata": {
                    "model": "MedCLIP" if "MedCLIP" in str(type(self.model)) else "CLIP",
                    "num_images": len(image_paths)
                }
            }
            
        except Exception as e:
            logger.error(f"Error comparing medical images: {e}")
            return self._mock_image_comparison(image_paths, comparison_type)
    
    def detect_brain_abnormalities(self, 
                                 image_path: str,
                                 abnormality_type: str = "general") -> Dict[str, Any]:
        """
        Detect brain abnormalities using MedCLIP.
        
        Args:
            image_path: Path to the brain image
            abnormality_type: Type of abnormality to detect (tumor, stroke, atrophy)
            
        Returns:
            Dictionary containing abnormality detection results
        """
        if not MEDCLIP_AVAILABLE:
            return self._mock_abnormality_detection(image_path, abnormality_type)
        
        try:
            # Get abnormality-specific queries
            abnormality_queries = self._get_abnormality_queries(abnormality_type)
            
            # Analyze image for abnormalities
            abnormality_results = []
            for query in abnormality_queries:
                result = self.analyze_medical_image(image_path, query, "brain")
                abnormality_results.append(result)
            
            # Synthesize abnormality detection results
            detection_results = self._synthesize_abnormality_detection(
                abnormality_results, abnormality_type
            )
            
            return {
                "image_path": image_path,
                "abnormality_type": abnormality_type,
                "status": "completed",
                "detection_results": detection_results,
                "abnormality_analysis": abnormality_results,
                "metadata": {
                    "model": "MedCLIP" if "MedCLIP" in str(type(self.model)) else "CLIP",
                    "queries_analyzed": len(abnormality_queries)
                }
            }
            
        except Exception as e:
            logger.error(f"Error detecting brain abnormalities: {e}")
            return self._mock_abnormality_detection(image_path, abnormality_type)
    
    # Helper methods for query generation
    def _get_default_queries(self, analysis_type: str) -> List[str]:
        """Get default text queries for different analysis types."""
        queries = {
            "general": [
                "normal medical image",
                "abnormal medical image",
                "healthy tissue",
                "diseased tissue"
            ],
            "brain": [
                "normal brain MRI",
                "brain tumor",
                "brain hemorrhage",
                "brain atrophy",
                "normal brain structure"
            ],
            "chest": [
                "normal chest X-ray",
                "pneumonia",
                "lung cancer",
                "normal lung tissue"
            ],
            "pathology": [
                "normal tissue",
                "cancerous tissue",
                "inflammatory tissue",
                "normal cell structure"
            ]
        }
        return queries.get(analysis_type, queries["general"])
    
    def _get_report_queries(self, report_type: str) -> List[str]:
        """Get queries for medical report generation."""
        queries = {
            "brain_mri": [
                "brain structure normal",
                "brain tumor present",
                "brain atrophy visible",
                "brain hemorrhage detected",
                "normal brain anatomy"
            ],
            "chest_xray": [
                "normal chest X-ray",
                "pneumonia visible",
                "lung cancer detected",
                "normal lung tissue",
                "chest abnormality"
            ],
            "pathology": [
                "normal tissue sample",
                "cancerous cells present",
                "inflammatory response",
                "normal cell morphology",
                "tissue abnormality"
            ]
        }
        return queries.get(report_type, queries["brain_mri"])
    
    def _get_abnormality_queries(self, abnormality_type: str) -> List[str]:
        """Get queries for abnormality detection."""
        queries = {
            "tumor": [
                "brain tumor present",
                "tumor mass visible",
                "abnormal growth",
                "tumor tissue"
            ],
            "stroke": [
                "brain hemorrhage",
                "stroke damage",
                "ischemic injury",
                "brain infarct"
            ],
            "atrophy": [
                "brain atrophy",
                "tissue loss",
                "brain shrinkage",
                "degenerative changes"
            ],
            "general": [
                "brain abnormality",
                "pathological changes",
                "disease signs",
                "abnormal brain tissue"
            ]
        }
        return queries.get(abnormality_type, queries["general"])
    
    # Processing methods
    def _process_analysis_results(self, queries: List[str], predictions: torch.Tensor, 
                                analysis_type: str) -> Dict[str, Any]:
        """Process analysis results into structured format."""
        results = {
            "query_results": [],
            "top_predictions": [],
            "confidence_scores": []
        }
        
        for i, query in enumerate(queries):
            confidence = predictions[i].item()
            results["query_results"].append({
                "query": query,
                "confidence": confidence,
                "interpretation": self._interpret_confidence(confidence)
            })
            
            if confidence > 0.5:  # High confidence threshold
                results["top_predictions"].append(query)
                results["confidence_scores"].append(confidence)
        
        return results
    
    def _interpret_confidence(self, confidence: float) -> str:
        """Interpret confidence score."""
        if confidence > 0.8:
            return "high_confidence"
        elif confidence > 0.6:
            return "medium_confidence"
        elif confidence > 0.4:
            return "low_confidence"
        else:
            return "very_low_confidence"
    
    def _calculate_similarity_matrix(self, image_features: List[torch.Tensor]) -> torch.Tensor:
        """Calculate similarity matrix between image features."""
        features = torch.cat(image_features, dim=0)
        similarity_matrix = F.cosine_similarity(
            features.unsqueeze(1), features.unsqueeze(0), dim=2
        )
        return similarity_matrix
    
    def _analyze_comparison_results(self, image_paths: List[str], 
                                  similarity_matrix: torch.Tensor,
                                  comparison_type: str) -> Dict[str, Any]:
        """Analyze comparison results."""
        results = {
            "most_similar_pair": None,
            "least_similar_pair": None,
            "average_similarity": float(similarity_matrix.mean()),
            "similarity_distribution": similarity_matrix.flatten().tolist()
        }
        
        # Find most and least similar pairs
        n = len(image_paths)
        max_sim = -1
        min_sim = 2
        
        for i in range(n):
            for j in range(i+1, n):
                sim = similarity_matrix[i, j].item()
                if sim > max_sim:
                    max_sim = sim
                    results["most_similar_pair"] = (image_paths[i], image_paths[j], sim)
                if sim < min_sim:
                    min_sim = sim
                    results["least_similar_pair"] = (image_paths[i], image_paths[j], sim)
        
        return results
    
    def _synthesize_medical_report(self, analysis_results: List[Dict[str, Any]], 
                                 report_type: str) -> Dict[str, Any]:
        """Synthesize medical report from analysis results."""
        report = {
            "summary": f"Medical report for {report_type} analysis",
            "findings": [],
            "recommendations": [],
            "confidence_level": "medium"
        }
        
        # Extract findings from analysis results
        for result in analysis_results:
            if "results" in result and "top_predictions" in result["results"]:
                for prediction in result["results"]["top_predictions"]:
                    report["findings"].append(prediction)
        
        # Generate recommendations based on findings
        if "tumor" in str(report["findings"]).lower():
            report["recommendations"].append("Consider additional imaging for tumor characterization")
        if "atrophy" in str(report["findings"]).lower():
            report["recommendations"].append("Monitor for progressive atrophy patterns")
        
        return report
    
    def _synthesize_abnormality_detection(self, abnormality_results: List[Dict[str, Any]], 
                                        abnormality_type: str) -> Dict[str, Any]:
        """Synthesize abnormality detection results."""
        detection = {
            "abnormality_detected": False,
            "detection_confidence": 0.0,
            "detected_features": [],
            "severity_estimate": "none"
        }
        
        # Analyze results for abnormality detection
        total_confidence = 0
        feature_count = 0
        
        for result in abnormality_results:
            if "results" in result and "confidence_scores" in result["results"]:
                for score in result["results"]["confidence_scores"]:
                    total_confidence += score
                    feature_count += 1
                    if score > 0.6:
                        detection["detected_features"].append(result.get("query_text", "unknown"))
        
        if feature_count > 0:
            detection["detection_confidence"] = total_confidence / feature_count
            detection["abnormality_detected"] = detection["detection_confidence"] > 0.5
            
            # Estimate severity
            if detection["detection_confidence"] > 0.8:
                detection["severity_estimate"] = "high"
            elif detection["detection_confidence"] > 0.6:
                detection["severity_estimate"] = "medium"
            elif detection["detection_confidence"] > 0.4:
                detection["severity_estimate"] = "low"
        
        return detection
    
    # Mock implementations for when MedCLIP is not available
    def _mock_medical_analysis(self, image_path: str, query_text: Optional[str], 
                             analysis_type: str) -> Dict[str, Any]:
        """Mock implementation for medical image analysis."""
        return {
            "image_path": image_path,
            "query_text": query_text,
            "analysis_type": analysis_type,
            "status": "mock_completed",
            "results": {
                "query_results": [{"query": query_text or "mock_query", "confidence": 0.5, "interpretation": "medium_confidence"}],
                "top_predictions": ["mock_prediction"],
                "confidence_scores": [0.5]
            },
            "metadata": {"model": "mock", "device": "cpu"}
        }
    
    def _mock_medical_report(self, image_path: str, report_type: str) -> Dict[str, Any]:
        """Mock implementation for medical report generation."""
        return {
            "image_path": image_path,
            "report_type": report_type,
            "status": "mock_completed",
            "report": {
                "summary": f"Mock medical report for {report_type}",
                "findings": ["mock_finding"],
                "recommendations": ["mock_recommendation"],
                "confidence_level": "mock"
            },
            "metadata": {"model": "mock"}
        }
    
    def _mock_image_comparison(self, image_paths: List[str], comparison_type: str) -> Dict[str, Any]:
        """Mock implementation for image comparison."""
        return {
            "image_paths": image_paths,
            "comparison_type": comparison_type,
            "status": "mock_completed",
            "similarity_matrix": [[1.0, 0.5], [0.5, 1.0]],
            "comparison_results": {
                "most_similar_pair": (image_paths[0], image_paths[1], 0.5),
                "average_similarity": 0.75
            },
            "metadata": {"model": "mock"}
        }
    
    def _mock_abnormality_detection(self, image_path: str, abnormality_type: str) -> Dict[str, Any]:
        """Mock implementation for abnormality detection."""
        return {
            "image_path": image_path,
            "abnormality_type": abnormality_type,
            "status": "mock_completed",
            "detection_results": {
                "abnormality_detected": False,
                "detection_confidence": 0.0,
                "detected_features": [],
                "severity_estimate": "none"
            },
            "metadata": {"model": "mock"}
        }


# Example usage and testing
def test_medclip_integration():
    """Test the MedCLIP integration."""
    config = {
        "model_type": "MedCLIP",
        "device": "auto"
    }
    
    medclip_integration = MedCLIPIntegration(config)
    
    # Test medical image analysis
    analysis_result = medclip_integration.analyze_medical_image(
        "sample_brain_mri.jpg", 
        "brain tumor present", 
        "brain"
    )
    print(f"Medical Analysis: {analysis_result['status']}")
    
    # Test medical report generation
    report_result = medclip_integration.generate_medical_report(
        "sample_brain_mri.jpg", 
        "brain_mri"
    )
    print(f"Medical Report: {report_result['status']}")
    
    # Test abnormality detection
    abnormality_result = medclip_integration.detect_brain_abnormalities(
        "sample_brain_mri.jpg", 
        "tumor"
    )
    print(f"Abnormality Detection: {abnormality_result['status']}")


if __name__ == "__main__":
    test_medclip_integration() 
"""
Nilearn Integration Wrapper
Provides standardized interface for neuroimaging analysis in medical research
"""

import sys
import os
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import numpy as np
import pandas as pd

# Add Nilearn submodule to path
nilearn_path = Path(__file__).parent / "neurotrale"
if str(nilearn_path) not in sys.path:
    sys.path.insert(0, str(nilearn_path))

try:
    # Import Nilearn components when available
    import nilearn
    from nilearn import datasets, plotting, image, maskers, connectome
    from nilearn.input_data import NiftiMasker, MultiNiftiMasker
    from nilearn.connectome import ConnectivityMeasure
    from nilearn.decoding import Decoder
    from nilearn.regions import connected_regions
    from nilearn.surface import vol_to_surf
    NILEARN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Nilearn not available: {e}")
    NILEARN_AVAILABLE = False


class NilearnIntegration:
    """Integration wrapper for Nilearn neuroimaging analysis"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.atlases = {}
        self.maskers = {}
        self.connectivity_measures = {}
        self.decoders = {}
        
        if not NILEARN_AVAILABLE:
            print("Warning: Nilearn integration running in mock mode")
        else:
            self._initialize_nilearn()
    
    def _initialize_nilearn(self) -> None:
        """Initialize Nilearn components for medical neuroimaging analysis"""
        try:
            # Initialize brain atlases
            self._initialize_atlases()
            
            # Initialize maskers
            self._initialize_maskers()
            
            # Initialize connectivity measures
            self._initialize_connectivity_measures()
            
            # Initialize decoders
            self._initialize_decoders()
            
        except Exception as e:
            print(f"Error initializing Nilearn: {e}")
    
    def _initialize_atlases(self) -> None:
        """Initialize brain atlases for different analysis types"""
        try:
            # Harvard-Oxford atlas for cortical regions
            self.atlases['harvard_oxford'] = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
            
            # AAL atlas for automated anatomical labeling
            self.atlases['aal'] = datasets.fetch_atlas_aal()
            
            # Power atlas for functional connectivity
            self.atlases['power'] = datasets.fetch_coords_power_2011()
            
            # Schaefer atlas for parcellation
            self.atlases['schaefer'] = datasets.fetch_atlas_schaefer_2018(n_rois=100)
            
        except Exception as e:
            print(f"Error initializing atlases: {e}")
    
    def _initialize_maskers(self) -> None:
        """Initialize maskers for different brain regions"""
        try:
            # Whole brain masker
            self.maskers['whole_brain'] = NiftiMasker(
                smoothing_fwhm=6,
                standardize=True,
                detrend=True
            )
            
            # ROI-based masker
            if 'harvard_oxford' in self.atlases:
                self.maskers['roi'] = NiftiMasker(
                    mask_img=self.atlases['harvard_oxford'].maps,
                    smoothing_fwhm=6,
                    standardize=True
                )
            
        except Exception as e:
            print(f"Error initializing maskers: {e}")
    
    def _initialize_connectivity_measures(self) -> None:
        """Initialize connectivity measures for network analysis"""
        try:
            # Correlation-based connectivity
            self.connectivity_measures['correlation'] = ConnectivityMeasure(
                kind='correlation'
            )
            
            # Partial correlation
            self.connectivity_measures['partial_correlation'] = ConnectivityMeasure(
                kind='partial correlation'
            )
            
            # Tangent space embedding
            self.connectivity_measures['tangent'] = ConnectivityMeasure(
                kind='tangent'
            )
            
        except Exception as e:
            print(f"Error initializing connectivity measures: {e}")
    
    def _initialize_decoders(self) -> None:
        """Initialize decoders for pattern analysis"""
        try:
            # SVM decoder
            self.decoders['svm'] = Decoder(
                estimator='svc',
                cv=5,
                scoring='accuracy'
            )
            
            # Logistic regression decoder
            self.decoders['logistic'] = Decoder(
                estimator='logistic',
                cv=5,
                scoring='accuracy'
            )
            
        except Exception as e:
            print(f"Error initializing decoders: {e}")
    
    def analyze_brain_connectivity(self, functional_data: List[str], 
                                 atlas_name: str = "harvard_oxford",
                                 connectivity_type: str = "correlation") -> Dict[str, Any]:
        """Analyze brain connectivity patterns"""
        if not NILEARN_AVAILABLE:
            return self._mock_connectivity_analysis(functional_data, atlas_name, connectivity_type)
        
        try:
            # Load functional data
            func_imgs = [image.load_img(img) for img in functional_data]
            
            # Get atlas
            atlas = self.atlases.get(atlas_name)
            if not atlas:
                raise ValueError(f"Atlas {atlas_name} not found")
            
            # Create masker
            masker = NiftiMasker(
                mask_img=atlas.maps,
                smoothing_fwhm=6,
                standardize=True
            )
            
            # Extract time series
            time_series = masker.fit_transform(func_imgs)
            
            # Calculate connectivity
            connectivity_measure = self.connectivity_measures.get(connectivity_type)
            if not connectivity_measure:
                raise ValueError(f"Connectivity type {connectivity_type} not found")
            
            connectivity_matrix = connectivity_measure.fit_transform([time_series])[0]
            
            # Analyze connectivity patterns
            analysis_result = self._analyze_connectivity_patterns(connectivity_matrix, atlas)
            
            return {
                "connectivity_type": connectivity_type,
                "atlas_name": atlas_name,
                "connectivity_matrix": connectivity_matrix.tolist(),
                "analysis": analysis_result,
                "confidence": self._calculate_connectivity_confidence(connectivity_matrix)
            }
            
        except Exception as e:
            print(f"Error analyzing brain connectivity: {e}")
            return self._mock_connectivity_analysis(functional_data, atlas_name, connectivity_type)
    
    def _analyze_connectivity_patterns(self, connectivity_matrix: np.ndarray, 
                                     atlas: Any) -> Dict[str, Any]:
        """Analyze connectivity patterns for medical insights"""
        try:
            # Calculate network metrics
            network_metrics = {
                "mean_connectivity": np.mean(connectivity_matrix),
                "connectivity_std": np.std(connectivity_matrix),
                "max_connectivity": np.max(connectivity_matrix),
                "min_connectivity": np.min(connectivity_matrix)
            }
            
            # Identify highly connected regions
            threshold = np.percentile(connectivity_matrix, 90)
            high_connectivity = np.where(connectivity_matrix > threshold)
            
            # Extract region names if available
            region_names = []
            if hasattr(atlas, 'labels'):
                region_names = atlas.labels
            
            # Identify key connections
            key_connections = []
            for i, j in zip(high_connectivity[0], high_connectivity[1]):
                if i != j:  # Exclude self-connections
                    connection_strength = connectivity_matrix[i, j]
                    region_i = region_names[i] if i < len(region_names) else f"Region_{i}"
                    region_j = region_names[j] if j < len(region_names) else f"Region_{j}"
                    key_connections.append({
                        "region_1": region_i,
                        "region_2": region_j,
                        "strength": float(connection_strength)
                    })
            
            # Sort by connection strength
            key_connections.sort(key=lambda x: x["strength"], reverse=True)
            
            return {
                "network_metrics": network_metrics,
                "key_connections": key_connections[:10],  # Top 10 connections
                "high_connectivity_threshold": float(threshold)
            }
            
        except Exception as e:
            print(f"Error analyzing connectivity patterns: {e}")
            return {
                "network_metrics": {},
                "key_connections": [],
                "high_connectivity_threshold": 0.0
            }
    
    def decode_brain_patterns(self, functional_data: List[str], 
                            labels: List[str],
                            decoder_type: str = "svm") -> Dict[str, Any]:
        """Decode brain patterns for classification"""
        if not NILEARN_AVAILABLE:
            return self._mock_pattern_decoding(functional_data, labels, decoder_type)
        
        try:
            # Load functional data
            func_imgs = [image.load_img(img) for img in functional_data]
            
            # Get decoder
            decoder = self.decoders.get(decoder_type)
            if not decoder:
                raise ValueError(f"Decoder type {decoder_type} not found")
            
            # Fit and score decoder
            decoder.fit(func_imgs, labels)
            scores = decoder.cv_scores_
            
            # Get feature importance
            feature_importance = decoder.coef_ if hasattr(decoder, 'coef_') else None
            
            # Analyze decoding results
            analysis_result = self._analyze_decoding_results(scores, feature_importance, labels)
            
            return {
                "decoder_type": decoder_type,
                "cv_scores": scores.tolist(),
                "mean_score": float(np.mean(scores)),
                "std_score": float(np.std(scores)),
                "feature_importance": feature_importance.tolist() if feature_importance is not None else None,
                "analysis": analysis_result,
                "confidence": self._calculate_decoding_confidence(scores)
            }
            
        except Exception as e:
            print(f"Error decoding brain patterns: {e}")
            return self._mock_pattern_decoding(functional_data, labels, decoder_type)
    
    def _analyze_decoding_results(self, scores: np.ndarray, 
                                feature_importance: Optional[np.ndarray],
                                labels: List[str]) -> Dict[str, Any]:
        """Analyze decoding results for medical insights"""
        try:
            # Basic statistics
            analysis = {
                "mean_accuracy": float(np.mean(scores)),
                "accuracy_std": float(np.std(scores)),
                "min_accuracy": float(np.min(scores)),
                "max_accuracy": float(np.max(scores))
            }
            
            # Performance interpretation
            mean_acc = np.mean(scores)
            if mean_acc > 0.8:
                analysis["performance_level"] = "excellent"
                analysis["clinical_relevance"] = "high"
            elif mean_acc > 0.7:
                analysis["performance_level"] = "good"
                analysis["clinical_relevance"] = "moderate"
            elif mean_acc > 0.6:
                analysis["performance_level"] = "fair"
                analysis["clinical_relevance"] = "low"
            else:
                analysis["performance_level"] = "poor"
                analysis["clinical_relevance"] = "minimal"
            
            # Feature importance analysis
            if feature_importance is not None:
                top_features = np.argsort(np.abs(feature_importance))[-10:]  # Top 10 features
                analysis["top_features"] = top_features.tolist()
                analysis["feature_importance_range"] = {
                    "min": float(np.min(feature_importance)),
                    "max": float(np.max(feature_importance))
                }
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing decoding results: {e}")
            return {
                "mean_accuracy": 0.0,
                "performance_level": "unknown",
                "clinical_relevance": "unknown"
            }
    
    def extract_brain_regions(self, structural_data: str, 
                            atlas_name: str = "harvard_oxford") -> Dict[str, Any]:
        """Extract brain regions from structural data"""
        if not NILEARN_AVAILABLE:
            return self._mock_region_extraction(structural_data, atlas_name)
        
        try:
            # Load structural data
            struct_img = image.load_img(structural_data)
            
            # Get atlas
            atlas = self.atlases.get(atlas_name)
            if not atlas:
                raise ValueError(f"Atlas {atlas_name} not found")
            
            # Extract regions
            masker = NiftiMasker(mask_img=atlas.maps)
            region_data = masker.fit_transform(struct_img)
            
            # Analyze region characteristics
            region_analysis = self._analyze_region_characteristics(region_data, atlas)
            
            return {
                "atlas_name": atlas_name,
                "region_data": region_data.tolist(),
                "analysis": region_analysis,
                "confidence": self._calculate_region_confidence(region_data)
            }
            
        except Exception as e:
            print(f"Error extracting brain regions: {e}")
            return self._mock_region_extraction(structural_data, atlas_name)
    
    def _analyze_region_characteristics(self, region_data: np.ndarray, 
                                      atlas: Any) -> Dict[str, Any]:
        """Analyze characteristics of brain regions"""
        try:
            # Basic statistics for each region
            region_stats = []
            for i in range(region_data.shape[1]):
                region_values = region_data[:, i]
                stats = {
                    "region_index": i,
                    "mean": float(np.mean(region_values)),
                    "std": float(np.std(region_values)),
                    "min": float(np.min(region_values)),
                    "max": float(np.max(region_values))
                }
                
                # Add region name if available
                if hasattr(atlas, 'labels') and i < len(atlas.labels):
                    stats["region_name"] = atlas.labels[i]
                
                region_stats.append(stats)
            
            # Overall analysis
            overall_stats = {
                "total_regions": len(region_stats),
                "mean_region_intensity": float(np.mean(region_data)),
                "region_variability": float(np.std(region_data))
            }
            
            return {
                "region_statistics": region_stats,
                "overall_statistics": overall_stats
            }
            
        except Exception as e:
            print(f"Error analyzing region characteristics: {e}")
            return {
                "region_statistics": [],
                "overall_statistics": {}
            }
    
    def generate_neuroimaging_report(self, analysis_results: Dict[str, Any], 
                                   patient_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive neuroimaging report"""
        try:
            report = {
                "patient_info": patient_info,
                "analysis_summary": self._summarize_analysis(analysis_results),
                "clinical_insights": self._extract_clinical_insights(analysis_results),
                "recommendations": self._generate_clinical_recommendations(analysis_results),
                "technical_details": analysis_results
            }
            
            return report
            
        except Exception as e:
            print(f"Error generating neuroimaging report: {e}")
            return self._mock_neuroimaging_report(patient_info)
    
    def _summarize_analysis(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize neuroimaging analysis results"""
        try:
            summary = {
                "analysis_types": [],
                "key_findings": [],
                "confidence_levels": []
            }
            
            # Extract analysis types
            if "connectivity_type" in analysis_results:
                summary["analysis_types"].append("connectivity_analysis")
                summary["key_findings"].append(f"Brain connectivity analyzed using {analysis_results['connectivity_type']}")
                summary["confidence_levels"].append(analysis_results.get("confidence", 0.5))
            
            if "decoder_type" in analysis_results:
                summary["analysis_types"].append("pattern_decoding")
                summary["key_findings"].append(f"Pattern decoding performed with {analysis_results['decoder_type']}")
                summary["confidence_levels"].append(analysis_results.get("confidence", 0.5))
            
            if "atlas_name" in analysis_results:
                summary["analysis_types"].append("region_extraction")
                summary["key_findings"].append(f"Brain regions extracted using {analysis_results['atlas_name']} atlas")
                summary["confidence_levels"].append(analysis_results.get("confidence", 0.5))
            
            # Calculate overall confidence
            if summary["confidence_levels"]:
                summary["overall_confidence"] = np.mean(summary["confidence_levels"])
            else:
                summary["overall_confidence"] = 0.5
            
            return summary
            
        except Exception as e:
            print(f"Error summarizing analysis: {e}")
            return {
                "analysis_types": ["unknown"],
                "key_findings": ["Analysis summary unavailable"],
                "confidence_levels": [0.5],
                "overall_confidence": 0.5
            }
    
    def _extract_clinical_insights(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Extract clinical insights from neuroimaging analysis"""
        try:
            insights = []
            
            # Connectivity insights
            if "connectivity_type" in analysis_results:
                analysis = analysis_results.get("analysis", {})
                network_metrics = analysis.get("network_metrics", {})
                
                mean_conn = network_metrics.get("mean_connectivity", 0.0)
                if mean_conn > 0.3:
                    insights.append("Strong overall brain connectivity detected")
                elif mean_conn < 0.1:
                    insights.append("Reduced brain connectivity observed")
                
                key_connections = analysis.get("key_connections", [])
                if key_connections:
                    insights.append(f"Key functional connections identified between {len(key_connections)} region pairs")
            
            # Decoding insights
            if "decoder_type" in analysis_results:
                mean_score = analysis_results.get("mean_score", 0.0)
                if mean_score > 0.8:
                    insights.append("High accuracy in brain pattern classification")
                elif mean_score > 0.7:
                    insights.append("Moderate accuracy in brain pattern classification")
                else:
                    insights.append("Low accuracy in brain pattern classification")
            
            return insights[:5]  # Limit to top 5 insights
            
        except Exception as e:
            print(f"Error extracting clinical insights: {e}")
            return ["Clinical insights extraction failed"]
    
    def _generate_clinical_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate clinical recommendations based on neuroimaging analysis"""
        try:
            recommendations = []
            
            # Connectivity-based recommendations
            if "connectivity_type" in analysis_results:
                analysis = analysis_results.get("analysis", {})
                network_metrics = analysis.get("network_metrics", {})
                
                mean_conn = network_metrics.get("mean_connectivity", 0.0)
                if mean_conn < 0.1:
                    recommendations.append("Consider functional connectivity assessment in follow-up")
                
                key_connections = analysis.get("key_connections", [])
                if len(key_connections) > 5:
                    recommendations.append("Monitor key brain network connections")
            
            # Decoding-based recommendations
            if "decoder_type" in analysis_results:
                mean_score = analysis_results.get("mean_score", 0.0)
                if mean_score > 0.8:
                    recommendations.append("High confidence in pattern classification - suitable for clinical use")
                elif mean_score < 0.6:
                    recommendations.append("Low classification accuracy - require additional validation")
            
            return recommendations[:3]  # Limit to top 3 recommendations
            
        except Exception as e:
            print(f"Error generating clinical recommendations: {e}")
            return ["Unable to generate specific recommendations"]
    
    def _calculate_connectivity_confidence(self, connectivity_matrix: np.ndarray) -> float:
        """Calculate confidence score for connectivity analysis"""
        try:
            # Confidence based on matrix properties
            matrix_std = np.std(connectivity_matrix)
            matrix_range = np.max(connectivity_matrix) - np.min(connectivity_matrix)
            
            # Higher confidence for more structured matrices
            confidence = min((matrix_std + matrix_range) / 2, 1.0)
            return max(confidence, 0.0)
            
        except Exception as e:
            print(f"Error calculating connectivity confidence: {e}")
            return 0.5
    
    def _calculate_decoding_confidence(self, scores: np.ndarray) -> float:
        """Calculate confidence score for decoding analysis"""
        try:
            # Confidence based on mean accuracy and consistency
            mean_score = np.mean(scores)
            score_std = np.std(scores)
            
            # Higher confidence for high accuracy and low variance
            confidence = mean_score * (1 - score_std)
            return max(min(confidence, 1.0), 0.0)
            
        except Exception as e:
            print(f"Error calculating decoding confidence: {e}")
            return 0.5
    
    def _calculate_region_confidence(self, region_data: np.ndarray) -> float:
        """Calculate confidence score for region extraction"""
        try:
            # Confidence based on data quality
            data_std = np.std(region_data)
            data_range = np.max(region_data) - np.min(region_data)
            
            # Higher confidence for more structured data
            confidence = min((data_std + data_range) / 2, 1.0)
            return max(confidence, 0.0)
            
        except Exception as e:
            print(f"Error calculating region confidence: {e}")
            return 0.5
    
    # Mock implementations for graceful degradation
    def _mock_connectivity_analysis(self, functional_data: List[str], 
                                  atlas_name: str, connectivity_type: str) -> Dict[str, Any]:
        """Mock connectivity analysis when Nilearn is not available"""
        return {
            "connectivity_type": connectivity_type,
            "atlas_name": atlas_name,
            "connectivity_matrix": [[0.1, 0.2], [0.2, 0.1]],
            "analysis": {
                "network_metrics": {"mean_connectivity": 0.15},
                "key_connections": [{"region_1": "Mock_Region_1", "region_2": "Mock_Region_2", "strength": 0.2}]
            },
            "confidence": 0.5,
            "status": "mock_analysis"
        }
    
    def _mock_pattern_decoding(self, functional_data: List[str], 
                             labels: List[str], decoder_type: str) -> Dict[str, Any]:
        """Mock pattern decoding when Nilearn is not available"""
        return {
            "decoder_type": decoder_type,
            "cv_scores": [0.6, 0.7, 0.65, 0.68, 0.62],
            "mean_score": 0.65,
            "std_score": 0.03,
            "analysis": {
                "mean_accuracy": 0.65,
                "performance_level": "fair",
                "clinical_relevance": "low"
            },
            "confidence": 0.5,
            "status": "mock_decoding"
        }
    
    def _mock_region_extraction(self, structural_data: str, atlas_name: str) -> Dict[str, Any]:
        """Mock region extraction when Nilearn is not available"""
        return {
            "atlas_name": atlas_name,
            "region_data": [[0.1, 0.2, 0.3]],
            "analysis": {
                "region_statistics": [{"region_index": 0, "mean": 0.2, "region_name": "Mock_Region"}],
                "overall_statistics": {"total_regions": 1, "mean_region_intensity": 0.2}
            },
            "confidence": 0.5,
            "status": "mock_extraction"
        }
    
    def _mock_neuroimaging_report(self, patient_info: Dict[str, Any]) -> Dict[str, Any]:
        """Mock neuroimaging report when Nilearn is not available"""
        return {
            "patient_info": patient_info,
            "analysis_summary": {
                "analysis_types": ["mock_analysis"],
                "key_findings": ["Mock neuroimaging analysis performed"],
                "confidence_levels": [0.5],
                "overall_confidence": 0.5
            },
            "clinical_insights": ["Mock clinical insight"],
            "recommendations": ["Mock recommendation"],
            "technical_details": {"status": "mock_report"}
        } 
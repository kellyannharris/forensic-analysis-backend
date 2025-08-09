"""
Bloodsplatter Analysis Service

FastAPI service wrapper for bloodsplatter analysis functionality.
Integrates the completed BloodSpatterCNN module into the forensic API.

Author: Kelly-Ann Harris
Date: January 2025
"""

import os
import sys
import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging
from datetime import datetime
import tempfile
import shutil

from fastapi import HTTPException, UploadFile
from pydantic import BaseModel

# Add the data-processing directory to the path
# Navigate from forensic_backend to forensic-application-capstone/data-processing
data_processing_path = Path(__file__).parent.parent.parent.parent.parent / "forensic-application-capstone" / "data-processing"
sys.path.append(str(data_processing_path))

try:
    from unstructured.bloodsplatter_cnn import BloodSpatterCNN
except ImportError:
    # Fallback import path
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent / "forensic-application-capstone" / "data-processing" / "unstructured"))
    from bloodsplatter_cnn import BloodSpatterCNN

logger = logging.getLogger(__name__)

# Pydantic models for API responses
class BloodspatterAnalysisResponse(BaseModel):
    """Response model for bloodsplatter analysis"""
    image_path: str
    timestamp: str
    cnn_prediction: Dict[str, Any]
    rule_based_pattern: str
    droplet_analysis: Dict[str, Any]
    incident_reconstruction: Dict[str, Any]
    forensic_insights: List[str]
    metadata: Optional[Dict[str, Any]] = None

class BloodspatterBatchResponse(BaseModel):
    """Response model for batch bloodsplatter analysis"""
    total_processed: int
    successful_analyses: int
    failed_analyses: int
    results: List[BloodspatterAnalysisResponse]
    processing_time: float

class BloodspatterComparisonResponse(BaseModel):
    """Response model for bloodsplatter pattern comparison"""
    image_a: str
    image_b: str
    similarity_score: float
    pattern_match: bool
    feature_similarities: Dict[str, float]
    forensic_assessment: str

class BloodspatterService:
    """Service class for bloodsplatter analysis API endpoints"""
    
    def __init__(self):
        self.analyzer = None
        self.model_loaded = False
        self.data_path = Path(__file__).parent.parent.parent.parent.parent / "data" / "unstructured"
        
    async def initialize(self):
        """Initialize the bloodsplatter analyzer"""
        try:
            self.analyzer = BloodSpatterCNN()
            # Load and preprocess data
            await asyncio.get_event_loop().run_in_executor(
                None, 
                self.analyzer.load_and_preprocess_data
            )
            # Build model
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.analyzer.build_model
            )
            self.model_loaded = True
            logger.info("Bloodsplatter analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize bloodsplatter analyzer: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize bloodsplatter analyzer: {e}")
    
    async def analyze_single_image(self, image_file: UploadFile) -> BloodspatterAnalysisResponse:
        """Analyze a single bloodsplatter image"""
        if not self.model_loaded:
            await self.initialize()
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            shutil.copyfileobj(image_file.file, tmp_file)
            tmp_path = tmp_file.name
        
        try:
            # Perform analysis
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.analyzer.analyze_bloodspatter,
                tmp_path
            )
            
            # Format response
            response = BloodspatterAnalysisResponse(
                image_path=image_file.filename,
                timestamp=datetime.now().isoformat(),
                cnn_prediction=result.get('cnn_prediction', {}),
                rule_based_pattern=result.get('rule_based_pattern', ''),
                droplet_analysis=result.get('droplet_analysis', {}),
                incident_reconstruction=result.get('incident_reconstruction', {}),
                forensic_insights=result.get('forensic_insights', []),
                metadata=result.get('metadata', {})
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error analyzing bloodsplatter image: {e}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    async def analyze_batch(self, image_files: List[UploadFile]) -> BloodspatterBatchResponse:
        """Analyze multiple bloodsplatter images"""
        if not self.model_loaded:
            await self.initialize()
        
        results = []
        failed_count = 0
        start_time = datetime.now()
        
        for image_file in image_files:
            try:
                result = await self.analyze_single_image(image_file)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to analyze {image_file.filename}: {e}")
                failed_count += 1
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return BloodspatterBatchResponse(
            total_processed=len(image_files),
            successful_analyses=len(results),
            failed_analyses=failed_count,
            results=results,
            processing_time=processing_time
        )
    
    async def compare_patterns(self, image_a: UploadFile, image_b: UploadFile) -> BloodspatterComparisonResponse:
        """Compare two bloodsplatter patterns"""
        if not self.model_loaded:
            await self.initialize()
        
        # Analyze both images
        analysis_a = await self.analyze_single_image(image_a)
        analysis_b = await self.analyze_single_image(image_b)
        
        # Calculate similarity metrics
        similarity_score = self._calculate_pattern_similarity(analysis_a, analysis_b)
        
        # Determine pattern match
        pattern_match = similarity_score > 0.7  # Threshold for pattern match
        
        # Calculate feature similarities
        feature_similarities = self._calculate_feature_similarities(analysis_a, analysis_b)
        
        # Generate forensic assessment
        forensic_assessment = self._generate_forensic_assessment(similarity_score, pattern_match, feature_similarities)
        
        return BloodspatterComparisonResponse(
            image_a=image_a.filename,
            image_b=image_b.filename,
            similarity_score=similarity_score,
            pattern_match=pattern_match,
            feature_similarities=feature_similarities,
            forensic_assessment=forensic_assessment
        )
    
    def _calculate_pattern_similarity(self, analysis_a: BloodspatterAnalysisResponse, analysis_b: BloodspatterAnalysisResponse) -> float:
        """Calculate similarity between two bloodsplatter patterns"""
        # Compare droplet analysis features
        droplet_a = analysis_a.droplet_analysis
        droplet_b = analysis_b.droplet_analysis
        
        # Calculate similarity based on key features
        similarities = []
        
        # Compare droplet count (normalized)
        if droplet_a.get('count') and droplet_b.get('count'):
            count_diff = abs(droplet_a['count'] - droplet_b['count'])
            count_sim = max(0, 1 - count_diff / max(droplet_a['count'], droplet_b['count']))
            similarities.append(count_sim)
        
        # Compare average size
        if droplet_a.get('average_size') and droplet_b.get('average_size'):
            size_diff = abs(droplet_a['average_size'] - droplet_b['average_size'])
            size_sim = max(0, 1 - size_diff / max(droplet_a['average_size'], droplet_b['average_size']))
            similarities.append(size_sim)
        
        # Compare aspect ratio
        if droplet_a.get('aspect_ratio') and droplet_b.get('aspect_ratio'):
            ratio_diff = abs(droplet_a['aspect_ratio'] - droplet_b['aspect_ratio'])
            ratio_sim = max(0, 1 - ratio_diff / max(droplet_a['aspect_ratio'], droplet_b['aspect_ratio']))
            similarities.append(ratio_sim)
        
        # Compare pattern types
        if analysis_a.rule_based_pattern == analysis_b.rule_based_pattern:
            similarities.append(1.0)
        else:
            similarities.append(0.0)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _calculate_feature_similarities(self, analysis_a: BloodspatterAnalysisResponse, analysis_b: BloodspatterAnalysisResponse) -> Dict[str, float]:
        """Calculate detailed feature similarities"""
        features = {}
        
        # Droplet count similarity
        if analysis_a.droplet_analysis.get('count') and analysis_b.droplet_analysis.get('count'):
            count_diff = abs(analysis_a.droplet_analysis['count'] - analysis_b.droplet_analysis['count'])
            features['droplet_count'] = max(0, 1 - count_diff / max(analysis_a.droplet_analysis['count'], analysis_b.droplet_analysis['count']))
        
        # Size similarity
        if analysis_a.droplet_analysis.get('average_size') and analysis_b.droplet_analysis.get('average_size'):
            size_diff = abs(analysis_a.droplet_analysis['average_size'] - analysis_b.droplet_analysis['average_size'])
            features['average_size'] = max(0, 1 - size_diff / max(analysis_a.droplet_analysis['average_size'], analysis_b.droplet_analysis['average_size']))
        
        # Pattern type similarity
        features['pattern_type'] = 1.0 if analysis_a.rule_based_pattern == analysis_b.rule_based_pattern else 0.0
        
        # Impact angle similarity
        if (analysis_a.incident_reconstruction.get('estimated_impact_angle') and 
            analysis_b.incident_reconstruction.get('estimated_impact_angle')):
            angle_diff = abs(analysis_a.incident_reconstruction['estimated_impact_angle'] - 
                           analysis_b.incident_reconstruction['estimated_impact_angle'])
            features['impact_angle'] = max(0, 1 - angle_diff / 90)  # Normalize by 90 degrees
        
        return features
    
    def _generate_forensic_assessment(self, similarity_score: float, pattern_match: bool, feature_similarities: Dict[str, float]) -> str:
        """Generate forensic assessment based on comparison results"""
        if similarity_score > 0.8:
            return "High likelihood of same source event. Strong pattern correlation across multiple features."
        elif similarity_score > 0.6:
            return "Moderate likelihood of same source event. Some pattern similarities observed."
        elif similarity_score > 0.4:
            return "Low likelihood of same source event. Limited pattern correlation."
        else:
            return "Patterns appear to be from different source events. Significant differences in key features."
    
    async def get_available_datasets(self) -> Dict[str, Any]:
        """Get information about available bloodsplatter datasets"""
        datasets = {}
        
        # Check bloodsplatter1 directory
        bs1_path = self.data_path / "bloodsplatter1"
        if bs1_path.exists():
            datasets["bloodsplatter1"] = {
                "path": str(bs1_path),
                "experiments": [d.name for d in bs1_path.iterdir() if d.is_dir()],
                "total_experiments": len([d for d in bs1_path.iterdir() if d.is_dir()])
            }
        
        # Check bloodsplatter2 directory
        bs2_path = self.data_path / "bloodsplatter2"
        if bs2_path.exists():
            datasets["bloodsplatter2"] = {
                "path": str(bs2_path),
                "experiments": [d.name for d in bs2_path.iterdir() if d.is_dir()],
                "total_experiments": len([d for d in bs2_path.iterdir() if d.is_dir()])
            }
        
        return datasets
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the bloodsplatter analysis model"""
        if not self.model_loaded:
            await self.initialize()
        
        return {
            "model_type": "BloodSpatterCNN",
            "architecture": "Transfer Learning (ResNet50 + EfficientNetB0)",
            "capabilities": [
                "Pattern classification",
                "Impact angle estimation",
                "Velocity estimation",
                "Weapon type suggestion",
                "Droplet analysis",
                "Incident reconstruction"
            ],
            "supported_formats": ["JPG", "PNG", "TIFF"],
            "max_image_size": "45MB",
            "processing_time": "~2-3 seconds per image"
        } 
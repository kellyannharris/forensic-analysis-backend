"""
Handwriting Analysis Service

FastAPI service wrapper for handwriting analysis functionality.
Integrates the completed HandwritingAnalyzer module into the forensic API.

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
    from unstructured.handwriting_analyzer import HandwritingAnalyzer
except ImportError:
    # Fallback import path
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent / "forensic-application-capstone" / "data-processing" / "unstructured"))
    from handwriting_analyzer import HandwritingAnalyzer

logger = logging.getLogger(__name__)

# Pydantic models for API responses
class HandwritingIdentificationResponse(BaseModel):
    """Response model for handwriting identification"""
    image_path: str
    timestamp: str
    top_predictions: List[Dict[str, Any]]
    handwriting_features: Dict[str, Any]
    demographics: Optional[Dict[str, Any]] = None

class HandwritingVerificationResponse(BaseModel):
    """Response model for handwriting verification"""
    image_a: str
    image_b: str
    timestamp: str
    similarity_score: float
    same_writer: bool
    confidence: float
    feature_similarity: Dict[str, float]

class HandwritingBatchResponse(BaseModel):
    """Response model for batch handwriting analysis"""
    total_processed: int
    successful_analyses: int
    failed_analyses: int
    results: List[HandwritingIdentificationResponse]
    processing_time: float

class WriterProfileResponse(BaseModel):
    """Response model for writer profile information"""
    writer_id: str
    demographics: Dict[str, Any]
    total_samples: int
    session_info: Dict[str, Any]
    writing_characteristics: Dict[str, Any]

class HandwritingService:
    """Service class for handwriting analysis API endpoints"""
    
    def __init__(self):
        self.analyzer = None
        self.model_loaded = False
        self.data_path = Path(__file__).parent.parent.parent.parent.parent / "data" / "unstructured"
        
    async def initialize(self):
        """Initialize the handwriting analyzer"""
        try:
            self.analyzer = HandwritingAnalyzer()
            # Load metadata
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.analyzer.load_metadata
            )
            # Build models
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.analyzer.build_identification_model
            )
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.analyzer.build_verification_model
            )
            self.model_loaded = True
            logger.info("Handwriting analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize handwriting analyzer: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize handwriting analyzer: {e}")
    
    async def identify_writer(self, image_file: UploadFile, top_k: int = 5) -> HandwritingIdentificationResponse:
        """Identify writer from handwriting sample"""
        if not self.model_loaded:
            await self.initialize()
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            shutil.copyfileobj(image_file.file, tmp_file)
            tmp_path = tmp_file.name
        
        try:
            # Perform identification
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.analyzer.identify_writer,
                tmp_path,
                top_k
            )
            
            # Format response
            response = HandwritingIdentificationResponse(
                image_path=image_file.filename,
                timestamp=datetime.now().isoformat(),
                top_predictions=result.get('top_predictions', []),
                handwriting_features=result.get('handwriting_features', {}),
                demographics=result.get('demographics', {})
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error identifying writer: {e}")
            raise HTTPException(status_code=500, detail=f"Identification failed: {e}")
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    async def verify_writer(self, image_a: UploadFile, image_b: UploadFile) -> HandwritingVerificationResponse:
        """Verify if two handwriting samples are from the same writer"""
        if not self.model_loaded:
            await self.initialize()
        
        # Save uploaded files temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_a:
            shutil.copyfileobj(image_a.file, tmp_a)
            tmp_path_a = tmp_a.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_b:
            shutil.copyfileobj(image_b.file, tmp_b)
            tmp_path_b = tmp_b.name
        
        try:
            # Perform verification
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.analyzer.verify_writer,
                tmp_path_a,
                tmp_path_b
            )
            
            # Format response
            response = HandwritingVerificationResponse(
                image_a=image_a.filename,
                image_b=image_b.filename,
                timestamp=datetime.now().isoformat(),
                similarity_score=result.get('similarity_score', 0.0),
                same_writer=result.get('same_writer', False),
                confidence=result.get('confidence', 0.0),
                feature_similarity=result.get('feature_similarity', {})
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error verifying writer: {e}")
            raise HTTPException(status_code=500, detail=f"Verification failed: {e}")
        finally:
            # Clean up temporary files
            if os.path.exists(tmp_path_a):
                os.unlink(tmp_path_a)
            if os.path.exists(tmp_path_b):
                os.unlink(tmp_path_b)
    
    async def analyze_batch(self, image_files: List[UploadFile], top_k: int = 5) -> HandwritingBatchResponse:
        """Analyze multiple handwriting samples"""
        if not self.model_loaded:
            await self.initialize()
        
        results = []
        failed_count = 0
        start_time = datetime.now()
        
        for image_file in image_files:
            try:
                result = await self.identify_writer(image_file, top_k)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to analyze {image_file.filename}: {e}")
                failed_count += 1
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return HandwritingBatchResponse(
            total_processed=len(image_files),
            successful_analyses=len(results),
            failed_analyses=failed_count,
            results=results,
            processing_time=processing_time
        )
    
    async def get_writer_profile(self, writer_id: str) -> WriterProfileResponse:
        """Get detailed profile information for a specific writer"""
        if not self.model_loaded:
            await self.initialize()
        
        try:
            # Get writer information from metadata
            writer_info = self.analyzer.metadata[self.analyzer.metadata['writer_id'] == writer_id]
            
            if writer_info.empty:
                raise HTTPException(status_code=404, detail=f"Writer {writer_id} not found")
            
            writer_row = writer_info.iloc[0]
            
            # Get writing characteristics
            writing_chars = {}
            
            # Calculate session information
            session_info = {
                "total_sessions": len(writer_info['session'].unique()),
                "sessions": writer_info['session'].unique().tolist(),
                "types": writer_info['type'].unique().tolist(),
                "repetitions": writer_info['repetition'].unique().tolist()
            }
            
            response = WriterProfileResponse(
                writer_id=writer_id,
                demographics={
                    "gender": writer_row.get('gender', 'Unknown'),
                    "age_group": writer_row.get('agegroup', 'Unknown'),
                    "handedness": writer_row.get('hand', 'Unknown'),
                    "third_grade_usa": writer_row.get('thirdgrade_usa', 'Unknown'),
                    "region": writer_row.get('region', 'Unknown')
                },
                total_samples=len(writer_info),
                session_info=session_info,
                writing_characteristics=writing_chars
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error getting writer profile: {e}")
            raise HTTPException(status_code=500, detail=f"Profile retrieval failed: {e}")
    
    async def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get statistics about the handwriting dataset"""
        if not self.model_loaded:
            await self.initialize()
        
        try:
            stats = {
                "total_writers": len(self.analyzer.metadata['writer_id'].unique()),
                "total_samples": len(self.analyzer.metadata),
                "demographics": {
                    "gender": self.analyzer.metadata['gender'].value_counts().to_dict(),
                    "age_groups": self.analyzer.metadata['agegroup'].value_counts().to_dict(),
                    "handedness": self.analyzer.metadata['hand'].value_counts().to_dict(),
                    "regions": self.analyzer.metadata['region'].value_counts().to_dict()
                },
                "sessions": {
                    "total_sessions": len(self.analyzer.metadata['session'].unique()),
                    "session_distribution": self.analyzer.metadata['session'].value_counts().to_dict()
                },
                "writing_types": {
                    "types": self.analyzer.metadata['type'].value_counts().to_dict(),
                    "repetitions": self.analyzer.metadata['repetition'].value_counts().to_dict()
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting dataset statistics: {e}")
            raise HTTPException(status_code=500, detail=f"Statistics retrieval failed: {e}")
    
    async def compare_writing_styles(self, writer_ids: List[str]) -> Dict[str, Any]:
        """Compare writing styles between multiple writers"""
        if not self.model_loaded:
            await self.initialize()
        
        if len(writer_ids) < 2:
            raise HTTPException(status_code=400, detail="At least 2 writers required for comparison")
        
        try:
            comparisons = {}
            
            # Get writer information
            for writer_id in writer_ids:
                writer_info = self.analyzer.metadata[self.analyzer.metadata['writer_id'] == writer_id]
                if not writer_info.empty:
                    writer_row = writer_info.iloc[0]
                    comparisons[writer_id] = {
                        "demographics": {
                            "gender": writer_row.get('gender', 'Unknown'),
                            "age_group": writer_row.get('agegroup', 'Unknown'),
                            "handedness": writer_row.get('hand', 'Unknown'),
                            "region": writer_row.get('region', 'Unknown')
                        },
                        "sample_count": len(writer_info)
                    }
            
            # Calculate demographic similarities
            similarities = {}
            for i, writer_a in enumerate(writer_ids):
                for writer_b in writer_ids[i+1:]:
                    if writer_a in comparisons and writer_b in comparisons:
                        demo_a = comparisons[writer_a]["demographics"]
                        demo_b = comparisons[writer_b]["demographics"]
                        
                        similarity = 0
                        total_features = 0
                        
                        for key in demo_a:
                            if demo_a[key] != 'Unknown' and demo_b[key] != 'Unknown':
                                if demo_a[key] == demo_b[key]:
                                    similarity += 1
                                total_features += 1
                        
                        similarities[f"{writer_a}_vs_{writer_b}"] = {
                            "demographic_similarity": similarity / total_features if total_features > 0 else 0,
                            "common_features": similarity,
                            "total_features": total_features
                        }
            
            return {
                "writers": comparisons,
                "similarities": similarities,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error comparing writing styles: {e}")
            raise HTTPException(status_code=500, detail=f"Style comparison failed: {e}")
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the handwriting analysis models"""
        if not self.model_loaded:
            await self.initialize()
        
        return {
            "identification_model": {
                "type": "CNN with ResNet50 backbone",
                "parameters": "~24.8M",
                "architecture": "Transfer Learning + Dense Layers",
                "output": "Writer classification (466 classes)"
            },
            "verification_model": {
                "type": "Siamese Network",
                "parameters": "~24.9M", 
                "architecture": "Shared CNN + Similarity Learning",
                "output": "Same writer probability"
            },
            "feature_extraction": {
                "features": [
                    "Text density",
                    "Stroke width",
                    "Character count",
                    "Character spacing",
                    "Slant angle",
                    "Texture complexity"
                ],
                "preprocessing": [
                    "Histogram equalization",
                    "Noise reduction",
                    "Resize to 224x224",
                    "Normalization"
                ]
            },
            "dataset_info": {
                "total_writers": 466,
                "total_samples": 12825,
                "formats": ["PNG"],
                "sessions": 3,
                "types": ["LND", "PHR", "WOZ"],
                "repetitions": "Up to 3"
            },
            "capabilities": [
                "Writer identification",
                "Writer verification",
                "Demographic analysis",
                "Cross-type consistency",
                "Batch processing"
            ]
        } 
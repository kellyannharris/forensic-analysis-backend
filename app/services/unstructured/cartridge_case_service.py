"""
Cartridge Case Analysis Service

FastAPI service wrapper for cartridge case analysis functionality.
Integrates the completed CartridgeCaseAnalyzer module into the forensic API.

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
    from unstructured.cartridge_case_analyzer import CartridgeCaseAnalyzer
except ImportError:
    # Fallback import path
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent / "forensic-application-capstone" / "data-processing" / "unstructured"))
    from cartridge_case_analyzer import CartridgeCaseAnalyzer

logger = logging.getLogger(__name__)

# Pydantic models for API responses
class CartridgeCaseAnalysisResponse(BaseModel):
    """Response model for cartridge case analysis"""
    filename: str
    timestamp: str
    file_info: Dict[str, Any]
    surface_dimensions: List[int]
    features: Dict[str, float]
    firearm_identification: Optional[Dict[str, Any]] = None
    ballistics_analysis: Optional[Dict[str, Any]] = None

class CartridgeCaseComparisonResponse(BaseModel):
    """Response model for cartridge case comparison"""
    case_a: str
    case_b: str
    timestamp: str
    similarity_score: float
    correlation: float
    cosine_similarity: float
    euclidean_distance: float
    same_firearm_likelihood: str
    confidence_level: str
    forensic_conclusion: str

class CartridgeCaseBatchResponse(BaseModel):
    """Response model for batch cartridge case analysis"""
    total_processed: int
    successful_analyses: int
    failed_analyses: int
    results: List[CartridgeCaseAnalysisResponse]
    processing_time: float

class FirearmIdentificationResponse(BaseModel):
    """Response model for firearm identification"""
    case_filename: str
    timestamp: str
    top_predictions: List[Dict[str, Any]]
    confidence_scores: Dict[str, float]
    features_used: List[str]
    model_info: Dict[str, Any]

class BallisticsReportResponse(BaseModel):
    """Response model for comprehensive ballistics report"""
    case_filename: str
    timestamp: str
    surface_analysis: Dict[str, Any]
    tool_mark_analysis: Dict[str, Any]
    firearm_identification: Dict[str, Any]
    comparison_results: List[Dict[str, Any]]
    forensic_conclusions: List[str]

class CartridgeCaseService:
    """Service class for cartridge case analysis API endpoints"""
    
    def __init__(self):
        self.analyzer = None
        self.model_loaded = False
        self.data_path = Path(__file__).parent.parent.parent.parent.parent / "data" / "unstructured" / "cartridgeCaseScans"
        
    async def initialize(self):
        """Initialize the cartridge case analyzer"""
        try:
            self.analyzer = CartridgeCaseAnalyzer()
            self.model_loaded = True
            logger.info("Cartridge case analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize cartridge case analyzer: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize cartridge case analyzer: {e}")
    
    async def analyze_single_case(self, x3p_file: UploadFile) -> CartridgeCaseAnalysisResponse:
        """Analyze a single cartridge case X3P file"""
        if not self.model_loaded:
            await self.initialize()
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.x3p') as tmp_file:
            shutil.copyfileobj(x3p_file.file, tmp_file)
            tmp_path = tmp_file.name
        
        try:
            # Perform analysis
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.analyzer.analyze_cartridge_case,
                tmp_path
            )
            
            # Extract file information
            file_info = await asyncio.get_event_loop().run_in_executor(
                None,
                self.analyzer.parse_filename,
                x3p_file.filename
            )
            
            # Format response
            response = CartridgeCaseAnalysisResponse(
                filename=x3p_file.filename,
                timestamp=datetime.now().isoformat(),
                file_info=file_info,
                surface_dimensions=result.get('surface_dimensions', []),
                features=result.get('features', {}),
                firearm_identification=result.get('firearm_identification', {}),
                ballistics_analysis=result.get('ballistics_analysis', {})
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error analyzing cartridge case: {e}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    async def compare_cases(self, case_a: UploadFile, case_b: UploadFile) -> CartridgeCaseComparisonResponse:
        """Compare two cartridge cases"""
        if not self.model_loaded:
            await self.initialize()
        
        # Save uploaded files temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.x3p') as tmp_a:
            shutil.copyfileobj(case_a.file, tmp_a)
            tmp_path_a = tmp_a.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.x3p') as tmp_b:
            shutil.copyfileobj(case_b.file, tmp_b)
            tmp_path_b = tmp_b.name
        
        try:
            # Perform comparison
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.analyzer.compare_cases,
                tmp_path_a,
                tmp_path_b
            )
            
            # Generate forensic conclusion
            forensic_conclusion = self._generate_forensic_conclusion(result)
            
            # Format response
            response = CartridgeCaseComparisonResponse(
                case_a=case_a.filename,
                case_b=case_b.filename,
                timestamp=datetime.now().isoformat(),
                similarity_score=result.get('similarity_score', 0.0),
                correlation=result.get('correlation', 0.0),
                cosine_similarity=result.get('cosine_similarity', 0.0),
                euclidean_distance=result.get('euclidean_distance', 0.0),
                same_firearm_likelihood=result.get('same_firearm_likelihood', 'Unknown'),
                confidence_level=result.get('confidence_level', 'Unknown'),
                forensic_conclusion=forensic_conclusion
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error comparing cartridge cases: {e}")
            raise HTTPException(status_code=500, detail=f"Comparison failed: {e}")
        finally:
            # Clean up temporary files
            if os.path.exists(tmp_path_a):
                os.unlink(tmp_path_a)
            if os.path.exists(tmp_path_b):
                os.unlink(tmp_path_b)
    
    async def identify_firearm(self, x3p_file: UploadFile, top_k: int = 5) -> FirearmIdentificationResponse:
        """Identify source firearm from cartridge case"""
        if not self.model_loaded:
            await self.initialize()
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.x3p') as tmp_file:
            shutil.copyfileobj(x3p_file.file, tmp_file)
            tmp_path = tmp_file.name
        
        try:
            # Perform identification
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.analyzer.identify_firearm,
                tmp_path,
                top_k
            )
            
            # Format response
            response = FirearmIdentificationResponse(
                case_filename=x3p_file.filename,
                timestamp=datetime.now().isoformat(),
                top_predictions=result.get('top_predictions', []),
                confidence_scores=result.get('confidence_scores', {}),
                features_used=result.get('features_used', []),
                model_info=result.get('model_info', {})
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error identifying firearm: {e}")
            raise HTTPException(status_code=500, detail=f"Identification failed: {e}")
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    async def analyze_batch(self, x3p_files: List[UploadFile]) -> CartridgeCaseBatchResponse:
        """Analyze multiple cartridge case files"""
        if not self.model_loaded:
            await self.initialize()
        
        results = []
        failed_count = 0
        start_time = datetime.now()
        
        for x3p_file in x3p_files:
            try:
                result = await self.analyze_single_case(x3p_file)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to analyze {x3p_file.filename}: {e}")
                failed_count += 1
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return CartridgeCaseBatchResponse(
            total_processed=len(x3p_files),
            successful_analyses=len(results),
            failed_analyses=failed_count,
            results=results,
            processing_time=processing_time
        )
    
    async def generate_ballistics_report(self, x3p_file: UploadFile, compare_with: Optional[List[str]] = None) -> BallisticsReportResponse:
        """Generate comprehensive ballistics report"""
        if not self.model_loaded:
            await self.initialize()
        
        # Analyze the primary case
        analysis = await self.analyze_single_case(x3p_file)
        
        # Perform firearm identification
        identification = await self.identify_firearm(x3p_file)
        
        # Compare with other cases if provided
        comparison_results = []
        if compare_with:
            for case_path in compare_with:
                if os.path.exists(case_path):
                    with open(case_path, 'rb') as f:
                        comparison_file = UploadFile(filename=os.path.basename(case_path), file=f)
                        comparison = await self.compare_cases(x3p_file, comparison_file)
                        comparison_results.append(comparison.dict())
        
        # Generate forensic conclusions
        forensic_conclusions = self._generate_forensic_conclusions(analysis, identification, comparison_results)
        
        response = BallisticsReportResponse(
            case_filename=x3p_file.filename,
            timestamp=datetime.now().isoformat(),
            surface_analysis={
                "dimensions": analysis.surface_dimensions,
                "features": analysis.features
            },
            tool_mark_analysis={
                "dominant_angle": analysis.features.get('dominant_tool_mark_angle', 0),
                "directionality": analysis.features.get('tool_mark_directionality', 0),
                "texture_uniformity": analysis.features.get('texture_uniformity', 0)
            },
            firearm_identification=identification.dict(),
            comparison_results=comparison_results,
            forensic_conclusions=forensic_conclusions
        )
        
        return response
    
    def _generate_forensic_conclusion(self, comparison_result: Dict[str, Any]) -> str:
        """Generate forensic conclusion from comparison results"""
        similarity_score = comparison_result.get('similarity_score', 0.0)
        correlation = comparison_result.get('correlation', 0.0)
        
        if similarity_score > 0.8 and correlation > 0.7:
            return "Strong evidence suggests cases fired from same firearm. High correlation in ballistics features."
        elif similarity_score > 0.6 and correlation > 0.5:
            return "Moderate evidence suggests cases may be from same firearm. Further analysis recommended."
        elif similarity_score > 0.4:
            return "Limited evidence of same source. Some similarities observed but not conclusive."
        else:
            return "Cases appear to be from different firearms. Significant differences in ballistics features."
    
    def _generate_forensic_conclusions(self, analysis: CartridgeCaseAnalysisResponse, 
                                     identification: FirearmIdentificationResponse,
                                     comparisons: List[Dict[str, Any]]) -> List[str]:
        """Generate comprehensive forensic conclusions"""
        conclusions = []
        
        # Surface analysis conclusions
        if analysis.features.get('surface_coverage', 0) > 0.8:
            conclusions.append("High quality surface scan with excellent coverage for analysis.")
        
        # Feature analysis conclusions
        if analysis.features.get('edge_density', 0) > 0.3:
            conclusions.append("Well-defined tool marks present, suitable for comparison analysis.")
        
        # Identification conclusions
        if identification.top_predictions:
            top_pred = identification.top_predictions[0]
            if top_pred.get('confidence', 0) > 0.7:
                conclusions.append(f"High confidence identification of source firearm: {top_pred.get('firearm_id', 'Unknown')}")
            elif top_pred.get('confidence', 0) > 0.5:
                conclusions.append(f"Moderate confidence identification suggests firearm: {top_pred.get('firearm_id', 'Unknown')}")
        
        # Comparison conclusions
        for comp in comparisons:
            if comp.get('similarity_score', 0) > 0.7:
                conclusions.append(f"Strong match with {comp.get('case_b', 'comparison case')}")
        
        return conclusions
    
    async def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about available cartridge case datasets"""
        datasets = {}
        
        # Check Fadul dataset
        fadul_path = self.data_path / "fadulMasked"
        if fadul_path.exists():
            datasets["fadul"] = {
                "name": "Fadul Dataset",
                "path": str(fadul_path),
                "files": [f.name for f in fadul_path.glob("*.x3p")],
                "count": len(list(fadul_path.glob("*.x3p"))),
                "description": "Miami-Dade Police Department Crime Laboratory dataset"
            }
        
        # Check Weller dataset
        weller_path = self.data_path / "wellerMasked"
        if weller_path.exists():
            datasets["weller"] = {
                "name": "Weller Dataset", 
                "path": str(weller_path),
                "files": [f.name for f in weller_path.glob("*.x3p")],
                "count": len(list(weller_path.glob("*.x3p"))),
                "description": "Consecutive manufacturing study dataset"
            }
        
        return {
            "datasets": datasets,
            "total_files": sum(d["count"] for d in datasets.values()),
            "supported_format": "X3P (XML 3D Surface Profile)"
        }
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the cartridge case analysis model"""
        if not self.model_loaded:
            await self.initialize()
        
        return {
            "model_type": "CartridgeCaseAnalyzer",
            "machine_learning": {
                "algorithm": "RandomForestClassifier",
                "trees": 100,
                "max_depth": 10,
                "features": 20
            },
            "feature_extraction": {
                "surface_statistics": [
                    "mean", "std", "skewness", "kurtosis", "range", "coverage"
                ],
                "gradient_analysis": [
                    "gradient_mean", "gradient_std", "gradient_max", "dominant_tool_mark_angle"
                ],
                "texture_analysis": [
                    "texture_uniformity", "edge_density", "contour_count", "contour_area"
                ],
                "frequency_domain": [
                    "spectral_centroid_x", "spectral_centroid_y", "spectral_energy"
                ]
            },
            "capabilities": [
                "3D surface analysis",
                "Firearm identification",
                "Case comparison",
                "Tool mark analysis",
                "Ballistics report generation"
            ],
            "supported_formats": ["X3P"],
            "datasets_supported": ["Fadul", "Weller"],
            "processing_time": "~1-2 seconds per case"
        }
    
    async def train_model(self, dataset_name: str = "combined") -> Dict[str, Any]:
        """Train or retrain the identification model"""
        if not self.model_loaded:
            await self.initialize()
        
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.analyzer.train_identification_model,
                dataset_name
            )
            
            return {
                "training_completed": True,
                "dataset_used": dataset_name,
                "model_performance": result.get('performance', {}),
                "training_time": result.get('training_time', 0),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise HTTPException(status_code=500, detail=f"Model training failed: {e}") 
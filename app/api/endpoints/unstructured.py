"""
Unstructured Data Analysis API Endpoints

FastAPI endpoints for analyzing unstructured forensic data:
- Bloodsplatter Analysis
- Handwriting Analysis  
- Cartridge Case Analysis

Author: Kelly-Ann Harris
Date: January 2025
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
import logging

from app.services.unstructured import (
    BloodspatterService,
    HandwritingService, 
    CartridgeCaseService
)

logger = logging.getLogger(__name__)

# Create routers
router = APIRouter()
bloodsplatter_router = APIRouter(prefix="/bloodsplatter", tags=["Bloodsplatter Analysis"])
handwriting_router = APIRouter(prefix="/handwriting", tags=["Handwriting Analysis"])
cartridge_router = APIRouter(prefix="/cartridge-case", tags=["Cartridge Case Analysis"])

# Service instances
bloodsplatter_service = BloodspatterService()
handwriting_service = HandwritingService()
cartridge_service = CartridgeCaseService()

# ============================================================================
# BLOODSPLATTER ANALYSIS ENDPOINTS
# ============================================================================

@bloodsplatter_router.post("/analyze")
async def analyze_bloodsplatter(
    image: UploadFile = File(..., description="Bloodsplatter image (JPG/PNG/TIFF, max 45MB)")
):
    """
    Analyze a single bloodsplatter image for forensic reconstruction.
    
    Returns detailed analysis including:
    - CNN-based pattern classification
    - Rule-based forensic analysis
    - Droplet analysis and measurements
    - Impact angle estimation
    - Incident reconstruction insights
    """
    try:
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        result = await bloodsplatter_service.analyze_single_image(image)
        return result
    except Exception as e:
        logger.error(f"Bloodsplatter analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@bloodsplatter_router.post("/analyze/batch")
async def analyze_bloodsplatter_batch(
    images: List[UploadFile] = File(..., description="Multiple bloodsplatter images")
):
    """
    Analyze multiple bloodsplatter images in batch.
    
    Returns batch processing results with individual analysis for each image.
    """
    try:
        for image in images:
            if not image.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail=f"File {image.filename} must be an image")
        
        result = await bloodsplatter_service.analyze_batch(images)
        return result
    except Exception as e:
        logger.error(f"Bloodsplatter batch analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@bloodsplatter_router.post("/compare")
async def compare_bloodsplatter_patterns(
    image_a: UploadFile = File(..., description="First bloodsplatter image"),
    image_b: UploadFile = File(..., description="Second bloodsplatter image")
):
    """
    Compare two bloodsplatter patterns for similarity analysis.
    
    Returns detailed comparison with similarity scores and forensic assessment.
    """
    try:
        if not image_a.content_type.startswith('image/') or not image_b.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Both files must be images")
        
        result = await bloodsplatter_service.compare_patterns(image_a, image_b)
        return result
    except Exception as e:
        logger.error(f"Bloodsplatter comparison failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@bloodsplatter_router.get("/datasets")
async def get_bloodsplatter_datasets():
    """Get information about available bloodsplatter datasets."""
    try:
        result = await bloodsplatter_service.get_available_datasets()
        return result
    except Exception as e:
        logger.error(f"Failed to get bloodsplatter datasets: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@bloodsplatter_router.get("/model/info")
async def get_bloodsplatter_model_info():
    """Get information about the bloodsplatter analysis model."""
    try:
        result = await bloodsplatter_service.get_model_info()
        return result
    except Exception as e:
        logger.error(f"Failed to get bloodsplatter model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# HANDWRITING ANALYSIS ENDPOINTS
# ============================================================================

@handwriting_router.post("/identify")
async def identify_writer(
    image: UploadFile = File(..., description="Handwriting sample image (PNG)"),
    top_k: int = 5
):
    """
    Identify writer from handwriting sample.
    
    Returns top-K writer predictions with confidence scores and demographic information.
    """
    try:
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        result = await handwriting_service.identify_writer(image, top_k)
        return result
    except Exception as e:
        logger.error(f"Handwriting identification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@handwriting_router.post("/verify")
async def verify_writer(
    image_a: UploadFile = File(..., description="First handwriting sample"),
    image_b: UploadFile = File(..., description="Second handwriting sample")
):
    """
    Verify if two handwriting samples are from the same writer.
    
    Returns similarity score and same-writer determination.
    """
    try:
        if not image_a.content_type.startswith('image/') or not image_b.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Both files must be images")
        
        result = await handwriting_service.verify_writer(image_a, image_b)
        return result
    except Exception as e:
        logger.error(f"Handwriting verification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@handwriting_router.post("/analyze/batch")
async def analyze_handwriting_batch(
    images: List[UploadFile] = File(..., description="Multiple handwriting sample images"),
    top_k: int = 5
):
    """
    Analyze multiple handwriting samples in batch.
    
    Returns batch processing results with writer identification for each sample.
    """
    try:
        for image in images:
            if not image.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail=f"File {image.filename} must be an image")
        
        result = await handwriting_service.analyze_batch(images, top_k)
        return result
    except Exception as e:
        logger.error(f"Handwriting batch analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@handwriting_router.get("/writer/{writer_id}")
async def get_writer_profile(writer_id: str):
    """Get detailed profile information for a specific writer."""
    try:
        result = await handwriting_service.get_writer_profile(writer_id)
        return result
    except Exception as e:
        logger.error(f"Failed to get writer profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@handwriting_router.get("/dataset/statistics")
async def get_handwriting_dataset_statistics():
    """Get comprehensive statistics about the handwriting dataset."""
    try:
        result = await handwriting_service.get_dataset_statistics()
        return result
    except Exception as e:
        logger.error(f"Failed to get dataset statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@handwriting_router.post("/compare-styles")
async def compare_writing_styles(writer_ids: List[str]):
    """Compare writing styles between multiple writers."""
    try:
        result = await handwriting_service.compare_writing_styles(writer_ids)
        return result
    except Exception as e:
        logger.error(f"Failed to compare writing styles: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@handwriting_router.get("/model/info")
async def get_handwriting_model_info():
    """Get information about the handwriting analysis models."""
    try:
        result = await handwriting_service.get_model_info()
        return result
    except Exception as e:
        logger.error(f"Failed to get handwriting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# CARTRIDGE CASE ANALYSIS ENDPOINTS
# ============================================================================

@cartridge_router.post("/analyze")
async def analyze_cartridge_case(
    x3p_file: UploadFile = File(..., description="Cartridge case X3P file")
):
    """
    Analyze a single cartridge case X3P file for ballistics analysis.
    
    Returns detailed surface analysis and ballistics features.
    """
    try:
        if not x3p_file.filename.endswith('.x3p'):
            raise HTTPException(status_code=400, detail="File must be in X3P format")
        
        result = await cartridge_service.analyze_single_case(x3p_file)
        return result
    except Exception as e:
        logger.error(f"Cartridge case analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@cartridge_router.post("/compare")
async def compare_cartridge_cases(
    case_a: UploadFile = File(..., description="First cartridge case X3P file"),
    case_b: UploadFile = File(..., description="Second cartridge case X3P file")
):
    """
    Compare two cartridge cases for ballistics analysis.
    
    Returns detailed comparison with similarity metrics and forensic conclusions.
    """
    try:
        if not case_a.filename.endswith('.x3p') or not case_b.filename.endswith('.x3p'):
            raise HTTPException(status_code=400, detail="Both files must be in X3P format")
        
        result = await cartridge_service.compare_cases(case_a, case_b)
        return result
    except Exception as e:
        logger.error(f"Cartridge case comparison failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@cartridge_router.post("/identify")
async def identify_firearm(
    x3p_file: UploadFile = File(..., description="Cartridge case X3P file"),
    top_k: int = 5
):
    """
    Identify source firearm from cartridge case analysis.
    
    Returns top-K firearm predictions with confidence scores.
    """
    try:
        if not x3p_file.filename.endswith('.x3p'):
            raise HTTPException(status_code=400, detail="File must be in X3P format")
        
        result = await cartridge_service.identify_firearm(x3p_file, top_k)
        return result
    except Exception as e:
        logger.error(f"Firearm identification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@cartridge_router.post("/analyze/batch")
async def analyze_cartridge_batch(
    x3p_files: List[UploadFile] = File(..., description="Multiple cartridge case X3P files")
):
    """
    Analyze multiple cartridge case files in batch.
    
    Returns batch processing results with individual analysis for each case.
    """
    try:
        for x3p_file in x3p_files:
            if not x3p_file.filename.endswith('.x3p'):
                raise HTTPException(status_code=400, detail=f"File {x3p_file.filename} must be in X3P format")
        
        result = await cartridge_service.analyze_batch(x3p_files)
        return result
    except Exception as e:
        logger.error(f"Cartridge batch analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@cartridge_router.post("/report")
async def generate_ballistics_report(
    x3p_file: UploadFile = File(..., description="Primary cartridge case X3P file"),
    compare_with: Optional[List[str]] = None
):
    """
    Generate comprehensive ballistics report for a cartridge case.
    
    Returns detailed forensic report with surface analysis, tool mark analysis,
    firearm identification, and comparison results.
    """
    try:
        if not x3p_file.filename.endswith('.x3p'):
            raise HTTPException(status_code=400, detail="File must be in X3P format")
        
        result = await cartridge_service.generate_ballistics_report(x3p_file, compare_with)
        return result
    except Exception as e:
        logger.error(f"Ballistics report generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@cartridge_router.get("/datasets")
async def get_cartridge_datasets():
    """Get information about available cartridge case datasets."""
    try:
        result = await cartridge_service.get_dataset_info()
        return result
    except Exception as e:
        logger.error(f"Failed to get cartridge datasets: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@cartridge_router.get("/model/info")
async def get_cartridge_model_info():
    """Get information about the cartridge case analysis model."""
    try:
        result = await cartridge_service.get_model_info()
        return result
    except Exception as e:
        logger.error(f"Failed to get cartridge model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@cartridge_router.post("/model/train")
async def train_cartridge_model(dataset_name: str = "combined"):
    """Train or retrain the cartridge case identification model."""
    try:
        result = await cartridge_service.train_model(dataset_name)
        return result
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# CROSS-MODAL ANALYSIS ENDPOINTS
# ============================================================================

@router.post("/cross-modal/analyze")
async def cross_modal_analysis(
    bloodsplatter_image: Optional[UploadFile] = File(None, description="Bloodsplatter image"),
    handwriting_image: Optional[UploadFile] = File(None, description="Handwriting sample"),
    cartridge_case: Optional[UploadFile] = File(None, description="Cartridge case X3P file")
):
    """
    Perform cross-modal analysis combining multiple types of unstructured evidence.
    
    Analyzes multiple evidence types and provides integrated forensic insights.
    """
    try:
        results = {}
        
        # Analyze bloodsplatter if provided
        if bloodsplatter_image:
            if not bloodsplatter_image.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail="Bloodsplatter file must be an image")
            results["bloodsplatter"] = await bloodsplatter_service.analyze_single_image(bloodsplatter_image)
        
        # Analyze handwriting if provided
        if handwriting_image:
            if not handwriting_image.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail="Handwriting file must be an image")
            results["handwriting"] = await handwriting_service.identify_writer(handwriting_image)
        
        # Analyze cartridge case if provided
        if cartridge_case:
            if not cartridge_case.filename.endswith('.x3p'):
                raise HTTPException(status_code=400, detail="Cartridge case file must be in X3P format")
            results["cartridge_case"] = await cartridge_service.analyze_single_case(cartridge_case)
        
        if not results:
            raise HTTPException(status_code=400, detail="At least one evidence type must be provided")
        
        # Generate cross-modal insights
        cross_modal_insights = _generate_cross_modal_insights(results)
        
        return {
            "individual_analyses": results,
            "cross_modal_insights": cross_modal_insights,
            "analysis_timestamp": "2025-01-XX"  # Replace with actual timestamp
        }
        
    except Exception as e:
        logger.error(f"Cross-modal analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def _generate_cross_modal_insights(results: Dict[str, Any]) -> List[str]:
    """Generate insights from cross-modal analysis."""
    insights = []
    
    # Analyze bloodsplatter + cartridge case combination
    if "bloodsplatter" in results and "cartridge_case" in results:
        bs_result = results["bloodsplatter"]
        cc_result = results["cartridge_case"]
        
        # Check for high-velocity bloodsplatter + firearm identification
        if (bs_result.get("rule_based_pattern") == "high_velocity" and
            cc_result.get("firearm_identification")):
            insights.append("High-velocity bloodsplatter pattern consistent with firearm discharge. Cartridge case analysis supports ballistics connection.")
    
    # Analyze handwriting + other evidence
    if "handwriting" in results:
        hw_result = results["handwriting"]
        if hw_result.get("top_predictions"):
            top_writer = hw_result["top_predictions"][0]
            insights.append(f"Handwriting analysis identified potential writer: {top_writer.get('writer_id', 'Unknown')} with {top_writer.get('confidence', 0):.2f} confidence.")
    
    # General multi-modal insights
    if len(results) > 1:
        insights.append(f"Multi-modal analysis combines {len(results)} different evidence types for comprehensive forensic assessment.")
    
    return insights

# ============================================================================
# SYSTEM STATUS ENDPOINTS
# ============================================================================

@router.get("/status")
async def get_unstructured_analysis_status():
    """Get status of all unstructured analysis services."""
    try:
        status = {
            "bloodsplatter_service": {
                "initialized": bloodsplatter_service.model_loaded,
                "available": True
            },
            "handwriting_service": {
                "initialized": handwriting_service.model_loaded,
                "available": True
            },
            "cartridge_service": {
                "initialized": cartridge_service.model_loaded,
                "available": True
            }
        }
        return status
    except Exception as e:
        logger.error(f"Failed to get service status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/capabilities")
async def get_analysis_capabilities():
    """Get comprehensive list of analysis capabilities."""
    try:
        capabilities = {
            "bloodsplatter_analysis": {
                "supported_formats": ["JPG", "PNG", "TIFF"],
                "max_file_size": "45MB",
                "capabilities": [
                    "Pattern classification",
                    "Impact angle estimation",
                    "Droplet analysis",
                    "Incident reconstruction",
                    "Velocity estimation",
                    "Weapon type suggestion"
                ]
            },
            "handwriting_analysis": {
                "supported_formats": ["PNG"],
                "capabilities": [
                    "Writer identification",
                    "Writer verification",
                    "Demographic analysis",
                    "Feature extraction",
                    "Cross-type consistency analysis"
                ]
            },
            "cartridge_case_analysis": {
                "supported_formats": ["X3P"],
                "capabilities": [
                    "3D surface analysis",
                    "Firearm identification",
                    "Case comparison",
                    "Tool mark analysis",
                    "Ballistics report generation"
                ]
            },
            "cross_modal_analysis": {
                "supported_combinations": [
                    "Bloodsplatter + Cartridge Case",
                    "Handwriting + Bloodsplatter",
                    "Handwriting + Cartridge Case",
                    "All three evidence types"
                ]
            }
        }
        return capabilities
    except Exception as e:
        logger.error(f"Failed to get capabilities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Include all sub-routers
router.include_router(bloodsplatter_router)
router.include_router(handwriting_router)
router.include_router(cartridge_router) 
# rom/api/config_endpoints.py
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Body, Query, Path, Depends
from fastapi.responses import JSONResponse
import json
import os
from typing import Dict, List, Any, Optional
from pydantic import BaseModel

from rom.config.config_manager import ConfigManager

# Initialize router
router = APIRouter(prefix="/api/config", tags=["configuration"])

# Initialize configuration manager
config_manager = ConfigManager()

# Pydantic models for request validation
class PoseConfig(BaseModel):
    model_type: Optional[str] = None
    det_frequency: Optional[int] = None
    tracking_mode: Optional[str] = None
    keypoint_likelihood_threshold: Optional[float] = None
    average_likelihood_threshold: Optional[float] = None
    keypoint_number_threshold: Optional[float] = None
    model_complexity: Optional[int] = None

class AssessmentConfig(BaseModel):
    ready_time_required: Optional[int] = None
    angle_buffer_size: Optional[int] = None
    position_tolerance: Optional[int] = None

class VisualizationConfig(BaseModel):
    theme: Optional[str] = None
    show_landmarks: Optional[bool] = None
    show_connections: Optional[bool] = None
    show_angles: Optional[bool] = None
    show_info_panel: Optional[bool] = None
    show_trajectory: Optional[bool] = None
    trajectory_length: Optional[int] = None
    highlight_primary_angle: Optional[bool] = None
    display_mode: Optional[str] = None
    font_size: Optional[float] = None
    line_thickness: Optional[int] = None

class ProcessingConfig(BaseModel):
    filter_type: Optional[str] = None
    interpolate: Optional[bool] = None
    max_gap_size: Optional[int] = None
    smoothing_window: Optional[int] = None
    butterworth_cutoff: Optional[float] = None
    butterworth_order: Optional[int] = None
    fps: Optional[float] = None

class TestConfig(BaseModel):
    joint_angles: Optional[List[str]] = None
    segment_angles: Optional[List[str]] = None
    primary_angle: Optional[str] = None
    target_rom: Optional[float] = None
    relevant_body_parts: Optional[List[str]] = None

class CustomTestConfig(BaseModel):
    name: str
    description: Optional[str] = None
    joint_angles: List[Dict[str, Any]]
    segment_angles: Optional[List[Dict[str, Any]]] = None
    body_parts: List[str]
    primary_angle: str
    target_rom: float

# Routes for managing configuration
@router.get("/")
async def get_all_config():
    """Get complete configuration."""
    return config_manager.get_complete_config()

@router.get("/pose")
async def get_pose_config():
    """Get pose detection configuration."""
    return config_manager.get_pose_config()

@router.put("/pose")
async def update_pose_config(config: PoseConfig):
    """Update pose detection configuration."""
    updates = {k: v for k, v in config.dict().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No updates provided")
    
    success = config_manager.update_section("pose", updates)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update configuration")
    
    return {"status": "success", "config": config_manager.get_pose_config()}

@router.get("/assessment")
async def get_assessment_config():
    """Get assessment configuration."""
    return config_manager.get_assessment_config()

@router.put("/assessment")
async def update_assessment_config(config: AssessmentConfig):
    """Update assessment configuration."""
    updates = {k: v for k, v in config.dict().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No updates provided")
    
    success = config_manager.update_section("assessment", updates)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update configuration")
    
    return {"status": "success", "config": config_manager.get_assessment_config()}

@router.get("/visualization")
async def get_visualization_config():
    """Get visualization configuration."""
    return config_manager.get_visualization_config()

@router.put("/visualization")
async def update_visualization_config(config: VisualizationConfig):
    """Update visualization configuration."""
    updates = {k: v for k, v in config.dict().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No updates provided")
    
    success = config_manager.update_section("visualization", updates)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update configuration")
    
    return {"status": "success", "config": config_manager.get_visualization_config()}

@router.get("/processing")
async def get_processing_config():
    """Get processing configuration."""
    return config_manager.get_processing_config()

@router.put("/processing")
async def update_processing_config(config: ProcessingConfig):
    """Update processing configuration."""
    updates = {k: v for k, v in config.dict().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No updates provided")
    
    success = config_manager.update_section("processing", updates)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update configuration")
    
    return {"status": "success", "config": config_manager.get_processing_config()}

@router.get("/test/{test_type}")
async def get_test_config(test_type: str):
    """Get configuration for a specific test type."""
    return config_manager.get_test_config(test_type)

@router.put("/test/{test_type}")
async def update_test_config(test_type: str, config: TestConfig, is_custom: bool = Query(False)):
    """Update configuration for a specific test type."""
    updates = {k: v for k, v in config.dict().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No updates provided")
    
    success = config_manager.update_test_config(test_type, updates, is_custom)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update configuration")
    
    return {"status": "success", "config": config_manager.get_test_config(test_type)}

@router.get("/tests")
async def get_available_tests():
    """Get all available tests (default and custom)."""
    test_defaults = config_manager.config.get("test_defaults", {})
    custom_tests = config_manager.config.get("custom_tests", {})
    
    return {
        "default_tests": list(test_defaults.keys()),
        "custom_tests": list(custom_tests.keys())
    }

@router.post("/custom_test")
async def create_custom_test(config: CustomTestConfig):
    """Create a new custom test configuration."""
    # Validate that primary angle exists in joint or segment angles
    primary_found = False
    
    for angle in config.joint_angles:
        if angle.get("name") == config.primary_angle:
            primary_found = True
            break
    
    if not primary_found and config.segment_angles:
        for angle in config.segment_angles:
            if angle.get("name") == config.primary_angle:
                primary_found = True
                break
    
    if not primary_found:
        raise HTTPException(
            status_code=400, 
            detail=f"Primary angle '{config.primary_angle}' not found in defined angles"
        )
    
    # Create test configuration
    test_id = config_manager.create_custom_test(
        config.name,
        {
            "name": config.name,
            "description": config.description,
            "joint_angles": config.joint_angles,
            "segment_angles": config.segment_angles or [],
            "body_parts": config.body_parts,
            "primary_angle": config.primary_angle,
            "target_rom": config.target_rom
        }
    )
    
    return {
        "status": "success",
        "test_id": test_id,
        "config": config_manager.get_test_config(test_id)
    }

@router.delete("/custom_test/{test_id}")
async def delete_custom_test(test_id: str):
    """Delete a custom test configuration."""
    success = config_manager.delete_custom_test(test_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Custom test '{test_id}' not found")
    
    return {"status": "success"}

@router.post("/import")
async def import_config(file: UploadFile = File(...)):
    """Import configuration from file."""
    # Save uploaded file temporarily
    temp_path = f"temp_config_{int(time.time())}.json"
    try:
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        
        # Import configuration
        success = config_manager.import_config(temp_path)
        if not success:
            raise HTTPException(status_code=400, detail="Invalid configuration file")
        
        return {"status": "success", "config": config_manager.get_complete_config()}
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@router.get("/export")
async def export_config():
    """Export current configuration as JSON."""
    return config_manager.get_complete_config()

@router.post("/reset")
async def reset_config():
    """Reset configuration to defaults."""
    success = config_manager.reset_to_defaults()
    if not success:
        raise HTTPException(status_code=500, detail="Failed to reset configuration")
    
    return {"status": "success", "config": config_manager.get_complete_config()}

@router.get("/presets")
async def get_presets():
    """Get list of available configuration presets."""
    # In a production system, you would scan a presets directory
    # For now, we'll return a hardcoded list
    return {
        "presets": [
            {
                "id": "default",
                "name": "Default",
                "description": "Standard configuration for ROM assessment."
            },
            {
                "id": "performance",
                "name": "Performance Mode",
                "description": "Optimized for speed with reduced visual elements."
            },
            {
                "id": "detail",
                "name": "Detail Mode",
                "description": "Comprehensive visualization with detailed analytics."
            }
        ]
    }

@router.post("/preset/{preset_id}")
async def apply_preset(preset_id: str):
    """Apply a configuration preset."""
    # In a production system, you would load the preset from a file
    # For now, we'll hardcode a few presets
    
    if preset_id == "default":
        # Reset to defaults
        config_manager.reset_to_defaults()
        return {"status": "success", "preset": "default"}
    
    elif preset_id == "performance":
        # Performance-oriented settings
        config_manager.update_section("pose", {
            "det_frequency": 10,
            "model_complexity": 0
        })
        
        config_manager.update_section("visualization", {
            "show_trajectory": False,
            "show_info_panel": False
        })
        
        return {"status": "success", "preset": "performance"}
    
    elif preset_id == "detail":
        # Detail-oriented settings
        config_manager.update_section("pose", {
            "det_frequency": 1,
            "model_complexity": 2,
            "keypoint_likelihood_threshold": 0.2
        })
        
        config_manager.update_section("visualization", {
            "show_landmarks": True,
            "show_connections": True,
            "show_angles": True,
            "show_info_panel": True,
            "show_trajectory": True,
            "trajectory_length": 200,
            "highlight_primary_angle": True
        })
        
        config_manager.update_section("processing", {
            "filter_type": "butterworth",
            "interpolate": True,
            "butterworth_order": 6,
            "butterworth_cutoff": 8.0
        })
        
        return {"status": "success", "preset": "detail"}
    
    else:
        raise HTTPException(status_code=404, detail=f"Preset '{preset_id}' not found")
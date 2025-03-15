# rom/api/main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Body
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional, Union
import uvicorn
import base64
import cv2
import numpy as np
import json
import os
import time
import logging
import httpx
from datetime import datetime

# Import ROM modules
from rom.api.config_endpoints import ConfigManager
from rom.core.base import AssessmentStatus, JointType
from rom.utils.pose_detector import PoseDetector
from rom.utils.visualization import EnhancedVisualizer
from rom.tests.lower_back_test import (
    EnhancedLowerBackFlexionTest, 
    EnhancedLowerBackExtensionTest,
    EnhancedLowerBackLateralFlexionTest,
    EnhancedLowerBackRotationTest
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rom-api")

# Define API models
class AssessmentConfig(BaseModel):
    """Configuration for ROM assessment."""
    test_type: str
    options: Dict[str, Any] = {}
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class AssessmentResult(BaseModel):
    """Result of ROM assessment."""
    test_type: str
    joint_type: str
    status: str
    angles: Dict[str, float] = {}
    rom: Optional[float] = None
    timestamp: str
    duration: Optional[float] = None
    metadata: Dict[str, Any] = {}


# Create FastAPI app
app = FastAPI(
    title="ROM Assessment API",
    description="API for Range of Motion assessment using pose estimation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize test factories
test_factories = {
    "lower_back_flexion": lambda pose_detector, visualizer, config: 
        EnhancedLowerBackFlexionTest(pose_detector, visualizer, config),
    "lower_back_extension": lambda pose_detector, visualizer, config: 
        EnhancedLowerBackExtensionTest(pose_detector, visualizer, config),
    "lower_back_lateral_flexion_left": lambda pose_detector, visualizer, config: 
        EnhancedLowerBackLateralFlexionTest(pose_detector, visualizer, config, side="left"),
    "lower_back_lateral_flexion_right": lambda pose_detector, visualizer, config: 
        EnhancedLowerBackLateralFlexionTest(pose_detector, visualizer, config, side="right"),
    "lower_back_rotation_left": lambda pose_detector, visualizer, config: 
        EnhancedLowerBackRotationTest(pose_detector, visualizer, config, side="left"),
    "lower_back_rotation_right": lambda pose_detector, visualizer, config: 
        EnhancedLowerBackRotationTest(pose_detector, visualizer, config, side="right")
}

# Track active connections
active_connections = {}

# LLM API integration
LLM_API_URL = os.environ.get("LLM_API_URL", "http://localhost:8001/analyze")

# Routes
@app.get("/api/tests")
async def get_available_tests():
    """Get a list of available ROM tests."""
    return {
        "tests": list(test_factories.keys()),
        "descriptions": {
            "lower_back_flexion": "Assessment of lower back forward bending motion",
            "lower_back_extension": "Assessment of lower back backward bending motion",
            "lower_back_lateral_flexion_left": "Assessment of lower back side bending to the left",
            "lower_back_lateral_flexion_right": "Assessment of lower back side bending to the right",
            "lower_back_rotation_left": "Assessment of lower back rotation to the left",
            "lower_back_rotation_right": "Assessment of lower back rotation to the right"
        }
    }


@app.post("/api/configure")
async def configure_assessment(config: AssessmentConfig):
    """Configure an assessment for later use."""
    if config.test_type not in test_factories:
        raise HTTPException(status_code=400, detail=f"Unsupported test type: {config.test_type}")
    
    # Generate session ID if not provided
    session_id = config.session_id or f"session_{int(time.time())}"
    
    # Store configuration for later use
    # In a production system, you would store this in a database
    
    return {
        "status": "success",
        "message": "Assessment configured successfully",
        "session_id": session_id
    }

@app.websocket("/api/assessment/{test_type}")
async def assessment_websocket(websocket: WebSocket, test_type: str):
    """WebSocket endpoint for real-time ROM assessment with full configuration support."""
    # Import asyncio at the handler level to ensure it's available
    import asyncio
    
    # Check if test type is valid
    is_custom_test = test_type.startswith("custom_")
    config_manager = ConfigManager()
    
    available_tests = list(config_manager.config.get("test_defaults", {}).keys())
    custom_tests = list(config_manager.config.get("custom_tests", {}).keys())
    
    if test_type not in available_tests and test_type not in custom_tests and test_type != "custom" and test_type not in test_factories:
        await websocket.close(code=1008, reason=f"Unsupported test type: {test_type}")
        return
    
    await websocket.accept()
    
    # Generate unique connection ID
    connection_id = f"conn_{int(time.time())}_{id(websocket)}"
    active_connections[connection_id] = websocket
    
    # Parse query parameters
    query_params = dict(websocket.query_params)
    
    # Extract configuration options from query parameters
    config_options = {}
    
    # Sports2D configuration options
    use_sports2d = query_params.get("use_sports2d", "true").lower() == "true"
    sports2d_mode = query_params.get("mode", "balanced")
    sports2d_model = query_params.get("pose_model", "body_with_feet")
    device = query_params.get("device", "auto")
    backend = query_params.get("backend", "auto")
    
    # Pose detection options
    if "model_type" in query_params:
        config_options["model_type"] = query_params["model_type"]
    if "det_frequency" in query_params:
        config_options["det_frequency"] = int(query_params["det_frequency"])
    if "tracking_mode" in query_params:
        config_options["tracking_mode"] = query_params["tracking_mode"]
    if "likelihood_threshold" in query_params:
        config_options["keypoint_likelihood_threshold"] = float(query_params["likelihood_threshold"])
    
    # Assessment options
    if "ready_time" in query_params:
        config_options["ready_time_required"] = int(query_params["ready_time"])
    
    # Visualization options
    if "theme" in query_params:
        config_options["theme"] = query_params["theme"]
    if "display_mode" in query_params:
        config_options["display_mode"] = query_params["display_mode"]
    if "show_trajectory" in query_params:
        config_options["show_trajectory"] = query_params["show_trajectory"].lower() == "true"
    if "font_size" in query_params:
        config_options["font_size"] = float(query_params["font_size"])
    
    # Processing options
    if "filter_type" in query_params:
        config_options["filter_type"] = query_params["filter_type"]
    if "interpolate" in query_params:
        config_options["interpolate"] = query_params["interpolate"].lower() == "true"
    
    # Custom body parts and angles
    body_parts = []
    if "body_parts" in query_params:
        body_parts = query_params["body_parts"].split(",")
        config_options["body_parts"] = body_parts
    
    joint_angles = []
    if "joint_angles" in query_params:
        # Format: "angle_name:point1,point2,point3;angle_name2:point1,point2,point3"
        angle_defs = query_params["joint_angles"].split(";")
        for angle_def in angle_defs:
            if ":" in angle_def:
                name, points_str = angle_def.split(":")
                points = points_str.split(",")
                if len(points) == 3:
                    joint_angles.append({
                        "name": name,
                        "points": points,
                        "type": "joint"
                    })
        
        if joint_angles:
            config_options["joint_angles"] = joint_angles
    
    segment_angles = []
    if "segment_angles" in query_params:
        # Format: "angle_name:point1,point2,reference;angle_name2:point1,point2,reference"
        angle_defs = query_params["segment_angles"].split(";")
        for angle_def in angle_defs:
            if ":" in angle_def:
                parts = angle_def.split(":")
                if len(parts) >= 2:
                    name = parts[0]
                    points_str = parts[1]
                    points = points_str.split(",")
                    
                    if len(points) >= 2:
                        segment_def = {
                            "name": name,
                            "points": points[:2],
                            "type": "segment"
                        }
                        
                        if len(points) > 2:
                            segment_def["reference"] = points[2]
                        
                        segment_angles.append(segment_def)
        
        if segment_angles:
            config_options["segment_angles"] = segment_angles
    
    # Primary angle
    if "primary_angle" in query_params:
        config_options["primary_angle"] = query_params["primary_angle"]
    elif joint_angles:
        config_options["primary_angle"] = joint_angles[0]["name"]
    
    # Initialize HTTP client for LLM API
    async_client = httpx.AsyncClient(timeout=10.0)
    
    try:
        logger.info(f"Started assessment session: {test_type} ({connection_id})")
        
        # Initialize visualizer with theme from config
        theme = config_options.get("theme", "dark")
        from rom.utils.visualization import EnhancedVisualizer
        visualizer = EnhancedVisualizer(theme=theme)
        
        # Try to initialize the pose detector with error handling
        try:
            pose_detector = PoseDetector(
                model_type=config_options.get("model_type", sports2d_model),
                det_frequency=config_options.get("det_frequency", 4),
                tracking_mode=config_options.get("tracking_mode", "sports2d"),
                keypoint_likelihood_threshold=config_options.get("keypoint_likelihood_threshold", 0.3),
                average_likelihood_threshold=config_options.get("average_likelihood_threshold", 0.5),
                keypoint_number_threshold=config_options.get("keypoint_number_threshold", 0.3),
                mode=sports2d_mode, 
                backend=backend,
                device=device,
                use_sports2d=use_sports2d
            )
            
            # Verify the detector works by processing a test frame
            test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            _ = pose_detector.detect_pose(test_frame)
            logger.info("Pose detector initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing pose detector: {e}")
            # Fall back to a simpler pose detector if available
            try:
                from rom.utils.fallback_pose_detector import FallbackPoseDetector
                pose_detector = FallbackPoseDetector()
                logger.info("Using fallback pose detector due to initialization error")
            except ImportError:
                logger.error("Fallback pose detector not available")
                await websocket.send_text(json.dumps({
                    "error": f"Failed to initialize pose detector: {str(e)}",
                    "message": "Try disabling Sports2D or using a different model"
                }))
                await websocket.close(code=1011, reason="Failed to initialize pose detection")
                return
        
        # Create appropriate test instance
        if test_type == "custom" or body_parts or joint_angles or segment_angles:
            # Create a fully configurable test
            from rom.core.configurable_test import ConfigurableROMTest
            
            test_instance = ConfigurableROMTest(
                pose_detector=pose_detector,
                visualizer=visualizer,
                config=config_options,
                test_type="custom",
                config_manager=config_manager
            )
        elif is_custom_test:
            # Load a saved custom test
            custom_config = config_manager.get_test_config(test_type)
            custom_config.update(config_options)
            
            from rom.core.configurable_test import ConfigurableROMTest
            
            test_instance = ConfigurableROMTest(
                pose_detector=pose_detector,
                visualizer=visualizer,
                config=custom_config,
                test_type=test_type,
                config_manager=config_manager
            )
        else:
            # Use standard test from the factories
            test_config = config_manager.get_test_config(test_type)
            test_config.update(config_options)
            
            if test_type not in test_factories:
                # Fall back to configurable test for unknown test types
                from rom.core.configurable_test import ConfigurableROMTest
                
                test_instance = ConfigurableROMTest(
                    pose_detector=pose_detector,
                    visualizer=visualizer,
                    config=test_config,
                    test_type=test_type,
                    config_manager=config_manager
                )
            else:
                # Use registered test factory
                test_instance = test_factories[test_type](pose_detector, visualizer, test_config)
        
        # Helper function to process frames with timeout
        async def process_frame_with_timeout(frame, timeout=5.0):
            try:
                # Create a wrapper function that runs in a thread pool
                def process_wrapper():
                    return test_instance.process_frame(frame)
                
                # Run the processing with a timeout
                return await asyncio.wait_for(
                    asyncio.to_thread(process_wrapper),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.error(f"Frame processing timed out after {timeout} seconds")
                # Return original frame and error status
                return frame.copy(), {"status": "error", "error": "Processing timeout"}
            except Exception as e:
                logger.error(f"Error in frame processing: {str(e)}")
                # Return original frame and error status
                return frame.copy(), {"status": "error", "error": str(e)}
        
        # Process frames from WebSocket
        while True:
            try:
                # Receive frame from client
                data = await websocket.receive_text()
                
                # Decode base64 image
                if not data.startswith("data:image"):
                    continue
                
                image_data = base64.b64decode(data.split(",")[1])
                nparr = np.frombuffer(image_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None or frame.size == 0:
                    logger.warning("Received empty frame")
                    continue
                
                # Process frame with timeout protection
                logger.info(f"Processing frame: shape={frame.shape}")
                processed_frame, rom_data = await process_frame_with_timeout(frame)
                logger.info(f"Frame processed: status={rom_data.get('status', 'unknown')}")
                
                # Ensure we have a valid processed frame
                if processed_frame is None or processed_frame.size == 0:
                    logger.error("Processed frame is None or empty")
                    processed_frame = frame.copy()
                
                # Encode processed frame
                try:
                    _, buffer = cv2.imencode(".jpg", processed_frame)
                    processed_b64 = base64.b64encode(buffer).decode("utf-8")
                    logger.info(f"Image encoded, size: {len(processed_b64)} bytes")
                except Exception as encode_error:
                    logger.error(f"Error encoding processed frame: {str(encode_error)}")
                    # Use original frame as fallback
                    _, buffer = cv2.imencode(".jpg", frame)
                    processed_b64 = base64.b64encode(buffer).decode("utf-8")
                    logger.info(f"Using original frame as fallback, size: {len(processed_b64)} bytes")
                
                # Prepare response data
                response_data = {
                    "image": f"data:image/jpeg;base64,{processed_b64}",
                    "rom_data": rom_data
                }
                
                # If assessment is completed, send data to LLM API for analysis
                if rom_data.get("status") == "completed" and "rom" in rom_data:
                    try:
                        # Create comprehensive assessment data
                        assessment_data = {
                            "test_type": test_type,
                            "timestamp": datetime.now().isoformat(),
                            "rom_data": rom_data,
                            "config": {
                                k: v for k, v in config_options.items() 
                                if k not in ["joint_angles", "segment_angles", "body_parts"]
                            }
                        }
                        
                        # Generate analysis if analyzer is available
                        try:
                            from rom.analysis.assessment_analyzer import AssessmentAnalyzer
                            analyzer = AssessmentAnalyzer()
                            
                            # Get angle history from test instance
                            angle_history = {}
                            if hasattr(test_instance, "data_processor"):
                                for angle_name in test_instance.data_processor.angle_data:
                                    angle_history[angle_name] = list(test_instance.data_processor.angle_data[angle_name].angle_history)
                            
                            analysis_result = analyzer.analyze_assessment(
                                rom_data,
                                angle_history
                            )
                            
                            response_data["analysis"] = analysis_result
                        except ImportError:
                            logger.warning("AssessmentAnalyzer not available, skipping analysis")
                        
                        # Send to LLM API for additional insights
                        try:
                            llm_response = await async_client.post(
                                LLM_API_URL,
                                json=assessment_data
                            )
                            
                            if llm_response.status_code == 200:
                                response_data["llm_analysis"] = llm_response.json()
                            else:
                                logger.warning(f"LLM API returned status code {llm_response.status_code}")
                                response_data["llm_analysis"] = {
                                    "error": "Failed to get analysis from LLM API",
                                    "status_code": llm_response.status_code
                                }
                        except Exception as e:
                            logger.error(f"Error contacting LLM API: {str(e)}")
                            response_data["llm_analysis"] = {
                                "error": "Failed to get analysis from LLM API",
                                "message": str(e)
                            }
                        
                        # Generate visualization report if configured
                        if visualizer and hasattr(visualizer, "create_report_image"):
                            report_img = visualizer.create_report_image(
                                rom_data,
                                angle_history if 'angle_history' in locals() else {},
                                include_plots=True
                            )
                            
                            if report_img is not None:
                                _, report_buffer = cv2.imencode(".jpg", report_img)
                                report_b64 = base64.b64encode(report_buffer).decode("utf-8")
                                response_data["report_image"] = f"data:image/jpeg;base64,{report_b64}"
                    
                    except Exception as analysis_error:
                        logger.error(f"Error generating analysis: {str(analysis_error)}")
                        response_data["analysis_error"] = str(analysis_error)
                
                # Send response
                await websocket.send_text(json.dumps(response_data))
                
                # Add a small delay to prevent overwhelming the CPU
                await asyncio.sleep(0.05)
            
            except WebSocketDisconnect:
                logger.info(f"Client disconnected: {connection_id}")
                break
            
            except Exception as e:
                logger.error(f"Error processing frame: {str(e)}")
                try:
                    await websocket.send_text(json.dumps({
                        "error": str(e),
                        "message": "Error processing frame"
                    }))
                except:
                    break
    
    except Exception as e:
        logger.error(f"Unexpected error in WebSocket handler: {str(e)}")
    
    finally:
        # Clean up
        if connection_id in active_connections:
            del active_connections[connection_id]
        
        try:
            await async_client.aclose()
        except:
            pass
        
        try:
            await websocket.close()
        except:
            pass
        
        logger.info(f"Closed assessment session: {test_type} ({connection_id})")

async def send_processed_frame(websocket, processed_frame, rom_data):
    """Helper function to send a processed frame and ROM data to the client."""
    try:
        # Encode processed frame
        _, buffer = cv2.imencode(".jpg", processed_frame)
        processed_b64 = base64.b64encode(buffer).decode("utf-8")
        
        # Prepare response data
        response_data = {
            "image": f"data:image/jpeg;base64,{processed_b64}",
            "rom_data": rom_data
        }
        
        # If assessment is completed, add analysis
        if rom_data.get("status") == "completed" and "rom" in rom_data:
            try:
                # Create comprehensive assessment data
                assessment_data = {
                    "timestamp": datetime.now().isoformat(),
                    "rom_data": rom_data
                }
                
                # Generate analysis if analyzer is available
                try:
                    from rom.analysis.assessment_analyzer import AssessmentAnalyzer
                    analyzer = AssessmentAnalyzer()
                    
                    # Get angle history from test instance
                    angle_history = {}
                    
                    analysis_result = analyzer.analyze_assessment(
                        rom_data,
                        angle_history
                    )
                    
                    response_data["analysis"] = analysis_result
                except ImportError:
                    logger.warning("AssessmentAnalyzer not available")
                
            except Exception as analysis_error:
                logger.error(f"Error generating analysis: {str(analysis_error)}")
                response_data["analysis_error"] = str(analysis_error)
        
        # Send response
        await websocket.send_text(json.dumps(response_data))
    
    except Exception as e:
        logger.error(f"Error sending processed frame: {str(e)}")
        raise


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "version": "1.0.0",
        "active_connections": len(active_connections)
    }

@app.post("/api/analyze")
async def analyze_assessment(data: Dict[str, Any]):
    """
    Analyze completed assessment data without performing the assessment.
    
    This endpoint is useful for integrating with the chatbot after an assessment 
    has already been completed.
    """
    if "test_type" not in data or "rom_data" not in data:
        raise HTTPException(status_code=400, detail="Missing required fields test_type or rom_data")
    
    test_type = data["test_type"]
    rom_data = data["rom_data"]
    
    # Validate test type
    if test_type not in test_factories:
        raise HTTPException(status_code=400, detail=f"Unsupported test type: {test_type}")
    
    # Validate ROM data
    if not isinstance(rom_data, dict) or "rom" not in rom_data:
        raise HTTPException(status_code=400, detail="Invalid ROM data format")
    
    try:
        # Send to LLM API for analysis
        async with httpx.AsyncClient(timeout=10.0) as client:
            llm_response = await client.post(
                LLM_API_URL,
                json={
                    "assessment_type": test_type,
                    "rom_data": rom_data,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            if llm_response.status_code == 200:
                analysis = llm_response.json()
            else:
                logger.warning(f"LLM API returned status code {llm_response.status_code}")
                raise HTTPException(
                    status_code=500, 
                    detail=f"LLM API returned error: {llm_response.status_code}"
                )
        
        # Return assessment result with analysis
        return {
            "test_type": test_type,
            "rom_data": rom_data,
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }
    
    except httpx.RequestError as e:
        logger.error(f"Error contacting LLM API: {str(e)}")
        raise HTTPException(status_code=503, detail="LLM API service unavailable")
    
    except Exception as e:
        logger.error(f"Error analyzing assessment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import argparse
    import asyncio
    
    parser = argparse.ArgumentParser(description="ROM Assessment API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    uvicorn.run(
        "rom.api.main:app", 
        host=args.host, 
        port=args.port, 
        reload=args.reload,
        log_level="info" if not args.debug else "debug",
        access_log=True
    )
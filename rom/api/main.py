# rom/api/main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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
from rom.core.base import AssessmentStatus, JointType
from rom.utils.pose_detector import PoseDetector
from rom.utils.visualization import PoseVisualizer
from rom.tests.lower_back_test import (
    LowerBackFlexionTest, 
    LowerBackExtensionTest,
    LowerBackLateralFlexionTest,
    LowerBackRotationTest
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
        LowerBackFlexionTest(pose_detector, visualizer, config),
    "lower_back_extension": lambda pose_detector, visualizer, config: 
        LowerBackExtensionTest(pose_detector, visualizer, config),
    "lower_back_lateral_flexion_left": lambda pose_detector, visualizer, config: 
        LowerBackLateralFlexionTest(pose_detector, visualizer, config, side="left"),
    "lower_back_lateral_flexion_right": lambda pose_detector, visualizer, config: 
        LowerBackLateralFlexionTest(pose_detector, visualizer, config, side="right"),
    "lower_back_rotation_left": lambda pose_detector, visualizer, config: 
        LowerBackRotationTest(pose_detector, visualizer, config, side="left"),
    "lower_back_rotation_right": lambda pose_detector, visualizer, config: 
        LowerBackRotationTest(pose_detector, visualizer, config, side="right")
}

# Track active connections
active_connections = {}

# LLM API integration
LLM_API_URL = os.environ.get("LLM_API_URL", "http://localhost:8001/analyze")


# HTML content for the WebSocket client
html_client = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ROM Assessment</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .header {
            background-color: #333;
            color: white;
            padding: 20px;
            border-radius: 8px;
        }
        .video-container {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }
        .video-box {
            flex: 1;
            min-width: 480px;
            background-color: #fff;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 15px;
        }
        .data-container {
            flex: 1;
            min-width: 320px;
            background-color: #fff;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 15px;
        }
        .controls {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
        }
        select, button {
            padding: 10px;
            border-radius: 4px;
            border: 1px solid #ddd;
        }
        button {
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        button:hover {
            background-color: #45a049;
        }
        button:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
        video, canvas {
            width: 100%;
            border-radius: 4px;
            background-color: #000;
        }
        pre {
            background-color: #f8f8f8;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
            white-space: pre-wrap;
        }
        .status {
            font-weight: bold;
            margin-bottom: 10px;
        }
        .status.connected {
            color: #4CAF50;
        }
        .status.disconnected {
            color: #f44336;
        }
        .footer {
            margin-top: 30px;
            text-align: center;
            color: #666;
        }
        .permission-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: rgba(0,0,0,0.8);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            z-index: 1000;
            color: white;
            text-align: center;
            padding: 20px;
        }
        .permission-overlay button {
            margin-top: 20px;
            padding: 15px 30px;
            font-size: 18px;
        }
    </style>
</head>
<body>
    <div id="permissionOverlay" class="permission-overlay">
        <h2>Camera Access Required</h2>
        <p>This application needs access to your camera to perform ROM assessment.</p>
        <p>Please click the button below and accept the camera permission request.</p>
        <button id="requestPermission">Grant Camera Access</button>
    </div>

    <div class="container">
        <div class="header">
            <h1>ROM Assessment</h1>
            <p>Select an assessment type and follow the on-screen instructions.</p>
        </div>
        
        <div class="video-container">
            <div class="video-box">
                <div class="controls">
                    <select id="testType">
                        <option value="lower_back_flexion">Lower Back Flexion</option>
                        <option value="lower_back_extension">Lower Back Extension</option>
                        <option value="lower_back_lateral_flexion_left">Lower Back Lateral Flexion (Left)</option>
                        <option value="lower_back_lateral_flexion_right">Lower Back Lateral Flexion (Right)</option>
                        <option value="lower_back_rotation_left">Lower Back Rotation (Left)</option>
                        <option value="lower_back_rotation_right">Lower Back Rotation (Right)</option>
                    </select>
                    <button id="startAssessment">Start Assessment</button>
                    <button id="stopAssessment" disabled>Stop</button>
                </div>
                <div class="status disconnected" id="connectionStatus">Status: Disconnected</div>
                <video id="videoInput" autoplay playsinline style="display:none;"></video>
                <canvas id="canvasOutput"></canvas>
            </div>
            
            <div class="data-container">
                <h3>Assessment Data</h3>
                <pre id="assessmentData">No data available</pre>
            </div>
        </div>
        
        <div class="footer">
            <p>ROM Assessment System &copy; 2025</p>
        </div>
    </div>

    <script>
        const video = document.getElementById('videoInput');
        const canvas = document.getElementById('canvasOutput');
        const ctx = canvas.getContext('2d');
        const statusElement = document.getElementById('connectionStatus');
        const dataElement = document.getElementById('assessmentData');
        const testTypeSelect = document.getElementById('testType');
        const startButton = document.getElementById('startAssessment');
        const stopButton = document.getElementById('stopAssessment');
        const permissionOverlay = document.getElementById('permissionOverlay');
        const requestPermissionButton = document.getElementById('requestPermission');
        
        let ws = null;
        let stream = null;
        let assessmentRunning = false;
        
        // Handle permission request
        requestPermissionButton.addEventListener('click', async () => {
            try {
                stream = await navigator.mediaDevices.getUserMedia({ 
                    video: { 
                        width: { ideal: 1280 },
                        height: { ideal: 720 },
                        facingMode: 'user'
                    } 
                });
                video.srcObject = stream;
                permissionOverlay.style.display = 'none';
                
                // Set canvas dimensions based on video
                video.addEventListener('loadedmetadata', () => {
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                });
            } catch (err) {
                console.error('Error accessing camera:', err);
                alert('Camera access is required for this application. Please allow camera access and refresh the page.');
            }
        });
        
        // Start assessment
        startButton.addEventListener('click', () => {
            if (!stream) {
                alert('Camera access is required. Please grant permission first.');
                return;
            }
            
            const testType = testTypeSelect.value;
            startAssessment(testType);
        });
        
        // Stop assessment
        stopButton.addEventListener('click', () => {
            stopAssessment();
        });
        
        async function startAssessment(testType) {
            if (assessmentRunning) {
                stopAssessment();
            }
            
            try {
                // Connect to WebSocket
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/api/assessment/${testType}`;
                
                ws = new WebSocket(wsUrl);
                
                ws.onopen = () => {
                    statusElement.textContent = 'Status: Connected';
                    statusElement.className = 'status connected';
                    assessmentRunning = true;
                    startButton.disabled = true;
                    stopButton.disabled = false;
                    sendFrames();
                };
                
                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    
                    // Update canvas with processed frame
                    if (data.image) {
                        const image = new Image();
                        image.onload = () => {
                            ctx.clearRect(0, 0, canvas.width, canvas.height);
                            ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
                        };
                        image.src = data.image;
                    }
                    
                    // Update assessment data
                    if (data.rom_data) {
                        dataElement.textContent = JSON.stringify(data.rom_data, null, 2);
                        
                        // If assessment is completed, stop it
                        if (data.rom_data.status === 'completed') {
                            setTimeout(() => {
                                if (assessmentRunning) {
                                    stopAssessment();
                                    alert('Assessment completed successfully!');
                                }
                            }, 2000);
                        }
                    }
                };
                
                ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    statusElement.textContent = 'Status: Error';
                    statusElement.className = 'status disconnected';
                };
                
                ws.onclose = () => {
                    statusElement.textContent = 'Status: Disconnected';
                    statusElement.className = 'status disconnected';
                    assessmentRunning = false;
                    startButton.disabled = false;
                    stopButton.disabled = true;
                };
            } catch (error) {
                console.error('Error starting assessment:', error);
                alert('Failed to start assessment. Please try again.');
            }
        }
        
        function stopAssessment() {
            if (ws) {
                ws.close();
            }
            assessmentRunning = false;
            startButton.disabled = false;
            stopButton.disabled = true;
        }
        
        function sendFrames() {
            if (!assessmentRunning || !ws) return;
            
            // Capture frame from video and send to server
            const canvasTmp = document.createElement('canvas');
            canvasTmp.width = video.videoWidth;
            canvasTmp.height = video.videoHeight;
            const ctxTmp = canvasTmp.getContext('2d');
            
            ctxTmp.drawImage(video, 0, 0, canvasTmp.width, canvasTmp.height);
            const frameData = canvasTmp.toDataURL('image/jpeg', 0.8);
            
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(frameData);
            }
            
            // Schedule next frame
            setTimeout(sendFrames, 100); // 10 FPS
        }
    </script>
</body>
</html>
"""


# Routes
@app.get("/")
async def get_home():
    """Return the home page with WebSocket client."""
    return HTMLResponse(html_client)


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
    """WebSocket endpoint for real-time ROM assessment."""
    if test_type not in test_factories:
        await websocket.close(code=1008, reason=f"Unsupported test type: {test_type}")
        return
    
    await websocket.accept()
    
    # Generate unique connection ID
    connection_id = f"conn_{int(time.time())}_{id(websocket)}"
    active_connections[connection_id] = websocket
    
    # Initialize components
    pose_detector = PoseDetector()
    visualizer = PoseVisualizer()
    
    # Create appropriate test instance
    test_instance = test_factories[test_type](pose_detector, visualizer, {})
    
    # Initialize HTTP client for LLM API
    async_client = httpx.AsyncClient(timeout=10.0)
    
    try:
        logger.info(f"Started assessment session: {test_type} ({connection_id})")
        
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
                    continue
                
                # Process frame with appropriate test
                processed_frame, rom_data = test_instance.process_frame(frame)
                
                # Encode processed frame
                _, buffer = cv2.imencode(".jpg", processed_frame)
                processed_b64 = base64.b64encode(buffer).decode("utf-8")
                
                # Prepare response
                response_data = {
                    "image": f"data:image/jpeg;base64,{processed_b64}",
                    "rom_data": rom_data
                }
                
                # If assessment is completed, send data to LLM API
                if rom_data.get("status") == "completed" and "rom" in rom_data:
                    try:
                        llm_response = await async_client.post(
                            LLM_API_URL,
                            json={
                                "assessment_type": test_type,
                                "rom_data": rom_data,
                                "timestamp": datetime.now().isoformat()
                            }
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
                
                # Send response
                await websocket.send_text(json.dumps(response_data))
                
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


# For direct execution
if __name__ == "__main__":
    import argparse
    
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
        log_level="info",
        access_log=True
    )
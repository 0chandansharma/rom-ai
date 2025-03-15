# rom/api/main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Body
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
from rom.api.config_endpoints import ConfigManager
from rom.core.base import AssessmentStatus, JointType
from rom.utils.pose_detector import PoseDetector
from rom.utils.visualization import EnhancedVisualizer
from rom.tests.lower_back_test import (
    EnhancedLowerBackFlexionTest, 
    EnhancedLowerBackExtensionTest,
    EnhancedLowerBackLateralFlexionTest,
    EnhancedLowerBackRotationTest
    # LowerBackFlexionTest, 
    # LowerBackExtensionTest,
    # LowerBackLateralFlexionTest,
    # LowerBackRotationTest
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


# HTML content for the WebSocket client
# html_client = """
# <!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <title>ROM Assessment</title>
#     <style>
#         body { 
#             font-family: Arial, sans-serif; 
#             margin: 0; 
#             padding: 20px; 
#             background-color: #f5f5f5;
#         }
#         .container {
#             max-width: 1200px;
#             margin: 0 auto;
#             display: flex;
#             flex-direction: column;
#             gap: 20px;
#         }
#         .header {
#             background-color: #333;
#             color: white;
#             padding: 20px;
#             border-radius: 8px;
#         }
#         .video-container {
#             display: flex;
#             gap: 20px;
#             flex-wrap: wrap;
#         }
#         .video-box {
#             flex: 1;
#             min-width: 480px;
#             background-color: #fff;
#             border-radius: 8px;
#             box-shadow: 0 2px 10px rgba(0,0,0,0.1);
#             padding: 15px;
#         }
#         .data-container {
#             flex: 1;
#             min-width: 320px;
#             background-color: #fff;
#             border-radius: 8px;
#             box-shadow: 0 2px 10px rgba(0,0,0,0.1);
#             padding: 15px;
#         }
#         .controls {
#             display: flex;
#             gap: 10px;
#             margin-bottom: 15px;
#         }
#         select, button {
#             padding: 10px;
#             border-radius: 4px;
#             border: 1px solid #ddd;
#         }
#         button {
#             background-color: #4CAF50;
#             color: white;
#             border: none;
#             cursor: pointer;
#             transition: background-color 0.3s;
#         }
#         button:hover {
#             background-color: #45a049;
#         }
#         button:disabled {
#             background-color: #cccccc;
#             cursor: not-allowed;
#         }
#         video, canvas {
#             width: 100%;
#             border-radius: 4px;
#             background-color: #000;
#         }
#         pre {
#             background-color: #f8f8f8;
#             padding: 10px;
#             border-radius: 4px;
#             overflow-x: auto;
#             white-space: pre-wrap;
#         }
#         .status {
#             font-weight: bold;
#             margin-bottom: 10px;
#         }
#         .status.connected {
#             color: #4CAF50;
#         }
#         .status.disconnected {
#             color: #f44336;
#         }
#         .footer {
#             margin-top: 30px;
#             text-align: center;
#             color: #666;
#         }
#         .permission-overlay {
#             position: fixed;
#             top: 0;
#             left: 0;
#             right: 0;
#             bottom: 0;
#             background-color: rgba(0,0,0,0.8);
#             display: flex;
#             flex-direction: column;
#             align-items: center;
#             justify-content: center;
#             z-index: 1000;
#             color: white;
#             text-align: center;
#             padding: 20px;
#         }
#         .permission-overlay button {
#             margin-top: 20px;
#             padding: 15px 30px;
#             font-size: 18px;
#         }
#     </style>
# </head>
# <body>
#     <div id="permissionOverlay" class="permission-overlay">
#         <h2>Camera Access Required</h2>
#         <p>This application needs access to your camera to perform ROM assessment.</p>
#         <p>Please click the button below and accept the camera permission request.</p>
#         <button id="requestPermission">Grant Camera Access</button>
#     </div>

#     <div class="container">
#         <div class="header">
#             <h1>ROM Assessment</h1>
#             <p>Select an assessment type and follow the on-screen instructions.</p>
#         </div>
        
#         <div class="video-container">
#             <div class="video-box">
#                 <div class="controls">
#                     <select id="testType">
#                         <option value="lower_back_flexion">Lower Back Flexion</option>
#                         <option value="lower_back_extension">Lower Back Extension</option>
#                         <option value="lower_back_lateral_flexion_left">Lower Back Lateral Flexion (Left)</option>
#                         <option value="lower_back_lateral_flexion_right">Lower Back Lateral Flexion (Right)</option>
#                         <option value="lower_back_rotation_left">Lower Back Rotation (Left)</option>
#                         <option value="lower_back_rotation_right">Lower Back Rotation (Right)</option>
#                     </select>
#                     <button id="startAssessment">Start Assessment</button>
#                     <button id="stopAssessment" disabled>Stop</button>
#                 </div>
#                 <div class="status disconnected" id="connectionStatus">Status: Disconnected</div>
#                 <video id="videoInput" autoplay playsinline style="display:none;"></video>
#                 <canvas id="canvasOutput"></canvas>
#             </div>
            
#             <div class="data-container">
#                 <h3>Assessment Data</h3>
#                 <pre id="assessmentData">No data available</pre>
#             </div>
#         </div>
        
#         <div class="footer">
#             <p>ROM Assessment System &copy; 2025</p>
#         </div>
#     </div>

#     <script>
#         const video = document.getElementById('videoInput');
#         const canvas = document.getElementById('canvasOutput');
#         const ctx = canvas.getContext('2d');
#         const statusElement = document.getElementById('connectionStatus');
#         const dataElement = document.getElementById('assessmentData');
#         const testTypeSelect = document.getElementById('testType');
#         const startButton = document.getElementById('startAssessment');
#         const stopButton = document.getElementById('stopAssessment');
#         const permissionOverlay = document.getElementById('permissionOverlay');
#         const requestPermissionButton = document.getElementById('requestPermission');
        
#         let ws = null;
#         let stream = null;
#         let assessmentRunning = false;
        
#         // Handle permission request
#         requestPermissionButton.addEventListener('click', async () => {
#             try {
#                 stream = await navigator.mediaDevices.getUserMedia({ 
#                     video: { 
#                         width: { ideal: 1280 },
#                         height: { ideal: 720 },
#                         facingMode: 'user'
#                     } 
#                 });
#                 video.srcObject = stream;
#                 permissionOverlay.style.display = 'none';
                
#                 // Set canvas dimensions based on video
#                 video.addEventListener('loadedmetadata', () => {
#                     canvas.width = video.videoWidth;
#                     canvas.height = video.videoHeight;
#                 });
#             } catch (err) {
#                 console.error('Error accessing camera:', err);
#                 alert('Camera access is required for this application. Please allow camera access and refresh the page.');
#             }
#         });
        
#         // Start assessment
#         startButton.addEventListener('click', () => {
#             if (!stream) {
#                 alert('Camera access is required. Please grant permission first.');
#                 return;
#             }
            
#             const testType = testTypeSelect.value;
#             startAssessment(testType);
#         });
        
#         // Stop assessment
#         stopButton.addEventListener('click', () => {
#             stopAssessment();
#         });
        
#         async function startAssessment(testType) {
#             if (assessmentRunning) {
#                 stopAssessment();
#             }
            
#             try {
#                 // Connect to WebSocket
#                 const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
#                 const wsUrl = `${protocol}//${window.location.host}/api/assessment/${testType}`;
                
#                 ws = new WebSocket(wsUrl);
                
#                 ws.onopen = () => {
#                     statusElement.textContent = 'Status: Connected';
#                     statusElement.className = 'status connected';
#                     assessmentRunning = true;
#                     startButton.disabled = true;
#                     stopButton.disabled = false;
#                     sendFrames();
#                 };
                
#                 ws.onmessage = (event) => {
#                     const data = JSON.parse(event.data);
                    
#                     // Update canvas with processed frame
#                     if (data.image) {
#                         const image = new Image();
#                         image.onload = () => {
#                             ctx.clearRect(0, 0, canvas.width, canvas.height);
#                             ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
#                         };
#                         image.src = data.image;
#                     }
                    
#                     // Update assessment data
#                     if (data.rom_data) {
#                         dataElement.textContent = JSON.stringify(data.rom_data, null, 2);
                        
#                         // If assessment is completed, stop it
#                         if (data.rom_data.status === 'completed') {
#                             setTimeout(() => {
#                                 if (assessmentRunning) {
#                                     stopAssessment();
#                                     alert('Assessment completed successfully!');
#                                 }
#                             }, 2000);
#                         }
#                     }
#                 };
                
#                 ws.onerror = (error) => {
#                     console.error('WebSocket error:', error);
#                     statusElement.textContent = 'Status: Error';
#                     statusElement.className = 'status disconnected';
#                 };
                
#                 ws.onclose = () => {
#                     statusElement.textContent = 'Status: Disconnected';
#                     statusElement.className = 'status disconnected';
#                     assessmentRunning = false;
#                     startButton.disabled = false;
#                     stopButton.disabled = true;
#                 };
#             } catch (error) {
#                 console.error('Error starting assessment:', error);
#                 alert('Failed to start assessment. Please try again.');
#             }
#         }
        
#         function stopAssessment() {
#             if (ws) {
#                 ws.close();
#             }
#             assessmentRunning = false;
#             startButton.disabled = false;
#             stopButton.disabled = true;
#         }
        
#         function sendFrames() {
#             if (!assessmentRunning || !ws) return;
            
#             // Capture frame from video and send to server
#             const canvasTmp = document.createElement('canvas');
#             canvasTmp.width = video.videoWidth;
#             canvasTmp.height = video.videoHeight;
#             const ctxTmp = canvasTmp.getContext('2d');
            
#             ctxTmp.drawImage(video, 0, 0, canvasTmp.width, canvasTmp.height);
#             const frameData = canvasTmp.toDataURL('image/jpeg', 0.8);
            
#             if (ws.readyState === WebSocket.OPEN) {
#                 ws.send(frameData);
#             }
            
#             // Schedule next frame
#             setTimeout(sendFrames, 100); // 10 FPS
#         }
#         <!-- Add this to your web interface HTML in rom/api/main.py -->
# <div class="settings-panel">
#     <h3>Advanced Settings</h3>
#     <div class="setting-row">
#         <label for="use_sports2d">Use Sports2D:</label>
#         <input type="checkbox" id="use_sports2d" checked>
#     </div>
#     <div class="setting-row">
#         <label for="mode">Mode:</label>
#         <select id="mode">
#             <option value="lightweight">Lightweight (Faster)</option>
#             <option value="balanced" selected>Balanced</option>
#             <option value="performance">Performance (Accurate)</option>
#         </select>
#     </div>
#     <div class="setting-row">
#         <label for="pose_model">Pose Model:</label>
#         <select id="pose_model">
#             <option value="body_with_feet" selected>Body with Feet</option>
#             <option value="coco_133_wrist">Body, Feet & Hands</option>
#             <option value="coco_133">Body, Feet, Hands & Face</option>
#             <option value="coco_17">Body Only</option>
#         </select>
#     </div>
#     <div class="setting-row">
#         <label for="det_frequency">Detection Frequency:</label>
#         <input type="number" id="det_frequency" min="1" max="30" value="4">
#     </div>
#     <button id="apply_settings">Apply Settings</button>
# </div>

# <script>
#     // Add event listener to the apply settings button
#     document.getElementById('apply_settings').addEventListener('click', function() {
#         // Gather settings
#         const useSports2d = document.getElementById('use_sports2d').checked;
#         const mode = document.getElementById('mode').value;
#         const poseModel = document.getElementById('pose_model').value;
#         const detFrequency = document.getElementById('det_frequency').value;
        
#         // Build WebSocket URL with settings
#         const wsBaseUrl = protocol + '//' + window.location.host + '/api/assessment/' + testType;
#         const wsUrl = `${wsBaseUrl}?use_sports2d=${useSports2d}&mode=${mode}&pose_model=${poseModel}&det_frequency=${detFrequency}`;
        
#         // Restart assessment with new settings
#         if (ws) {
#             ws.close();
#         }
#         startAssessmentWithUrl(wsUrl);
#     });
    
#     function startAssessmentWithUrl(wsUrl) {
#         // Connect to WebSocket with settings
#         ws = new WebSocket(wsUrl);
        
#         // Your existing WebSocket code...
#     }
# </script>
#     </script>
# </body>
# </html>
# """

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
            flex-wrap: wrap;
        }
        select, button, input {
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
        .settings-panel {
            background-color: #fff;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 15px;
            margin-bottom: 20px;
        }
        .setting-row {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
            justify-content: space-between;
        }
        .setting-row label {
            margin-right: 10px;
            flex: 1;
        }
        .setting-row select, 
        .setting-row input {
            flex: 2;
        }
        #apply_settings {
            margin-top: 10px;
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
        
        <div class="settings-panel">
            <h3>Advanced Settings</h3>
            <div class="setting-row">
                <label for="use_sports2d">Use Sports2D:</label>
                <input type="checkbox" id="use_sports2d" checked>
            </div>
            <div class="setting-row">
                <label for="mode">Mode:</label>
                <select id="mode">
                    <option value="lightweight">Lightweight (Faster)</option>
                    <option value="balanced" selected>Balanced</option>
                    <option value="performance">Performance (Accurate)</option>
                </select>
            </div>
            <div class="setting-row">
                <label for="pose_model">Pose Model:</label>
                <select id="pose_model">
                    <option value="body_with_feet" selected>Body with Feet</option>
                    <option value="coco_133_wrist">Body, Feet & Hands</option>
                    <option value="coco_133">Body, Feet, Hands & Face</option>
                    <option value="coco_17">Body Only</option>
                </select>
            </div>
            <div class="setting-row">
                <label for="det_frequency">Detection Frequency:</label>
                <input type="number" id="det_frequency" min="1" max="30" value="4">
            </div>
            <button id="apply_settings">Apply Settings</button>
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
                    <button id="useTestVideo">Use Test Video</button>
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
        const useTestVideoButton = document.getElementById('useTestVideo');
        
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
                // Add this right after you get the canvas element
                console.log("Canvas dimensions:", canvas.width, "x", canvas.height);

                // Make sure canvas gets proper dimensions from video
                video.addEventListener('loadedmetadata', () => {
                    console.log("Video dimensions:", video.videoWidth, "x", video.videoHeight);
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    console.log("Canvas resized to:", canvas.width, "x", canvas.height);
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
        
        // Use test video instead of webcam
        useTestVideoButton.addEventListener('click', () => {
            const testType = testTypeSelect.value;
            startAssessmentWithUrl(buildWebSocketUrl(testType, true));
            permissionOverlay.style.display = 'none';
        });
        
        // Stop assessment
        stopButton.addEventListener('click', () => {
            stopAssessment();
        });
        
        // Add event listener to the apply settings button
        document.getElementById('apply_settings').addEventListener('click', function() {
            if (!assessmentRunning) {
                alert('Start an assessment first before applying new settings.');
                return;
            }
            
            const testType = testTypeSelect.value;
            startAssessmentWithUrl(buildWebSocketUrl(testType, false));
        });
        
        function buildWebSocketUrl(testType, useTestVideo) {
            // Gather settings
            const useSports2d = document.getElementById('use_sports2d').checked;
            const mode = document.getElementById('mode').value;
            const poseModel = document.getElementById('pose_model').value;
            const detFrequency = document.getElementById('det_frequency').value;
            
            // Build WebSocket URL with settings
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsBaseUrl = `${protocol}//${window.location.host}/api/assessment/${testType}`;
            return `${wsBaseUrl}?use_sports2d=${useSports2d}&mode=${mode}&pose_model=${poseModel}&det_frequency=${detFrequency}&use_test_video=${useTestVideo}`;
        }
        
        function startAssessmentWithUrl(wsUrl) {
            if (assessmentRunning) {
                stopAssessment();
            }
            
            try {
                ws = new WebSocket(wsUrl);
                
                ws.onopen = () => {
                    statusElement.textContent = 'Status: Connected';
                    statusElement.className = 'status connected';
                    assessmentRunning = true;
                    startButton.disabled = true;
                    stopButton.disabled = false;
                    
                    if (stream) {
                        sendFrames();
                    }
                };
                
                ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    
                    // Debug logging
                    console.log("Received WebSocket message:", 
                                "Has image:", !!data.image, 
                                "Has ROM data:", !!data.rom_data,
                                "Image data length:", data.image ? data.image.length : 0);
                    
                    // Update canvas with processed frame
                    if (data.image) {
                        const image = new Image();
                        
                        // Add load and error handlers
                        image.onload = () => {
                            console.log("Image loaded successfully:", image.width, "x", image.height);
                            ctx.clearRect(0, 0, canvas.width, canvas.height);
                            ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
                        };
                        
                        image.onerror = (error) => {
                            console.error("Error loading image:", error);
                            // Display error on canvas
                            ctx.clearRect(0, 0, canvas.width, canvas.height);
                            ctx.fillStyle = 'red';
                            ctx.font = '16px Arial';
                            ctx.fillText('Failed to load image: ' + (error.message || 'Unknown error'), 20, 50);
                        };
                        
                        image.src = data.image;
                    } else {
                        console.warn("No image data in WebSocket message");
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
                } catch (error) {
                    console.error("Error processing WebSocket message:", error);
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
        
        function startAssessment(testType) {
            startAssessmentWithUrl(buildWebSocketUrl(testType, false));
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
    """WebSocket endpoint for real-time ROM assessment with full configuration support."""
    # Check if test type is valid
    is_custom_test = test_type.startswith("custom_")
    config_manager = ConfigManager()
    
    available_tests = list(config_manager.config.get("test_defaults", {}).keys())
    custom_tests = list(config_manager.config.get("custom_tests", {}).keys())
    
    if test_type not in available_tests and test_type not in custom_tests and test_type != "custom":
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
        
        # Initialize pose detector with configuration
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
                
                # Process frame
                processed_frame, rom_data = test_instance.process_frame(frame)
                # Debug information
                logger.info(f"Frame processed: shape={processed_frame.shape if processed_frame is not None else 'None'}")

                # print("processed_frame",processed_frame)
                # print("rom_data", rom_data)
                # if processed_frame is None or processed_frame.size == 0:
                #     logger.warning("Received empty processed frame from process_frame")
                #     processed_frame = frame.copy()  # Use the original frame as a fallback
                # # Encode processed frame
                # _, buffer = cv2.imencode(".jpg", processed_frame)
                # processed_b64 = base64.b64encode(buffer).decode("utf-8")

                # Encode processed frame
                if processed_frame is not None and processed_frame.size > 0:
                    _, buffer = cv2.imencode(".jpg", processed_frame)
                    processed_b64 = base64.b64encode(buffer).decode("utf-8")
                    logger.info(f"Image encoded, size: {len(processed_b64)} bytes")
                else:
                    logger.error("Processed frame is None or empty")
                    # Use original frame as fallback
                    _, buffer = cv2.imencode(".jpg", frame)
                    processed_b64 = base64.b64encode(buffer).decode("utf-8")
                    logger.info(f"Using original frame as fallback, size: {len(processed_b64)} bytes")
                
                # Prepare response data
                response_data = {
                    "image": f"data:image/jpeg;base64,{processed_b64}",
                    "rom_data": rom_data
                }
                # print("response_data", response_data)
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

# @app.websocket("/api/assessment/{test_type}")
# async def assessment_websocket(websocket: WebSocket, test_type: str):
#     """WebSocket endpoint for real-time ROM assessment with full configuration support."""
#     # Check if test type is valid
#     is_custom_test = test_type.startswith("custom_")
#     config_manager = ConfigManager()
    
#     available_tests = list(config_manager.config.get("test_defaults", {}).keys())
#     custom_tests = list(config_manager.config.get("custom_tests", {}).keys())
    
#     if test_type not in available_tests and test_type not in custom_tests and test_type != "custom":
#         await websocket.close(code=1008, reason=f"Unsupported test type: {test_type}")
#         return
    
#     await websocket.accept()
    
#     # Generate unique connection ID
#     connection_id = f"conn_{int(time.time())}_{id(websocket)}"
#     active_connections[connection_id] = websocket
    
#     # Parse query parameters
#     # Parse query parameters
#     query_params = dict(websocket.query_params)
    
#     # Extract Sports2D configuration options
#     use_sports2d = query_params.get("use_sports2d", "true").lower() == "true"
#     sports2d_mode = query_params.get("mode", "balanced")
#     sports2d_model = query_params.get("pose_model", "body_with_feet")
#     device = query_params.get("device", "auto")
#     backend = query_params.get("backend", "auto")
    
#     # Initialize pose detector with Sports2D options if enabled
#     pose_detector = PoseDetector(
#         model_type=sports2d_model,
#         det_frequency=int(query_params.get("det_frequency", "4")),
#         tracking_mode=query_params.get("tracking_mode", "sports2d"),
#         keypoint_likelihood_threshold=float(query_params.get("likelihood_threshold", "0.3")),
#         mode=sports2d_mode,
#         backend=backend,
#         device=device,
#         use_sports2d=use_sports2d
#     )
    
#     # Extract configuration options from query parameters
#     config_options = {}
    
#     # Pose detection options
#     if "model_type" in query_params:
#         config_options["model_type"] = query_params["model_type"]
#     if "det_frequency" in query_params:
#         config_options["det_frequency"] = int(query_params["det_frequency"])
#     if "tracking_mode" in query_params:
#         config_options["tracking_mode"] = query_params["tracking_mode"]
#     if "likelihood_threshold" in query_params:
#         config_options["keypoint_likelihood_threshold"] = float(query_params["likelihood_threshold"])
    
#     # Assessment options
#     if "ready_time" in query_params:
#         config_options["ready_time_required"] = int(query_params["ready_time"])
    
#     # Visualization options
#     if "theme" in query_params:
#         config_options["theme"] = query_params["theme"]
#     if "display_mode" in query_params:
#         config_options["display_mode"] = query_params["display_mode"]
#     if "show_trajectory" in query_params:
#         config_options["show_trajectory"] = query_params["show_trajectory"].lower() == "true"
#     if "font_size" in query_params:
#         config_options["font_size"] = float(query_params["font_size"])
    
#     # Processing options
#     if "filter_type" in query_params:
#         config_options["filter_type"] = query_params["filter_type"]
#     if "interpolate" in query_params:
#         config_options["interpolate"] = query_params["interpolate"].lower() == "true"
    
#     # Custom body parts and angles
#     body_parts = []
#     if "body_parts" in query_params:
#         body_parts = query_params["body_parts"].split(",")
#         config_options["body_parts"] = body_parts
    
#     joint_angles = []
#     if "joint_angles" in query_params:
#         # Format: "angle_name:point1,point2,point3;angle_name2:point1,point2,point3"
#         angle_defs = query_params["joint_angles"].split(";")
#         for angle_def in angle_defs:
#             if ":" in angle_def:
#                 name, points_str = angle_def.split(":")
#                 points = points_str.split(",")
#                 if len(points) == 3:
#                     joint_angles.append({
#                         "name": name,
#                         "points": points,
#                         "type": "joint"
#                     })
        
#         if joint_angles:
#             config_options["joint_angles"] = joint_angles
    
#     segment_angles = []
#     if "segment_angles" in query_params:
#         # Format: "angle_name:point1,point2,reference;angle_name2:point1,point2,reference"
#         angle_defs = query_params["segment_angles"].split(";")
#         for angle_def in angle_defs:
#             if ":" in angle_def:
#                 parts = angle_def.split(":")
#                 if len(parts) >= 2:
#                     name = parts[0]
#                     points_str = parts[1]
#                     points = points_str.split(",")
                    
#                     if len(points) >= 2:
#                         segment_def = {
#                             "name": name,
#                             "points": points[:2],
#                             "type": "segment"
#                         }
                        
#                         if len(points) > 2:
#                             segment_def["reference"] = points[2]
                        
#                         segment_angles.append(segment_def)
        
#         if segment_angles:
#             config_options["segment_angles"] = segment_angles
    
#     # Primary angle
#     if "primary_angle" in query_params:
#         config_options["primary_angle"] = query_params["primary_angle"]
#     elif joint_angles:
#         config_options["primary_angle"] = joint_angles[0]["name"]
    
#     # Initialize HTTP client for LLM API
#     async_client = httpx.AsyncClient(timeout=10.0)
    
#     try:
#         logger.info(f"Started assessment session: {test_type} ({connection_id})")
        
#         # Initialize visualizer with theme from config
#         theme = config_options.get("theme", "dark")
#         from rom.utils.visualization import EnhancedVisualizer
#         visualizer = EnhancedVisualizer(theme=theme)
        
#         # Initialize pose detector with configuration
#         pose_detector = PoseDetector(
#             model_type=config_options.get("model_type", "HALPE_26"),
#             det_frequency=config_options.get("det_frequency", 4),
#             tracking_mode=config_options.get("tracking_mode", "sports2d"),
#             keypoint_likelihood_threshold=config_options.get("keypoint_likelihood_threshold", 0.3),
#             average_likelihood_threshold=config_options.get("average_likelihood_threshold", 0.5),
#             keypoint_number_threshold=config_options.get("keypoint_number_threshold", 0.3)
#         )
        
#         # Create appropriate test instance
#         if test_type == "custom" or body_parts or joint_angles or segment_angles:
#             # Create a fully configurable test
#             from rom.core.configurable_test import ConfigurableROMTest
            
#             test_instance = ConfigurableROMTest(
#                 pose_detector=pose_detector,
#                 visualizer=visualizer,
#                 config=config_options,
#                 test_type="custom",
#                 config_manager=config_manager
#             )
#         elif is_custom_test:
#             # Load a saved custom test
#             custom_config = config_manager.get_test_config(test_type)
#             custom_config.update(config_options)
            
#             from rom.core.configurable_test import ConfigurableROMTest
            
#             test_instance = ConfigurableROMTest(
#                 pose_detector=pose_detector,
#                 visualizer=visualizer,
#                 config=custom_config,
#                 test_type=test_type,
#                 config_manager=config_manager
#             )
#         else:
#             # Use standard test from the factories
#             test_config = config_manager.get_test_config(test_type)
#             test_config.update(config_options)
            
#             if test_type not in test_factories:
#                 # Fall back to configurable test for unknown test types
#                 from rom.core.configurable_test import ConfigurableROMTest
                
#                 test_instance = ConfigurableROMTest(
#                     pose_detector=pose_detector,
#                     visualizer=visualizer,
#                     config=test_config,
#                     test_type=test_type,
#                     config_manager=config_manager
#                 )
#             else:
#                 # Use registered test factory
#                 test_instance = test_factories[test_type](pose_detector, visualizer, test_config)
        
#         # Process frames from WebSocket
#         while True:
#             try:
#                 # Receive frame from client
#                 data = await websocket.receive_text()
                
#                 # Decode base64 image
#                 if not data.startswith("data:image"):
#                     continue
                
#                 image_data = base64.b64decode(data.split(",")[1])
#                 nparr = np.frombuffer(image_data, np.uint8)
#                 frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
#                 if frame is None or frame.size == 0:
#                     continue
                
#                 # Process frame
#                 processed_frame, rom_data = test_instance.process_frame(frame)
                
#                 # Encode processed frame
#                 _, buffer = cv2.imencode(".jpg", processed_frame)
#                 processed_b64 = base64.b64encode(buffer).decode("utf-8")
                
#                 # Prepare response data
#                 response_data = {
#                     "image": f"data:image/jpeg;base64,{processed_b64}",
#                     "rom_data": rom_data
#                 }
                
#                 # If assessment is completed, send data to LLM API for analysis
#                 if rom_data.get("status") == "completed" and "rom" in rom_data:
#                     try:
#                         # Create comprehensive assessment data
#                         assessment_data = {
#                             "test_type": test_type,
#                             "timestamp": datetime.now().isoformat(),
#                             "rom_data": rom_data,
#                             "config": {
#                                 k: v for k, v in config_options.items() 
#                                 if k not in ["joint_angles", "segment_angles", "body_parts"]
#                             }
#                         }
                        
#                         # Generate analysis if analyzer is available
#                         try:
#                             from rom.analysis.assessment_analyzer import AssessmentAnalyzer
#                             analyzer = AssessmentAnalyzer()
                            
#                             # Get angle history from test instance
#                             angle_history = {}
#                             if hasattr(test_instance, "data_processor"):
#                                 for angle_name in test_instance.data_processor.angle_data:
#                                     angle_history[angle_name] = list(test_instance.data_processor.angle_data[angle_name].angle_history)
                            
#                             analysis_result = analyzer.analyze_assessment(
#                                 rom_data,
#                                 angle_history
#                             )
                            
#                             response_data["analysis"] = analysis_result
#                         except ImportError:
#                             logger.warning("AssessmentAnalyzer not available, skipping analysis")
                        
#                         # Send to LLM API for additional insights
#                         try:
#                             llm_response = await async_client.post(
#                                 LLM_API_URL,
#                                 json=assessment_data
#                             )
                            
#                             if llm_response.status_code == 200:
#                                 response_data["llm_analysis"] = llm_response.json()
#                             else:
#                                 logger.warning(f"LLM API returned status code {llm_response.status_code}")
#                                 response_data["llm_analysis"] = {
#                                     "error": "Failed to get analysis from LLM API",
#                                     "status_code": llm_response.status_code
#                                 }
#                         except Exception as e:
#                             logger.error(f"Error contacting LLM API: {str(e)}")
#                             response_data["llm_analysis"] = {
#                                 "error": "Failed to get analysis from LLM API",
#                                 "message": str(e)
#                             }
                        
#                         # Generate visualization report if configured
#                         if visualizer and hasattr(visualizer, "create_report_image"):
#                             report_img = visualizer.create_report_image(
#                                 rom_data,
#                                 angle_history if 'angle_history' in locals() else {},
#                                 include_plots=True
#                             )
                            
#                             if report_img is not None:
#                                 _, report_buffer = cv2.imencode(".jpg", report_img)
#                                 report_b64 = base64.b64encode(report_buffer).decode("utf-8")
#                                 response_data["report_image"] = f"data:image/jpeg;base64,{report_b64}"
                    
#                     except Exception as analysis_error:
#                         logger.error(f"Error generating analysis: {str(analysis_error)}")
#                         response_data["analysis_error"] = str(analysis_error)
                
#                 # Send response
#                 await websocket.send_text(json.dumps(response_data))
            
#             except WebSocketDisconnect:
#                 logger.info(f"Client disconnected: {connection_id}")
#                 break
            
#             except Exception as e:
#                 logger.error(f"Error processing frame: {str(e)}")
#                 try:
#                     await websocket.send_text(json.dumps({
#                         "error": str(e),
#                         "message": "Error processing frame"
#                     }))
#                 except:
#                     break
    
#     except Exception as e:
#         logger.error(f"Unexpected error in WebSocket handler: {str(e)}")
    
#     finally:
#         # Clean up
#         if connection_id in active_connections:
#             del active_connections[connection_id]
        
#         try:
#             await async_client.aclose()
#         except:
#             pass
        
#         try:
#             await websocket.close()
#         except:
#             pass
        
#         logger.info(f"Closed assessment session: {test_type} ({connection_id})")
# # @app.websocket("/api/assessment/{test_type}")
# async def assessment_websocket(websocket: WebSocket, test_type: str):
#     """WebSocket endpoint for real-time ROM assessment with enhanced visualization."""
#     if test_type not in test_factories:
#         await websocket.close(code=1008, reason=f"Unsupported test type: {test_type}")
#         return
    
#     await websocket.accept()
    
#     # Generate unique connection ID
#     connection_id = f"conn_{int(time.time())}_{id(websocket)}"
#     active_connections[connection_id] = websocket
    
#     # Initialize components
#     pose_detector = PoseDetector()
#     visualizer = EnhancedVisualizer()
    
#     # Get query parameters
#     query_params = dict(websocket.query_params)
#     body_parts = query_params.get("body_parts", "").split(",") if "body_parts" in query_params else []
#     primary_angle = query_params.get("primary_angle")
    
#     # Create appropriate test instance based on parameters
#     if body_parts:
#         # Create custom test with specified body parts
#         from rom.core.custom_test_factory import CustomTestFactory
        
#         # Create custom joint angle definitions if needed
#         joint_angles = []
#         if "angle_points" in query_params:
#             angle_points = query_params["angle_points"].split(";")
#             for i, points in enumerate(angle_points):
#                 point_names = points.split(",")
#                 if len(point_names) == 3:
#                     joint_angles.append({
#                         "name": f"custom_angle_{i}",
#                         "points": point_names
#                     })
        
#         test_instance = CustomTestFactory.create_test(
#             body_parts,
#             joint_angles=joint_angles,
#             primary_angle=primary_angle or (joint_angles[0]["name"] if joint_angles else None),
#             pose_detector=pose_detector,
#             visualizer=visualizer
#         )
#     else:
#         # Create standard test
#         test_instance = test_factories[test_type](pose_detector, visualizer, {})
    
#     # ... (rest of the WebSocket handler implementation remains the same) TODO
    
#     # Create appropriate test instance
#     # test_instance = test_factories[test_type](pose_detector, visualizer, {})
    
#     # Initialize HTTP client for LLM API
#     async_client = httpx.AsyncClient(timeout=10.0)
    
#     try:
#         logger.info(f"Started assessment session: {test_type} ({connection_id})")
        
#         while True:
#             try:
#                 # Receive frame from client
#                 data = await websocket.receive_text()
                
#                 # Decode base64 image
#                 if not data.startswith("data:image"):
#                     continue
                
#                 image_data = base64.b64decode(data.split(",")[1])
#                 nparr = np.frombuffer(image_data, np.uint8)
#                 frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
#                 if frame is None or frame.size == 0:
#                     continue
                
#                 # Process frame with appropriate test
#                 processed_frame, rom_data = test_instance.process_frame(frame)
                
#                 # Encode processed frame
#                 _, buffer = cv2.imencode(".jpg", processed_frame)
#                 processed_b64 = base64.b64encode(buffer).decode("utf-8")
                
#                 # Prepare response
#                 response_data = {
#                     "image": f"data:image/jpeg;base64,{processed_b64}",
#                     "rom_data": rom_data
#                 }
                
#                 # If assessment is completed, send data to LLM API
#                 if rom_data.get("status") == "completed" and "rom" in rom_data:
#                     try:
#                         llm_response = await async_client.post(
#                             LLM_API_URL,
#                             json={
#                                 "assessment_type": test_type,
#                                 "rom_data": rom_data,
#                                 "timestamp": datetime.now().isoformat()
#                             }
#                         )
                        
#                         if llm_response.status_code == 200:
#                             response_data["llm_analysis"] = llm_response.json()
#                         else:
#                             logger.warning(f"LLM API returned status code {llm_response.status_code}")
#                             response_data["llm_analysis"] = {
#                                 "error": "Failed to get analysis from LLM API",
#                                 "status_code": llm_response.status_code
#                             }
#                     except Exception as e:
#                         logger.error(f"Error contacting LLM API: {str(e)}")
#                         response_data["llm_analysis"] = {
#                             "error": "Failed to get analysis from LLM API",
#                             "message": str(e)
#                         }
                
#                 # Send response
#                 await websocket.send_text(json.dumps(response_data))
                
#             except WebSocketDisconnect:
#                 logger.info(f"Client disconnected: {connection_id}")
#                 break
                
#             except Exception as e:
#                 logger.error(f"Error processing frame: {str(e)}")
#                 try:
#                     await websocket.send_text(json.dumps({
#                         "error": str(e),
#                         "message": "Error processing frame"
#                     }))
#                 except:
#                     break
    
#     except Exception as e:
#         logger.error(f"Unexpected error in WebSocket handler: {str(e)}")
    
#     finally:
#         # Clean up
#         if connection_id in active_connections:
#             del active_connections[connection_id]
        
#         try:
#             await async_client.aclose()
#         except:
#             pass
            
#         try:
#             await websocket.close()
#         except:
#             pass
            
#         logger.info(f"Closed assessment session: {test_type} ({connection_id})")


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "version": "1.0.0",
        "active_connections": len(active_connections)
    }

@app.post("/api/custom_angles")
async def create_custom_angle_definition(
    body_parts: List[str] = Body(..., description="List of body part names to track"),
    joint_angles: Optional[List[Dict[str, Any]]] = Body(None, description="Joint angle definitions"),
    segment_angles: Optional[List[Dict[str, Any]]] = Body(None, description="Segment angle definitions")
):
    """
    Create a custom angle definition for assessment.
    
    This endpoint allows defining custom body parts and angles for ROM assessment.
    """
    # Validate body parts
    valid_parts = set(PoseDetector().keypoint_mapping.keys())
    for part in body_parts:
        if part not in valid_parts:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid body part: {part}. Valid parts are: {', '.join(sorted(valid_parts))}"
            )
    
    # Validate joint angles
    if joint_angles:
        for angle in joint_angles:
            if "name" not in angle or "points" not in angle:
                raise HTTPException(
                    status_code=400,
                    detail="Joint angles must have 'name' and 'points' fields"
                )
            
            if len(angle["points"]) != 3:
                raise HTTPException(
                    status_code=400,
                    detail=f"Joint angle '{angle['name']}' must have exactly 3 points"
                )
            
            for point in angle["points"]:
                if point not in valid_parts:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid point '{point}' in angle '{angle['name']}'"
                    )
    
    # Validate segment angles
    if segment_angles:
        for angle in segment_angles:
            if "name" not in angle or "points" not in angle:
                raise HTTPException(
                    status_code=400,
                    detail="Segment angles must have 'name' and 'points' fields"
                )
            
            if len(angle["points"]) != 2:
                raise HTTPException(
                    status_code=400,
                    detail=f"Segment angle '{angle['name']}' must have exactly 2 points"
                )
            
            for point in angle["points"]:
                if point not in valid_parts:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid point '{point}' in angle '{angle['name']}'"
                    )
    
    # Generate a unique ID for this angle definition
    definition_id = f"angle_def_{int(time.time())}"
    
    # In a production system, you would store this in a database
    # For now, we'll just return the ID
    
    return {
        "status": "success",
        "definition_id": definition_id,
        "body_parts": body_parts,
        "joint_angles": joint_angles or [],
        "segment_angles": segment_angles or [],
        "websocket_url": f"/api/assessment/custom?definition_id={definition_id}"
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
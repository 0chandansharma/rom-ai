# rom_streamlit_demo.py
import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import mediapipe as mp
from PIL import Image
import io
import base64
import os
from collections import deque
import math

# Set page configuration
st.set_page_config(
    page_title="ROM Assessment Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    color: #3366ff !important;
    margin-bottom: 1rem !important;
}
.sub-header {
    font-size: 1.5rem !important;
    font-weight: 600 !important;
    color: #555555 !important;
    margin-bottom: 1rem !important;
}
.metric-container {
    background-color: #f0f2f6;
    border-radius: 10px;
    padding: 15px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin-bottom: 1rem;
}
.metric-label {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: #555555 !important;
}
.metric-value {
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    color: #3366ff !important;
}
.info-box {
    background-color: #e6f3ff;
    border-left: 5px solid #3366ff;
    padding: 15px;
    border-radius: 5px;
    margin-bottom: 1rem;
}
.results-container {
    background-color: #f9f9f9;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.status-ready {
    color: #28a745;
    font-weight: 600;
}
.status-measuring {
    color: #ffc107;
    font-weight: 600;
}
.status-completed {
    color: #17a2b8;
    font-weight: 600;
}
.gauge-container {
    width: 100%;
    height: 30px;
    background-color: #e9ecef;
    border-radius: 5px;
    overflow: hidden;
    margin-bottom: 10px;
}
.gauge-fill {
    height: 100%;
    background-color: #3366ff;
    border-radius: 5px;
    transition: width 0.5s ease;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'angle_history' not in st.session_state:
    st.session_state.angle_history = deque(maxlen=100)
    st.session_state.min_angle = float('inf')
    st.session_state.max_angle = float('-inf')
    st.session_state.rom = 0
    st.session_state.assessment_status = "not_started"  # not_started, preparing, in_progress, completed
    st.session_state.ready_time = 0
    st.session_state.required_ready_time = 20
    st.session_state.is_position_valid = False
    st.session_state.guidance_message = "Waiting for camera input..."
    st.session_state.current_test_type = "lower_back_flexion"
    st.session_state.results_df = pd.DataFrame(columns=["Timestamp", "Test Type", "ROM", "Min Angle", "Max Angle"])
    st.session_state.start_time = None
    st.session_state.chart_data = pd.DataFrame({"time": [], "angle": []})
    st.session_state.current_angle = None

# MediaPipe Pose utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Setup pose detector
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Utility functions
def calculate_angle(a, b, c):
    """Calculate angle between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Handle numerical errors
    
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def check_position(landmarks, frame_shape):
    """Check if person is in correct starting position for assessment."""
    h, w, _ = frame_shape
    messages = []
    is_valid = True
    
    # List of required landmarks for assessment
    required_landmarks = [
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE,
        mp_pose.PoseLandmark.RIGHT_KNEE,
        mp_pose.PoseLandmark.LEFT_ANKLE,
        mp_pose.PoseLandmark.RIGHT_ANKLE
    ]
    
    # Check if all required landmarks are detected
    for landmark in required_landmarks:
        if landmarks.landmark[landmark].visibility < 0.7:
            return False, "Cannot detect full body. Please step back."
    
    # Get coordinates of key landmarks
    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    left_ankle = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
    
    # Check if person is facing the camera (using shoulder width)
    shoulder_width = abs(left_shoulder.x - right_shoulder.x) * w
    if shoulder_width < w * 0.15:
        messages.append("Turn to face the camera")
        is_valid = False
    
    # Check if person is too close or too far
    body_height = abs(left_shoulder.y - left_ankle.y) * h
    if body_height < h * 0.5:
        messages.append("Step closer to the camera")
        is_valid = False
    elif body_height > h * 0.9:
        messages.append("Step back from the camera")
        is_valid = False
    
    # Check if person is centered
    center_x = (left_shoulder.x + right_shoulder.x) / 2
    if center_x < 0.3:
        messages.append("Move right")
        is_valid = False
    elif center_x > 0.7:
        messages.append("Move left")
        is_valid = False
    
    # Check if person is standing straight
    hip_shoulder_angle = calculate_angle(
        [left_shoulder.x, left_shoulder.y],
        [(left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2],
        [right_shoulder.x, right_shoulder.y]
    )
    
    if not (85 <= hip_shoulder_angle <= 95):
        messages.append("Stand straight")
        is_valid = False
    
    # Combine messages
    if messages:
        guidance_message = " | ".join(messages)
    else:
        guidance_message = "Good starting position. Hold still."
    
    return is_valid, guidance_message

def process_frame(frame, test_type):
    """Process a single frame for ROM assessment."""
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    
    if not results.pose_landmarks:
        return frame, {
        "status": "failed",
        "guidance_message": "No pose detected. Please make sure your full body is visible.",
        "current_angle": None,  # Add this line
        "min_angle": None,      # Add this line
        "max_angle": None,      # Add this line
        "rom": 0,               # Add this line
        "is_position_valid": False  # Add this line
    }
    
    # Draw pose landmarks
    annotated_frame = frame.copy()
    mp_drawing.draw_landmarks(
        annotated_frame,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
    )
    
    # Check position
    is_valid_position, guidance_message = check_position(results.pose_landmarks, frame.shape)
    
    # Update assessment status
    if st.session_state.assessment_status == "not_started" and is_valid_position:
        st.session_state.assessment_status = "preparing"
    
    # Handle preparation phase
    if st.session_state.assessment_status == "preparing":
        if is_valid_position:
            st.session_state.ready_time += 1
            if st.session_state.ready_time >= st.session_state.required_ready_time:
                st.session_state.assessment_status = "in_progress"
                st.session_state.start_time = time.time()
                guidance_message = "Begin bending forward slowly"
        else:
            st.session_state.ready_time = 0
    
    # Calculate angles based on test type
    landmarks = results.pose_landmarks.landmark
    trunk_angle = None
    
    if test_type == "lower_back_flexion":
        # Get relevant landmarks
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w,
                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * w,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * h]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * w,
                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * h]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x * w,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y * h]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x * w,
                     landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y * h]
        
        # Calculate midpoints
        shoulder_midpoint = [(left_shoulder[0] + right_shoulder[0]) / 2,
                            (left_shoulder[1] + right_shoulder[1]) / 2]
        hip_midpoint = [(left_hip[0] + right_hip[0]) / 2,
                       (left_hip[1] + right_hip[1]) / 2]
        knee_midpoint = [(left_knee[0] + right_knee[0]) / 2,
                        (left_knee[1] + right_knee[1]) / 2]
        
        # Calculate trunk angle
        trunk_angle = calculate_angle(shoulder_midpoint, hip_midpoint, knee_midpoint)
        st.session_state.current_angle = trunk_angle
        
        # Add angle to history
        if st.session_state.assessment_status == "in_progress":
            st.session_state.angle_history.append(trunk_angle)
            
            # Update min and max angles
            st.session_state.min_angle = min(st.session_state.angle_history)
            st.session_state.max_angle = max(st.session_state.angle_history)
            st.session_state.rom = st.session_state.max_angle - st.session_state.min_angle
            
            # Add to chart data
            current_time = time.time() - st.session_state.start_time if st.session_state.start_time else 0
            new_row = pd.DataFrame({"time": [current_time], "angle": [trunk_angle]})
            st.session_state.chart_data = pd.concat([st.session_state.chart_data, new_row], ignore_index=True)
            
            # Check if max flexion is reached (ROM stabilized)
            if len(st.session_state.angle_history) > 30:
                recent_angles = list(st.session_state.angle_history)[-10:]
                if abs(max(recent_angles) - min(recent_angles)) < 3 and current_time > 5:
                    st.session_state.assessment_status = "completed"
                    
                    # Save results
                    new_result = pd.DataFrame({
                        "Timestamp": [time.strftime("%Y-%m-%d %H:%M:%S")],
                        "Test Type": [test_type],
                        "ROM": [st.session_state.rom],
                        "Min Angle": [st.session_state.min_angle],
                        "Max Angle": [st.session_state.max_angle]
                    })
                    st.session_state.results_df = pd.concat([st.session_state.results_df, new_result], ignore_index=True)
                    
                    guidance_message = "Assessment completed"
        
        # Draw angle visualization
        cv2.line(annotated_frame, 
                (int(shoulder_midpoint[0]), int(shoulder_midpoint[1])), 
                (int(hip_midpoint[0]), int(hip_midpoint[1])), 
                (255, 0, 0), 4)
        cv2.line(annotated_frame, 
                (int(hip_midpoint[0]), int(hip_midpoint[1])), 
                (int(knee_midpoint[0]), int(knee_midpoint[1])), 
                (255, 0, 0), 4)
        
        # Add angle text
        cv2.putText(annotated_frame, 
                   f"{trunk_angle:.1f}°", 
                   (int(hip_midpoint[0]) - 50, int(hip_midpoint[1]) - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Draw guidance overlay
    overlay = annotated_frame.copy()
    cv2.rectangle(overlay, (0, h-100), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, annotated_frame, 0.5, 0, annotated_frame)
    
    # Add guidance text
    cv2.putText(annotated_frame, 
               f"Guidance: {guidance_message}", 
               (20, h-60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Draw progress bar for preparation phase
    if st.session_state.assessment_status == "preparing" and is_valid_position:
        progress = (st.session_state.ready_time / st.session_state.required_ready_time) * (w - 40)
        cv2.rectangle(annotated_frame, (20, h-30), (w-20, h-10), (255, 255, 255), 2)
        cv2.rectangle(annotated_frame, (20, h-30), (int(20 + progress), h-10), (0, 255, 0), -1)
    
    # Prepare result data
    result_data = {
        "status": st.session_state.assessment_status,
        "is_position_valid": is_valid_position,
        "guidance_message": guidance_message,
        "current_angle": st.session_state.current_angle,
        "min_angle": st.session_state.min_angle if st.session_state.min_angle != float('inf') else None,
        "max_angle": st.session_state.max_angle if st.session_state.max_angle != float('-inf') else None,
        "rom": st.session_state.rom,
    }
    
    return annotated_frame, result_data

# Sidebar
st.sidebar.markdown("<h1 class='main-header'>ROM Assessment</h1>", unsafe_allow_html=True)

# Test type selection
test_type = st.sidebar.selectbox(
    "Select Assessment Type",
    [
        "Lower Back Flexion",
        "Lower Back Extension",
        "Lower Back Lateral Flexion (Left)",
        "Lower Back Lateral Flexion (Right)",
        "Lower Back Rotation (Left)",
        "Lower Back Rotation (Right)"
    ],
    index=0
)

# Convert display name to internal code
test_type_map = {
    "Lower Back Flexion": "lower_back_flexion",
    "Lower Back Extension": "lower_back_extension",
    "Lower Back Lateral Flexion (Left)": "lower_back_lateral_flexion_left",
    "Lower Back Lateral Flexion (Right)": "lower_back_lateral_flexion_right",
    "Lower Back Rotation (Left)": "lower_back_rotation_left",
    "Lower Back Rotation (Right)": "lower_back_rotation_right"
}
current_test_type = test_type_map[test_type]

# If test type changed, reset assessment
if current_test_type != st.session_state.current_test_type:
    st.session_state.angle_history.clear()
    st.session_state.min_angle = float('inf')
    st.session_state.max_angle = float('-inf')
    st.session_state.rom = 0
    st.session_state.assessment_status = "not_started"
    st.session_state.ready_time = 0
    st.session_state.guidance_message = "Waiting for camera input..."
    st.session_state.current_test_type = current_test_type
    st.session_state.chart_data = pd.DataFrame({"time": [], "angle": []})
    st.session_state.current_angle = None

# Control buttons
col1, col2 = st.sidebar.columns(2)
start_button = col1.button("Start Assessment")
reset_button = col2.button("Reset")

if start_button:
    st.session_state.assessment_status = "not_started"
    st.session_state.angle_history.clear()
    st.session_state.min_angle = float('inf')
    st.session_state.max_angle = float('-inf')
    st.session_state.rom = 0
    st.session_state.ready_time = 0
    st.session_state.chart_data = pd.DataFrame({"time": [], "angle": []})
    st.session_state.current_angle = None

if reset_button:
    st.session_state.angle_history.clear()
    st.session_state.min_angle = float('inf')
    st.session_state.max_angle = float('-inf')
    st.session_state.rom = 0
    st.session_state.assessment_status = "not_started"
    st.session_state.ready_time = 0
    st.session_state.guidance_message = "Waiting for camera input..."
    st.session_state.chart_data = pd.DataFrame({"time": [], "angle": []})
    st.session_state.current_angle = None

# Assessment information
st.sidebar.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.markdown("<h3 class='sub-header'>Assessment Information</h3>", unsafe_allow_html=True)

test_descriptions = {
    "lower_back_flexion": "This test measures the forward bending ability of your lower back. Stand straight facing the camera, then slowly bend forward as far as comfortable.",
    "lower_back_extension": "This test measures the backward bending ability of your lower back. Stand straight facing the camera, then slowly bend backward as far as comfortable.",
    "lower_back_lateral_flexion_left": "This test measures your ability to bend sideways to the left. Stand straight facing the camera, then slowly bend to your left side.",
    "lower_back_lateral_flexion_right": "This test measures your ability to bend sideways to the right. Stand straight facing the camera, then slowly bend to your right side.",
    "lower_back_rotation_left": "This test measures your ability to rotate your trunk to the left. Stand straight facing the camera, then slowly rotate your upper body to the left.",
    "lower_back_rotation_right": "This test measures your ability to rotate your trunk to the right. Stand straight facing the camera, then slowly rotate your upper body to the right."
}

normal_rom_values = {
    "lower_back_flexion": 60,
    "lower_back_extension": 25,
    "lower_back_lateral_flexion_left": 25,
    "lower_back_lateral_flexion_right": 25,
    "lower_back_rotation_left": 45,
    "lower_back_rotation_right": 45
}

st.sidebar.markdown(f"""
<div class='info-box'>
{test_descriptions[current_test_type]}
<br><br>
<b>Normal ROM:</b> {normal_rom_values[current_test_type]}°
</div>
""", unsafe_allow_html=True)

# Show previous results
if not st.session_state.results_df.empty:
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    st.sidebar.markdown("<h3 class='sub-header'>Previous Results</h3>", unsafe_allow_html=True)
    st.sidebar.dataframe(st.session_state.results_df, hide_index=True)

# Main content
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("<h2 class='sub-header'>ROM Assessment Camera View</h2>", unsafe_allow_html=True)
    
    # Placeholder for video stream
    video_placeholder = st.empty()
    
    # Status indicator
    status_col1, status_col2 = st.columns(2)
    status_placeholder = status_col1.empty()
    guidance_placeholder = status_col2.empty()

with col2:
    st.markdown("<h2 class='sub-header'>Assessment Results</h2>", unsafe_allow_html=True)
    
    # Current angle
    current_angle_placeholder = st.empty()
    
    # ROM measurement
    rom_placeholder = st.empty()
    
    # Min/Max angles
    min_max_col1, min_max_col2 = st.columns(2)
    min_angle_placeholder = min_max_col1.empty()
    max_angle_placeholder = min_max_col2.empty()
    
    # Chart
    st.markdown("<h3 class='sub-header'>Angle Trajectory</h3>", unsafe_allow_html=True)
    chart_placeholder = st.empty()
    
    # Analysis
    st.markdown("<h3 class='sub-header'>Analysis</h3>", unsafe_allow_html=True)
    analysis_placeholder = st.empty()

# Start camera
cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video. Please check your camera connection.")
            continue
            # break
        
        # Process frame
        processed_frame, result_data = process_frame(frame, current_test_type)
        
        # Update video placeholder
        video_placeholder.image(processed_frame, channels="BGR", use_column_width=True)
        
        # Update status
        status_text = result_data["status"].replace("_", " ").title()
        if result_data["status"] == "not_started":
            status_placeholder.markdown(f"<div>Status: <span class='status-ready'>{status_text}</span></div>", unsafe_allow_html=True)
        elif result_data["status"] in ["preparing", "in_progress"]:
            status_placeholder.markdown(f"<div>Status: <span class='status-measuring'>{status_text}</span></div>", unsafe_allow_html=True)
        else:
            status_placeholder.markdown(f"<div>Status: <span class='status-completed'>{status_text}</span></div>", unsafe_allow_html=True)
        
        # Update guidance
        guidance_placeholder.markdown(f"<div>{result_data['guidance_message']}</div>", unsafe_allow_html=True)
        
        # Update current angle
        if result_data["current_angle"] is not None:
            current_angle_placeholder.markdown(
                f"<div class='metric-container'><div class='metric-label'>Current Angle</div><div class='metric-value'>{result_data['current_angle']:.1f}°</div></div>",
                unsafe_allow_html=True
            )
        else:
            current_angle_placeholder.markdown(
                f"<div class='metric-container'><div class='metric-label'>Current Angle</div><div class='metric-value'>--</div></div>",
                unsafe_allow_html=True
            )
        
        # Update ROM
        if result_data["rom"] > 0:
            rom_placeholder.markdown(
                f"<div class='metric-container'><div class='metric-label'>Range of Motion</div><div class='metric-value'>{result_data['rom']:.1f}°</div></div>",
                unsafe_allow_html=True
            )
            
            # Add progress bar for ROM compared to normal values
            normal_rom = normal_rom_values[current_test_type]
            rom_percentage = min(100, (result_data["rom"] / normal_rom) * 100)
            rom_placeholder.markdown(
                f"""
                <div class='gauge-container'>
                    <div class='gauge-fill' style='width: {rom_percentage}%;'></div>
                </div>
                <div style='display: flex; justify-content: space-between;'>
                    <div>0°</div>
                    <div>{normal_rom}° (Normal)</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            rom_placeholder.markdown(
                f"<div class='metric-container'><div class='metric-label'>Range of Motion</div><div class='metric-value'>--</div></div>",
                unsafe_allow_html=True
            )
        
        # Update min/max angles
        if result_data["min_angle"] is not None:
            min_angle_placeholder.markdown(
                f"<div class='metric-container'><div class='metric-label'>Min Angle</div><div class='metric-value'>{result_data['min_angle']:.1f}°</div></div>",
                unsafe_allow_html=True
            )
        else:
            min_angle_placeholder.markdown(
                f"<div class='metric-container'><div class='metric-label'>Min Angle</div><div class='metric-value'>--</div></div>",
                unsafe_allow_html=True
            )
            
        if result_data["max_angle"] is not None:
            max_angle_placeholder.markdown(
                f"<div class='metric-container'><div class='metric-label'>Max Angle</div><div class='metric-value'>{result_data['max_angle']:.1f}°</div></div>",
                unsafe_allow_html=True
            )
        else:
            max_angle_placeholder.markdown(
                f"<div class='metric-container'><div class='metric-label'>Max Angle</div><div class='metric-value'>--</div></div>",
                unsafe_allow_html=True
            )
        
        # Update chart
        if not st.session_state.chart_data.empty:
            chart = plt.figure(figsize=(10, 4))
            plt.plot(st.session_state.chart_data["time"], st.session_state.chart_data["angle"], 'b-')
            plt.xlabel("Time (seconds)")
            plt.ylabel("Angle (degrees)")
            plt.title("Angle vs. Time")
            plt.grid(True)
            plt.ylim(0, max(180, max(st.session_state.chart_data["angle"]) + 10) if len(st.session_state.chart_data["angle"]) > 0 else 180)
            chart_placeholder.pyplot(chart)
        
        # Update analysis
        if result_data["status"] == "completed":
            rom = result_data["rom"]
            normal_rom = normal_rom_values[current_test_type]
            
            if rom >= normal_rom:
                assessment = f"Your {test_type} is excellent at {rom:.1f}°, which meets or exceeds the normal range of {normal_rom}°."
                recommendations = [
                    "Continue with your current exercise routine",
                    "Focus on maintaining this excellent range of motion",
                    "Consider adding strength training to complement your flexibility"
                ]
            elif rom >= normal_rom * 0.75:
                assessment = f"Your {test_type} is good at {rom:.1f}°, which is {normal_rom - rom:.1f}° below the normal range of {normal_rom}°."
                recommendations = [
                    "Incorporate gentle stretching exercises daily",
                    "Consider yoga or pilates for improved flexibility",
                    "Maintain proper posture throughout the day"
                ]
            elif rom >= normal_rom * 0.5:
                assessment = f"Your {test_type} is limited at {rom:.1f}°, which is {normal_rom - rom:.1f}° below the normal range of {normal_rom}°."
                recommendations = [
                    "Start a dedicated stretching program for your lower back",
                    "Consider consulting with a physical therapist",
                    "Use heat therapy before stretching to improve tissue flexibility",
                    "Focus on core strengthening exercises"
                ]
            else:
                assessment = f"Your {test_type} is significantly limited at {rom:.1f}°, which is {normal_rom - rom:.1f}° below the normal range of {normal_rom}°."
                recommendations = [
                    "Consult with a healthcare provider or physical therapist",
                    "Begin with very gentle assisted stretching",
                    "Consider pain management techniques if needed",
                    "Gradually increase activity as tolerated"
                ]
            
            analysis_placeholder.markdown(
                f"""
                <div class='results-container'>
                    <h4>Assessment:</h4>
                    <p>{assessment}</p>
                    
                    <h4>Recommendations:</h4>
                    <ul>
                    {"".join([f"<li>{rec}</li>" for rec in recommendations])}
                    </ul>
                    
                    <h4>Comparison to Average:</h4>
                    <p>Your ROM is {((rom/normal_rom)*100):.1f}% of the normal range for this movement.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            analysis_placeholder.markdown(
                """
                <div class='results-container'>
                    <p>Complete the assessment to view analysis and recommendations.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Check for stop condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
finally:
    cap.release()

# Instructions for running the demo
st.markdown("""
## How to Run This Demo

1. Install the required packages:
```bash
pip install streamlit opencv-python mediapipe numpy pandas matplotlib
```

2. Save this script as `rom_streamlit_demo.py`

3. Run the Streamlit app:
```bash
streamlit run rom_streamlit_demo.py
```

4. Follow the on-screen instructions to perform the assessment
""")

# Explanation of the implementation
st.markdown("""
## Implementation Details

This demo showcases the core functionality of the ROM Assessment Library:

1. **Real-time pose estimation** using MediaPipe
2. **Position guidance** to ensure proper assessment form
3. **Angle calculation and tracking** for accurate ROM measurement
4. **Visualization** of assessment progress
5. **Automated guidance** throughout the assessment process
6. **Analysis and recommendations** based on assessment results

The full ROM library includes additional features such as:
- Multiple assessment types for various body parts
- WebSocket API for integration with other platforms
- LLM-powered personalized analysis
- Data persistence and session management
- Enhanced visualization and guidance
""")
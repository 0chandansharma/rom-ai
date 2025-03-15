# rom/demo/streamlit_demo.py
import streamlit as st
import cv2
import numpy as np
import base64
import time
import json
import requests
from datetime import datetime
from PIL import Image
import io

# Import ROM modules
from rom.utils.pose_detector import PoseDetector
from rom.utils.visualization import EnhancedVisualizer
from rom.tests.lower_back_test import (
    EnhancedLowerBackFlexionTest, 
    EnhancedLowerBackExtensionTest,
    EnhancedLowerBackLateralFlexionTest,
    EnhancedLowerBackRotationTest
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

# Configure page
st.set_page_config(
    page_title="ROM Assessment Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add custom CSS
st.markdown("""
<style>
.main {
    background-color: #f5f5f5;
}
.stApp {
    max-width: 1200px;
    margin: 0 auto;
}
.assessment-header {
    background-color: #333;
    color: white;
    padding: 20px;
    border-radius: 8px;
    margin-bottom: 20px;
}
.video-container {
    background-color: #fff;
    border-radius: 8px;
    padding: 15px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}
.data-container {
    background-color: #fff;
    border-radius: 8px;
    padding: 15px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# App title
st.markdown('<div class="assessment-header"><h1>ROM Assessment Demo</h1></div>', unsafe_allow_html=True)

# Sidebar controls
st.sidebar.title("Assessment Controls")

# Assessment type selection
test_type = st.sidebar.selectbox(
    "Select Assessment Type",
    list(test_factories.keys()),
    format_func=lambda x: x.replace("_", " ").title()
)

# Advanced settings
st.sidebar.subheader("Advanced Settings")

# Sports2D settings
use_sports2d = st.sidebar.checkbox("Use Sports2D", value=True)
sports2d_mode = st.sidebar.selectbox(
    "Mode",
    ["lightweight", "balanced", "performance"],
    index=1
)
pose_model = st.sidebar.selectbox(
    "Pose Model",
    ["body_with_feet", "coco_133_wrist", "coco_133", "coco_17"],
    index=0
)
det_frequency = st.sidebar.slider("Detection Frequency", 1, 30, 4)

# Input source selection
input_source = st.sidebar.radio(
    "Input Source",
    ["Webcam", "Upload Video", "Test Video"]
)

# Main content area
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown('<div class="video-container">', unsafe_allow_html=True)
    st.subheader("Video Feed")
    
    # Placeholder for video display
    video_placeholder = st.empty()
    
    # Start/Stop buttons
    start_button = st.button("Start Assessment")
    stop_button = st.button("Stop Assessment")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="data-container">', unsafe_allow_html=True)
    st.subheader("Assessment Data")
    
    # Placeholder for assessment data
    data_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

# Session state initialization
if 'running' not in st.session_state:
    st.session_state.running = False
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'rom_data' not in st.session_state:
    st.session_state.rom_data = {"status": "not_started"}
if 'pose_detector' not in st.session_state:
    st.session_state.pose_detector = None
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = None
if 'test_instance' not in st.session_state:
    st.session_state.test_instance = None
if 'video_capture' not in st.session_state:
    st.session_state.video_capture = None

# Handle button clicks
if start_button:
    st.session_state.running = True
    st.session_state.frame_count = 0
    st.session_state.start_time = time.time()
    st.session_state.rom_data = {"status": "initializing"}
    
    # Initialize components
    st.session_state.visualizer = EnhancedVisualizer(theme="dark")
    st.session_state.pose_detector = PoseDetector(
        model_type=pose_model,
        det_frequency=det_frequency,
        mode=sports2d_mode,
        use_sports2d=use_sports2d
    )
    
    # Create test instance
    st.session_state.test_instance = test_factories[test_type](
        st.session_state.pose_detector,
        st.session_state.visualizer,
        {}
    )
    
    # Setup video capture
    if input_source == "Webcam":
        st.session_state.video_capture = cv2.VideoCapture(0)
    elif input_source == "Upload Video":
        uploaded_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            temp_file = "temp_video.mp4"
            with open(temp_file, "wb") as f:
                f.write(uploaded_file.read())
            st.session_state.video_capture = cv2.VideoCapture(temp_file)
    else:  # Test Video
        st.session_state.rom_data = {"status": "using_test_video"}
        # Create a simple test pattern
        height, width = 720, 1280
        test_frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw a gradient background
        for i in range(height):
            color = int(255 * i / height)
            test_frame[i, :] = [color, color, color]
        
        # Add text
        cv2.putText(test_frame, "ROM Assessment Test Video", 
                   (int(width/2) - 200, int(height/2)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add a simple moving object to test tracking
        st.session_state.test_frame = test_frame
        st.session_state.test_circle_pos = [width//2, height//2]
        st.session_state.test_circle_dir = [5, 5]

if stop_button:
    st.session_state.running = False
    if st.session_state.video_capture is not None:
        st.session_state.video_capture.release()
    st.session_state.video_capture = None

# Main processing loop
if st.session_state.running:
    if input_source == "Test Video":
        # Generate a test frame with a moving circle
        frame = st.session_state.test_frame.copy()
        
        # Update circle position
        pos = st.session_state.test_circle_pos
        direction = st.session_state.test_circle_dir
        
        pos[0] += direction[0]
        pos[1] += direction[1]
        
        # Bounce off edges
        if pos[0] < 50 or pos[0] > frame.shape[1] - 50:
            direction[0] *= -1
        if pos[1] < 50 or pos[1] > frame.shape[0] - 50:
            direction[1] *= -1
        
        # Draw circle
        cv2.circle(frame, (pos[0], pos[1]), 30, (0, 0, 255), -1)
        
        # Save updated position
        st.session_state.test_circle_pos = pos
        st.session_state.test_circle_dir = direction
    else:
        # Read from video capture
        if st.session_state.video_capture is None or not st.session_state.video_capture.isOpened():
            st.error("Could not access video source")
            st.session_state.running = False
        else:
            ret, frame = st.session_state.video_capture.read()
            if not ret:
                st.error("Could not read frame from video source")
                st.session_state.running = False
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                
    
    if st.session_state.running:
        # Process frame with ROM assessment
        try:
            processed_frame, rom_data = st.session_state.test_instance.process_frame(frame)
            # print("processed_frame, rom_data", processed_frame, rom_data)
            # Update session state with latest ROM data
            st.session_state.rom_data = rom_data
            
            # Convert OpenCV frame to Streamlit-compatible format
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(processed_frame_rgb, channels="RGB", use_column_width=True)
            # print("processed_frame_rgb", processed_frame_rgb)
            # Update data display
            data_placeholder.json(rom_data)
            # print("data_pla", data_placeholder)
            # Check if assessment is completed
            if rom_data.get("status") == "completed":
                st.success("Assessment completed successfully!")
                st.session_state.running = False
                
                # Show summary
                st.subheader("Assessment Summary")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("ROM Angle", f"{rom_data.get('rom', 0):.1f}°")
                with col2:
                    duration = time.time() - st.session_state.start_time
                    st.metric("Duration", f"{duration:.1f} seconds")
                
                # Display recommended actions based on ROM results
                if "rom" in rom_data:
                    rom_value = rom_data["rom"]
                    if test_type == "lower_back_flexion":
                        if rom_value < 40:
                            st.warning("Limited lower back flexion detected. Consider exercises to increase mobility.")
                        elif rom_value > 60:
                            st.info("Good lower back flexion range.")
                        else:
                            st.success("Normal lower back flexion range.")
                    elif test_type == "lower_back_extension":
                        if rom_value < 15:
                            st.warning("Limited lower back extension detected. Consider exercises to increase mobility.")
                        elif rom_value > 30:
                            st.info("Good lower back extension range.")
                        else:
                            st.success("Normal lower back extension range.")
            else:
                    # This is critical - rerun to get the next frame
                    time.sleep(0.1)  # Small delay to control frame rate
                    st.rerun()
        except Exception as e:
            st.error(f"Error processing frame: {str(e)}")
            st.session_state.running = True

# Display instructions when not running
if not st.session_state.running:
    # Show instructions based on selected test
    instructions = {
        "lower_back_flexion": """
        """
    }
    
    st.markdown(instructions.get(test_type, "Select a test type and click 'Start Assessment'"))
    
    # Display sample images for proper positioning
    st.subheader("Sample Positioning")
    
    # Create a simple visualization of proper positioning
    height, width = 240, 320
    sample_image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Draw stick figure in appropriate position for selected test
    if test_type in ["lower_back_flexion", "lower_back_extension"]:
        # Side view stick figure
        # Head
        cv2.circle(sample_image, (width//2, height//4), 20, (0, 0, 0), 2)
        # Body
        cv2.line(sample_image, (width//2, height//4 + 20), (width//2, height//2 + 40), (0, 0, 0), 2)
        # Arms
        cv2.line(sample_image, (width//2, height//3), (width//2 - 30, height//2), (0, 0, 0), 2)
        cv2.line(sample_image, (width//2, height//3), (width//2 + 30, height//2), (0, 0, 0), 2)
        # Legs
        cv2.line(sample_image, (width//2, height//2 + 40), (width//2 - 20, height - 20), (0, 0, 0), 2)
        cv2.line(sample_image, (width//2, height//2 + 40), (width//2 + 20, height - 20), (0, 0, 0), 2)
        
        # Add text
        cv2.putText(sample_image, "Stand sideways to camera", (30, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    else:
        # Front view stick figure
        # Head
        cv2.circle(sample_image, (width//2, height//4), 20, (0, 0, 0), 2)
        # Body
        cv2.line(sample_image, (width//2, height//4 + 20), (width//2, height//2 + 40), (0, 0, 0), 2)
        # Arms
        cv2.line(sample_image, (width//2, height//3), (width//2 - 40, height//2), (0, 0, 0), 2)
        cv2.line(sample_image, (width//2, height//3), (width//2 + 40, height//2), (0, 0, 0), 2)
        # Legs
        cv2.line(sample_image, (width//2, height//2 + 40), (width//2 - 30, height - 20), (0, 0, 0), 2)
        cv2.line(sample_image, (width//2, height//2 + 40), (width//2 + 30, height - 20), (0, 0, 0), 2)
        
        # Add text
        cv2.putText(sample_image, "Stand facing camera", (50, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Display sample image
    st.image(cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB), width=320)

# Footer
st.markdown("---")
st.markdown("ROM Assessment System Demo | &copy; 2025")

# Add some useful links
with st.expander("About ROM Assessment"):
    st.write("""
    Range of Motion 
    """)
    
    st.subheader("Normal ROM Ranges")
    st.markdown("""

    """)
#!/usr/bin/env python
"""
Simple test script for ROM assessment code.
"""

import cv2
import numpy as np
import time
import os
import sys

# Add these to help with PyTorch issues
os.environ["PYTORCH_JIT"] = "0"
os.environ["TORCH_USE_RTLD_GLOBAL"] = "YES"

# Import ROM modules
try:
    from rom.utils.pose_detector import PoseDetector
    from rom.utils.visualization import EnhancedVisualizer
    from rom.tests.lower_back_test import (
        EnhancedLowerBackFlexionTest
    )
except ImportError as e:
    print(f"Error importing ROM modules: {e}")
    print("Make sure you're running this script from the correct directory.")
    sys.exit(1)

def test_with_webcam():
    """Test ROM assessment with webcam input."""
    print("Testing lower_back_flexion with webcam input")
    
    # Initialize components
    try:
        print("Initializing visualizer...")
        visualizer = EnhancedVisualizer(theme="dark")
        
        print("Initializing pose detector...")
        pose_detector = PoseDetector(
            model_type="body_with_feet",
            det_frequency=4,
            mode="balanced",
            use_sports2d=True
        )
        
        print("Creating test instance...")
        test_instance = EnhancedLowerBackFlexionTest(
            pose_detector, visualizer, {}
        )
        
        print("Opening webcam...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
        
        frame_count = 0
        start_time = time.time()
        
        print("Press 'q' to quit")
        
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from webcam.")
                break
            
            # Process frame
            try:
                print(f"Processing frame {frame_count}...")
                processed_frame, rom_data = test_instance.process_frame(frame)
                frame_count += 1
                
                # Display ROM data on console
                print(f"ROM data: {rom_data}")
                
                # Display frame
                if processed_frame is not None:
                    cv2.imshow("ROM Assessment", processed_frame)
                else:
                    print("Warning: Processed frame is None, displaying original")
                    cv2.imshow("ROM Assessment", frame)
                
                # Calculate FPS
                if frame_count % 10 == 0:
                    elapsed_time = time.time() - start_time
                    fps = frame_count / elapsed_time
                    print(f"FPS: {fps:.2f}")
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting...")
                    break
            
            except Exception as e:
                print(f"Error processing frame: {e}")
                # Continue with next frame
                continue
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error during test: {e}")
    
    print("Test completed.")

if __name__ == "__main__":
    test_with_webcam()
#!/usr/bin/env python
"""
Simple test script for ROM assessment code.
This script tests the core functionality without Streamlit or FastAPI.
"""

import cv2
import numpy as np
import time
import os
import sys
from datetime import datetime

# Add these environment variables to help with PyTorch issues
# os.environ["PYTORCH_JIT"] = "0"  # Disable JIT compilation
# os.environ["TORCH_USE_RTLD_GLOBAL"] = "YES"  # Make PyTorch symbols globally available

# Import ROM modules
try:
    from rom.utils.pose_detector import PoseDetector
    from rom.utils.visualization import EnhancedVisualizer
    from rom.tests.lower_back_test import (
        EnhancedLowerBackFlexionTest, 
        EnhancedLowerBackExtensionTest,
        EnhancedLowerBackLateralFlexionTest,
        EnhancedLowerBackRotationTest
    )
except ImportError as e:
    print(f"Error importing ROM modules: {e}")
    print("Make sure you're running this script from the correct directory.")
    sys.exit(1)

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

def test_with_webcam(test_type="lower_back_flexion", use_sports2d=True):
    """Test ROM assessment with webcam input."""
    print(f"Testing {test_type} with webcam input")
    
    # Initialize components
    try:
        print("Initializing visualizer...")
        visualizer = EnhancedVisualizer(theme="dark")
        
        print("Initializing pose detector...")
        pose_detector = PoseDetector(
            model_type="body_with_feet",
            det_frequency=4,
            mode="balanced",
            use_sports2d=use_sports2d
        )
        
        print("Creating test instance...")
        test_instance = test_factories[test_type](
            pose_detector, visualizer, {}
        )
        
        print("Opening webcam...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
        
        frame_count = 0
        start_time = time.time()
        
        print("Press 'q' to quit, 's' to save a screenshot")
        
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
                elif key == ord('s'):
                    # Save screenshot
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"rom_screenshot_{timestamp}.jpg"
                    cv2.imwrite(filename, processed_frame)
                    print(f"Screenshot saved as {filename}")
            
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

def test_with_image(image_path, test_type="lower_back_flexion", use_sports2d=True):
    """Test ROM assessment with a single image."""
    print(f"Testing {test_type} with image: {image_path}")
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found.")
        return
    
    # Initialize components
    try:
        print("Initializing visualizer...")
        visualizer = EnhancedVisualizer(theme="dark")
        
        print("Initializing pose detector...")
        pose_detector = PoseDetector(
            model_type="body_with_feet",
            det_frequency=4,
            mode="balanced",
            use_sports2d=use_sports2d
        )
        
        print("Creating test instance...")
        test_instance = test_factories[test_type](
            pose_detector, visualizer, {}
        )
        
        # Read image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not read image {image_path}")
            return
        
        # Process image
        print("Processing image...")
        start_time = time.time()
        
        try:
            processed_frame, rom_data = test_instance.process_frame(frame)
            
            # Display ROM data
            print("ROM Data:")
            for key, value in rom_data.items():
                print(f"  {key}: {value}")
            
            # Display processing time
            processing_time = time.time() - start_time
            print(f"Processing time: {processing_time:.2f} seconds")
            
            # Display image
            if processed_frame is not None:
                cv2.imshow("ROM Assessment", processed_frame)
                
                # Save processed image
                output_path = f"{os.path.splitext(image_path)[0]}_processed.jpg"
                cv2.imwrite(output_path, processed_frame)
                print(f"Processed image saved as {output_path}")
            else:
                print("Warning: Processed frame is None")
                cv2.imshow("ROM Assessment", frame)
            
            print("Press any key to close the window")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"Error processing image: {e}")
        
    except Exception as e:
        print(f"Error during test: {e}")
    
    print("Test completed.")

def test_with_test_pattern(test_type="lower_back_flexion", use_sports2d=True):
    """Test ROM assessment with a generated test pattern."""
    print(f"Testing {test_type} with test pattern")
    
    # Initialize components
    try:
        print("Initializing visualizer...")
        visualizer = EnhancedVisualizer(theme="dark")
        
        print("Initializing pose detector...")
        pose_detector = PoseDetector(
            model_type="body_with_feet",
            det_frequency=4,
            mode="balanced",
            use_sports2d=use_sports2d
        )
        
        print("Creating test instance...")
        test_instance = test_factories[test_type](
            pose_detector, visualizer, {}
        )
        
        # Create test pattern
        height, width = 720, 1280
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw a gradient background
        for i in range(height):
            color = int(255 * i / height)
            frame[i, :] = [color, color, color]
        
        # Add text
        cv2.putText(frame, "ROM Assessment Test Pattern", 
                   (width//2 - 200, height//2 - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw a simple stick figure
        # Head
        cv2.circle(frame, (width//2, height//2 - 100), 30, (0, 0, 255), -1)
        # Body
        cv2.line(frame, (width//2, height//2 - 70), (width//2, height//2 + 50), (0, 0, 255), 5)
        # Arms
        cv2.line(frame, (width//2, height//2 - 30), (width//2 - 80, height//2), (0, 0, 255), 5)
        cv2.line(frame, (width//2, height//2 - 30), (width//2 + 80, height//2), (0, 0, 255), 5)
        # Legs
        cv2.line(frame, (width//2, height//2 + 50), (width//2 - 50, height//2 + 150), (0, 0, 255), 5)
        cv2.line(frame, (width//2, height//2 + 50), (width//2 + 50, height//2 + 150), (0, 0, 255), 5)
        
        # Process frame
        print("Processing test pattern...")
        start_time = time.time()
        
        try:
            processed_frame, rom_data = test_instance.process_frame(frame)
            
            # Display ROM data
            print("ROM Data:")
            for key, value in rom_data.items():
                print(f"  {key}: {value}")
            
            # Display processing time
            processing_time = time.time() - start_time
            print(f"Processing time: {processing_time:.2f} seconds")
            
            # Display frame
            if processed_frame is not None:
                cv2.imshow("ROM Assessment", processed_frame)
            else:
                print("Warning: Processed frame is None, displaying original")
                cv2.imshow("ROM Assessment", frame)
            
            print("Press any key to close the window")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"Error processing test pattern: {e}")
        
    except Exception as e:
        print(f"Error during test: {e}")
    
    print("Test completed.")

def main():
    """Main function to run the test script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test ROM assessment code")
    parser.add_argument("--mode", type=str, choices=["webcam", "image", "pattern"], 
                        default="webcam", help="Test mode")
    parser.add_argument("--test-type", type=str, 
                        choices=list(test_factories.keys()), 
                        default="lower_back_flexion", 
                        help="ROM test type")
    parser.add_argument("--image", type=str, help="Path to image file (for image mode)")
    parser.add_argument("--no-sports2d", action="store_true", 
                        help="Disable Sports2D (use fallback pose detection)")
    
    args = parser.parse_args()
    
    # Run the appropriate test
    if args.mode == "webcam":
        test_with_webcam(args.test_type, not args.no_sports2d)
    elif args.mode == "image":
        if not args.image:
            print("Error: Image path is required for image mode.")
            return
        test_with_image(args.image, args.test_type, not args.no_sports2d)
    else:  # pattern
        test_with_test_pattern(args.test_type, not args.no_sports2d)

if __name__ == "__main__":
    main()
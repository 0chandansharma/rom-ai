# rom/tests/lower_back_test.py
import time
import cv2
import numpy as np
from collections import deque
from typing import Dict, Tuple, Any, List, Optional

from rom.core.base import ROMTest, ROMData, JointType, MovementPlane, AssessmentStatus, Point3D, Angle, MathUtils


class LowerBackTestType:
    """Types of lower back tests."""
    FLEXION = "flexion"  # Forward bending
    EXTENSION = "extension"  # Backward bending
    LATERAL_FLEXION_LEFT = "lateral_flexion_left"  # Side bending left
    LATERAL_FLEXION_RIGHT = "lateral_flexion_right"  # Side bending right
    ROTATION_LEFT = "rotation_left"  # Rotating left
    ROTATION_RIGHT = "rotation_right"  # Rotating right


class LowerBackFlexionTest(ROMTest):
    """Test for lower back flexion (bending forward)."""

    def __init__(self, pose_detector, visualizer, config=None):
        """
        Initialize the lower back flexion test.
        
        Args:
            pose_detector: Pose detection system
            visualizer: Visualization system
            config: Configuration parameters
        """
        config = config or {}
        # Default configuration values
        self.default_config = {
            "ready_time_required": 20,  # Frames to consider position ready
            "angle_buffer_size": 100,  # Size of the angle history buffer
            "position_tolerance": 10,  # Degrees of tolerance for position checks
            "smoothing_window": 5,  # Window size for angle smoothing
            "key_landmarks": [
                11,  # Left shoulder
                12,  # Right shoulder
                23,  # Left hip
                24,  # Right hip
                25,  # Left knee
                26,  # Right knee
                27,  # Left ankle
                28   # Right ankle
            ]
        }
        
        # Merge provided config with defaults
        self.config = {**self.default_config, **(config or {})}
        
        super().__init__(pose_detector, visualizer, self.config)
        
        # Additional instance variables
        self.ready_time = 0
        self.angle_buffer = deque(maxlen=self.config["angle_buffer_size"])
        self.is_ready = False
        self.start_time = None
        self.end_time = None

    def _initialize_rom_data(self) -> ROMData:
        """Initialize ROM data specific to lower back flexion test."""
        return ROMData(
            test_type=LowerBackTestType.FLEXION,
            joint_type=JointType.LOWER_BACK,
            status=AssessmentStatus.NOT_STARTED,
            min_angle=float('inf'),
            max_angle=float('-inf')
        )

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process a single frame for the lower back flexion test.
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (processed_frame, rom_data_dict)
        """
        # Get frame dimensions
        h, w, _ = frame.shape
        frame_shape = (h, w, 3)
        
        # Find pose landmarks
        landmarks = self.pose_detector.find_pose(frame)
        
        if not landmarks:
            self.data.guidance_message = "No pose detected. Please make sure your full body is visible."
            self.data.status = AssessmentStatus.FAILED
            # Create a simple frame with message
            self.visualizer.put_text(frame, self.data.guidance_message, (20, 50), color='red')
            return frame, self.data.to_dict()
        
        # Convert landmarks to dictionary of coordinates
        landmark_coords = self.pose_detector.get_landmark_coordinates(frame, landmarks)
        
        # Check if required landmarks are detected
        if not all(key in landmark_coords for key in self.config["key_landmarks"]):
            self.data.guidance_message = "Cannot detect all required body parts. Please step back to show your full body."
            self.data.status = AssessmentStatus.FAILED
            self.visualizer.put_text(frame, self.data.guidance_message, (20, 50), color='red')
            return frame, self.data.to_dict()
        
        # Check initial position
        is_valid_position, guidance_message = self.check_position(landmark_coords, frame_shape)
        
        # Update assessment status
        if self.data.status == AssessmentStatus.NOT_STARTED and is_valid_position:
            self.data.status = AssessmentStatus.PREPARING
        
        # Handle preparation phase
        if self.data.status == AssessmentStatus.PREPARING:
            if is_valid_position:
                self.ready_time += 1
                if self.ready_time >= self.config["ready_time_required"]:
                    self.is_ready = True
                    self.data.status = AssessmentStatus.IN_PROGRESS
                    self.start_time = time.time()
                    self.data.guidance_message = "Begin bending forward slowly"
            else:
                self.ready_time = 0
                self.is_ready = False
                self.data.guidance_message = guidance_message
        
        # Calculate angles
        angles = self.calculate_angles(landmark_coords)
        
        # Store lower back angle in ROM data
        if "trunk_angle" in angles:
            trunk_angle = angles["trunk_angle"]
            self.data.current_angle = trunk_angle
            
            # Only record angles if assessment is in progress
            if self.data.status == AssessmentStatus.IN_PROGRESS:
                self.data.history.append(trunk_angle)
                self.angle_buffer.append(trunk_angle.value)
                
                # Update ROM min/max
                self.data.update_rom()
                
                # Check if max flexion is reached
                if len(self.data.history) > 30:  # After reasonable number of frames
                    recent_angles = [a.value for a in self.data.history[-10:]]
                    if abs(max(recent_angles) - min(recent_angles)) < 3:  # If angle stabilized
                        self.data.status = AssessmentStatus.COMPLETED
                        self.end_time = time.time()
                        self.data.metadata["duration"] = self.end_time - self.start_time
                        self.data.guidance_message = "Assessment completed"
        
        # Store landmark positions in ROM data
        self.data.landmarks = {
            "left_shoulder": Point3D.from_tuple(landmark_coords[11]),
            "right_shoulder": Point3D.from_tuple(landmark_coords[12]),
            "left_hip": Point3D.from_tuple(landmark_coords[23]),
            "right_hip": Point3D.from_tuple(landmark_coords[24]),
            "left_knee": Point3D.from_tuple(landmark_coords[25]),
            "right_knee": Point3D.from_tuple(landmark_coords[26])
        }
        
        # Visualize assessment
        processed_frame = self.visualize_assessment(frame, self.data)
        
        return processed_frame, self.data.to_dict()

    def calculate_angles(self, landmarks: Dict[int, Tuple[float, float, float]]) -> Dict[str, Angle]:
        """
        Calculate angles from landmarks.
        
        Args:
            landmarks: Dictionary of landmark coordinates
            
        Returns:
            Dictionary of calculated angles
        """
        angles = {}
        
        # Calculate trunk angle (between shoulders, hips, and knees)
        if all(key in landmarks for key in [11, 12, 23, 24, 25, 26]):
            # Use midpoints for more stability
            shoulder_midpoint = (
                (landmarks[11][0] + landmarks[12][0]) / 2,
                (landmarks[11][1] + landmarks[12][1]) / 2,
                (landmarks[11][2] + landmarks[12][2]) / 2
            )
            hip_midpoint = (
                (landmarks[23][0] + landmarks[24][0]) / 2,
                (landmarks[23][1] + landmarks[24][1]) / 2,
                (landmarks[23][2] + landmarks[24][2]) / 2
            )
            knee_midpoint = (
                (landmarks[25][0] + landmarks[26][0]) / 2,
                (landmarks[25][1] + landmarks[26][1]) / 2,
                (landmarks[25][2] + landmarks[26][2]) / 2
            )
            
            trunk_angle_value = MathUtils.calculate_angle(shoulder_midpoint, hip_midpoint, knee_midpoint)
            
            # Create Angle object
            angles["trunk_angle"] = Angle(
                value=trunk_angle_value,
                joint_type=JointType.LOWER_BACK,
                plane=MovementPlane.SAGITTAL,
                timestamp=time.time()
            )
            
            # Calculate hip angle (hip flexion)
            hip_angle_value = MathUtils.calculate_angle(shoulder_midpoint, hip_midpoint, knee_midpoint)
            angles["hip_angle"] = Angle(
                value=hip_angle_value,
                joint_type=JointType.HIP,
                plane=MovementPlane.SAGITTAL,
                timestamp=time.time()
            )
        
        return angles

    def check_position(self, landmarks: Dict[int, Tuple[float, float, float]], frame_shape: Tuple[int, int, int]) -> Tuple[bool, str]:
        """
        Check if the person is in the correct position for the test.
        
        Args:
            landmarks: Dictionary of landmark coordinates
            frame_shape: Shape of the frame (height, width, channels)
            
        Returns:
            Tuple of (is_position_valid, guidance_message)
        """
        h, w, _ = frame_shape
        messages = []
        is_valid = True
        
        # Check if all required landmarks are detected
        for point in self.config["key_landmarks"]:
            if point not in landmarks:
                return False, "Cannot detect full body. Please step back."
        
        # Check if person is facing the camera (using shoulder width)
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
        if shoulder_width < w * 0.15:
            messages.append("Turn to face the camera")
            is_valid = False
        
        # Check if person is too close or too far
        body_height = abs(landmarks[11][1] - landmarks[27][1])  # Shoulder to ankle height
        if body_height < h * 0.5:
            messages.append("Step closer to the camera")
            is_valid = False
        elif body_height > h * 0.9:
            messages.append("Step back from the camera")
            is_valid = False
        
        # Check if person is centered
        center_x = (left_shoulder[0] + right_shoulder[0]) / 2
        if center_x < w * 0.3:
            messages.append("Move right")
            is_valid = False
        elif center_x > w * 0.7:
            messages.append("Move left")
            is_valid = False
        
        # Check if person is standing straight
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        hip_shoulder_angle = MathUtils.calculate_angle(
            left_shoulder,
            ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2, 0),
            right_shoulder
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

    def visualize_assessment(self, frame: np.ndarray, rom_data: ROMData) -> np.ndarray:
        """
        Visualize the assessment on the frame.
        
        Args:
            frame: Input video frame
            rom_data: Current ROM data
            
        Returns:
            Frame with visualization
        """
        h, w, _ = frame.shape
        
        # Draw key landmarks if available
        if rom_data.landmarks:
            if "left_shoulder" in rom_data.landmarks and "right_shoulder" in rom_data.landmarks:
                self.visualizer.draw_landmark_point(frame, 
                                                  int(rom_data.landmarks["left_shoulder"].x), 
                                                  int(rom_data.landmarks["left_shoulder"].y), 
                                                  'blue')
                self.visualizer.draw_landmark_point(frame, 
                                                  int(rom_data.landmarks["right_shoulder"].x), 
                                                  int(rom_data.landmarks["right_shoulder"].y), 
                                                  'blue')
            
            if "left_hip" in rom_data.landmarks and "right_hip" in rom_data.landmarks:
                self.visualizer.draw_landmark_point(frame, 
                                                  int(rom_data.landmarks["left_hip"].x), 
                                                  int(rom_data.landmarks["left_hip"].y), 
                                                  'green')
                self.visualizer.draw_landmark_point(frame, 
                                                  int(rom_data.landmarks["right_hip"].x), 
                                                  int(rom_data.landmarks["right_hip"].y), 
                                                  'green')
                
            if "left_knee" in rom_data.landmarks and "right_knee" in rom_data.landmarks:
                self.visualizer.draw_landmark_point(frame, 
                                                  int(rom_data.landmarks["left_knee"].x), 
                                                  int(rom_data.landmarks["left_knee"].y), 
                                                  'red')
                self.visualizer.draw_landmark_point(frame, 
                                                  int(rom_data.landmarks["right_knee"].x), 
                                                  int(rom_data.landmarks["right_knee"].y), 
                                                  'red')
            
            # Draw connecting lines for trunk
            if all(k in rom_data.landmarks for k in ["left_shoulder", "left_hip", "left_knee"]):
                # Left side
                self.visualizer.draw_connection(
                    frame,
                    (int(rom_data.landmarks["left_shoulder"].x), int(rom_data.landmarks["left_shoulder"].y)),
                    (int(rom_data.landmarks["left_hip"].x), int(rom_data.landmarks["left_hip"].y)),
                    'white'
                )
                self.visualizer.draw_connection(
                    frame,
                    (int(rom_data.landmarks["left_hip"].x), int(rom_data.landmarks["left_hip"].y)),
                    (int(rom_data.landmarks["left_knee"].x), int(rom_data.landmarks["left_knee"].y)),
                    'white'
                )
                
            if all(k in rom_data.landmarks for k in ["right_shoulder", "right_hip", "right_knee"]):
                # Right side
                self.visualizer.draw_connection(
                    frame,
                    (int(rom_data.landmarks["right_shoulder"].x), int(rom_data.landmarks["right_shoulder"].y)),
                    (int(rom_data.landmarks["right_hip"].x), int(rom_data.landmarks["right_hip"].y)),
                    'white'
                )
                self.visualizer.draw_connection(
                    frame,
                    (int(rom_data.landmarks["right_hip"].x), int(rom_data.landmarks["right_hip"].y)),
                    (int(rom_data.landmarks["right_knee"].x), int(rom_data.landmarks["right_knee"].y)),
                    'white'
                )
        
        # Draw current angle if available
        if rom_data.current_angle:
            angle_text = f"Current angle: {rom_data.current_angle.value:.1f}°"
            self.visualizer.put_text(frame, angle_text, (20, 50), 'white')
            
            if rom_data.min_angle is not None and rom_data.max_angle is not None:
                rom_text = f"ROM: {rom_data.min_angle:.1f}° - {rom_data.max_angle:.1f}° = {rom_data.rom:.1f}°"
                self.visualizer.put_text(frame, rom_text, (20, 80), 'white')
        
        # Draw status and guidance
        status_text = f"Status: {rom_data.status.value.replace('_', ' ').title()}"
        self.visualizer.put_text(frame, status_text, (20, 110), 'green')
        
        # Draw guidance overlay at the bottom
        self._draw_guidance_overlay(frame, rom_data)
        
        return frame
    
    def _draw_guidance_overlay(self, frame: np.ndarray, rom_data: ROMData) -> None:
        """
        Draw guidance overlay at the bottom of the frame.
        
        Args:
            frame: Input video frame
            rom_data: Current ROM data
        """
        h, w, _ = frame.shape
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 100), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw guidance text
        self.visualizer.put_text(
            frame,
            f"Guidance: {rom_data.guidance_message}",
            (20, h - 60),
            color='white'
        )
        
        # Draw progress bar if in preparing phase
        if rom_data.status == AssessmentStatus.PREPARING:
            progress = (self.ready_time / self.config["ready_time_required"]) * (w - 40)
            cv2.rectangle(frame, (20, h - 30), (w - 20, h - 20), (255, 255, 255), 2)
            cv2.rectangle(frame, (20, h - 30), (int(20 + progress), h - 20), (0, 255, 0), -1)
        
        # Draw ROM progress if in assessment phase
        if rom_data.status == AssessmentStatus.IN_PROGRESS and rom_data.min_angle is not None and rom_data.max_angle is not None:
            progress_text = f"Bending progress: {rom_data.rom:.1f}°"
            self.visualizer.put_text(frame, progress_text, (20, h - 30), color='white')


class LowerBackExtensionTest(LowerBackFlexionTest):
    """Test for lower back extension (bending backward)."""
    
    def _initialize_rom_data(self) -> ROMData:
        """Initialize ROM data specific to lower back extension test."""
        return ROMData(
            test_type=LowerBackTestType.EXTENSION,
            joint_type=JointType.LOWER_BACK,
            status=AssessmentStatus.NOT_STARTED,
            min_angle=float('inf'),
            max_angle=float('-inf')
        )
        
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process frame for extension test with modified guidance."""
        if self.data.status == AssessmentStatus.IN_PROGRESS:
            self.data.guidance_message = "Slowly bend backward as far as comfortable"
        return super().process_frame(frame)
        

class LowerBackLateralFlexionTest(LowerBackFlexionTest):
    """Test for lower back lateral flexion (side bending)."""
    
    def __init__(self, pose_detector, visualizer, config=None, side="left"):
        """
        Initialize lateral flexion test with specified side.
        
        Args:
            pose_detector: Pose detection system
            visualizer: Visualization system
            config: Configuration parameters
            side: Side to test (left or right)
        """
        self.side = side.lower()
        if self.side not in ["left", "right"]:
            raise ValueError("Side must be 'left' or 'right'")
        super().__init__(pose_detector, visualizer, config)
        
    def _initialize_rom_data(self) -> ROMData:
        """Initialize ROM data specific to lateral flexion test."""
        test_type = (LowerBackTestType.LATERAL_FLEXION_LEFT if self.side == "left" 
                    else LowerBackTestType.LATERAL_FLEXION_RIGHT)
        return ROMData(
            test_type=test_type,
            joint_type=JointType.LOWER_BACK,
            status=AssessmentStatus.NOT_STARTED,
            min_angle=float('inf'),
            max_angle=float('-inf')
        )
        
    def calculate_angles(self, landmarks: Dict[int, Tuple[float, float, float]]) -> Dict[str, Angle]:
        """
        Calculate angles for lateral flexion.
        
        Args:
            landmarks: Dictionary of landmark coordinates
            
        Returns:
            Dictionary of calculated angles
        """
        angles = {}
        
        # Check if all required landmarks are available
        if all(key in landmarks for key in [11, 12, 23, 24]):
            # Calculate angle in frontal plane
            if self.side == "left":
                # For left lateral flexion, use right landmarks as reference
                lateral_angle_value = MathUtils.calculate_angle(
                    landmarks[12],  # Right shoulder
                    landmarks[24],  # Right hip
                    (landmarks[24][0], landmarks[24][1] + 100, landmarks[24][2])  # Point below right hip
                )
            else:
                # For right lateral flexion, use left landmarks as reference
                lateral_angle_value = MathUtils.calculate_angle(
                    landmarks[11],  # Left shoulder
                    landmarks[23],  # Left hip
                    (landmarks[23][0], landmarks[23][1] + 100, landmarks[23][2])  # Point below left hip
                )
            
            # Subtract from 180 to get the inclination angle
            lateral_angle_value = 180 - lateral_angle_value
            
            angles["lateral_angle"] = Angle(
                value=lateral_angle_value,
                joint_type=JointType.LOWER_BACK,
                plane=MovementPlane.FRONTAL,
                timestamp=time.time()
            )
            
        return angles
        
    def check_position(self, landmarks: Dict[int, Tuple[float, float, float]], frame_shape: Tuple[int, int, int]) -> Tuple[bool, str]:
        """
        Check if the person is in the correct position for lateral flexion.
        
        Args:
            landmarks: Dictionary of landmark coordinates
            frame_shape: Shape of the frame
            
        Returns:
            Tuple of (is_position_valid, guidance_message)
        """
        is_valid, generic_message = super().check_position(landmarks, frame_shape)
        
        if not is_valid:
            return False, generic_message
        
        # Additional check: person should be facing sideways
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
        
        if shoulder_width > frame_shape[1] * 0.2:
            return False, f"Turn to face {'right' if self.side == 'left' else 'left'} side"
        
        return True, f"Good position for {self.side} side bend"
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process frame with modified guidance for lateral flexion."""
        if self.data.status == AssessmentStatus.IN_PROGRESS:
            self.data.guidance_message = f"Slowly bend to the {self.side} side"
        return super().process_frame(frame)


class LowerBackRotationTest(LowerBackFlexionTest):
    """Test for lower back rotation (twisting)."""
    
    def __init__(self, pose_detector, visualizer, config=None, side="left"):
        """
        Initialize rotation test with specified side.
        
        Args:
            pose_detector: Pose detection system
            visualizer: Visualization system
            config: Configuration parameters
            side: Side to test (left or right)
        """
        self.side = side.lower()
        if self.side not in ["left", "right"]:
            raise ValueError("Side must be 'left' or 'right'")
        super().__init__(pose_detector, visualizer, config)
        
    def _initialize_rom_data(self) -> ROMData:
        """Initialize ROM data specific to rotation test."""
        test_type = (LowerBackTestType.ROTATION_LEFT if self.side == "left" 
                    else LowerBackTestType.ROTATION_RIGHT)
        return ROMData(
            test_type=test_type,
            joint_type=JointType.LOWER_BACK,
            status=AssessmentStatus.NOT_STARTED,
            min_angle=float('inf'),
            max_angle=float('-inf')
        )
        
    def calculate_angles(self, landmarks: Dict[int, Tuple[float, float, float]]) -> Dict[str, Angle]:
        """
        Calculate angles for rotation.
        
        Args:
            landmarks: Dictionary of landmark coordinates
            
        Returns:
            Dictionary of calculated angles
        """
        angles = {}
        
        # Check if all required landmarks are available
        if all(key in landmarks for key in [11, 12, 23, 24]):
            # Calculate horizontal angle between shoulders and hips
            shoulder_midpoint = (
                (landmarks[11][0] + landmarks[12][0]) / 2,
                (landmarks[11][1] + landmarks[12][1]) / 2,
                (landmarks[11][2] + landmarks[12][2]) / 2
            )
            
            hip_midpoint = (
                (landmarks[23][0] + landmarks[24][0]) / 2,
                (landmarks[23][1] + landmarks[24][1]) / 2,
                (landmarks[23][2] + landmarks[24][2]) / 2
            )
            
            # Project to horizontal plane for rotation measurement
            shoulder_vector = [landmarks[12][0] - landmarks[11][0], 0, landmarks[12][2] - landmarks[11][2]]
            hip_vector = [landmarks[24][0] - landmarks[23][0], 0, landmarks[24][2] - landmarks[23][2]]
            
            # Normalize vectors
            shoulder_norm = np.linalg.norm(shoulder_vector)
            hip_norm = np.linalg.norm(hip_vector)
            
            if shoulder_norm > 0 and hip_norm > 0:
                shoulder_normalized = [x / shoulder_norm for x in shoulder_vector]
                hip_normalized = [x / hip_norm for x in hip_vector]
                
                # Calculate dot product
                dot_product = sum(a * b for a, b in zip(shoulder_normalized, hip_normalized))
                dot_product = max(-1.0, min(1.0, dot_product))  # Clip to avoid numerical errors
                
                # Calculate angle
                rotation_angle = np.degrees(np.arccos(dot_product))
                
                angles["rotation_angle"] = Angle(
                    value=rotation_angle,
                    joint_type=JointType.LOWER_BACK,
                    plane=MovementPlane.TRANSVERSE,
                    timestamp=time.time()
                )
        
        return angles
    
    def check_position(self, landmarks: Dict[int, Tuple[float, float, float]], frame_shape: Tuple[int, int, int]) -> Tuple[bool, str]:
        """
        Check if the person is in the correct position for rotation test.
        
        Args:
            landmarks: Dictionary of landmark coordinates
            frame_shape: Shape of the frame
            
        Returns:
            Tuple of (is_position_valid, guidance_message)
        """
        is_valid, generic_message = super().check_position(landmarks, frame_shape)
        
        if not is_valid:
            return False, generic_message
        
        # For rotation, person should be facing the camera
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
        
        if shoulder_width < frame_shape[1] * 0.15:
            return False, "Turn to face the camera directly"
        
        return True, f"Good position for {self.side} rotation"
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process frame with modified guidance for rotation."""
        if self.data.status == AssessmentStatus.IN_PROGRESS:
            self.data.guidance_message = f"Slowly rotate your upper body to the {self.side}"
        return super().process_frame(frame)
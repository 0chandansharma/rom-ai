# rom/tests/lower_back_test.py
import time
import cv2
import numpy as np
from typing import Dict, Tuple, List, Optional, Any

from rom.core.base import EnhancedROMTest, ROMData, JointType, MovementPlane, AssessmentStatus, Point3D, Angle
from rom.utils.math_utils import MathUtils

class LowerBackTestType:
    """Types of lower back tests."""
    FLEXION = "flexion"  # Forward bending
    EXTENSION = "extension"  # Backward bending
    LATERAL_FLEXION_LEFT = "lateral_flexion_left"  # Side bending left
    LATERAL_FLEXION_RIGHT = "lateral_flexion_right"  # Side bending right
    ROTATION_LEFT = "rotation_left"  # Rotating left
    ROTATION_RIGHT = "rotation_right"  # Rotating right


class EnhancedLowerBackFlexionTest(EnhancedROMTest):
    """Enhanced test for lower back flexion with Sports2D features."""
    
    def _initialize_rom_data(self) -> ROMData:
        """Initialize ROM data specific to lower back flexion test."""
        return ROMData(
            test_type=LowerBackTestType.FLEXION,
            joint_type=JointType.LOWER_BACK,
            status=AssessmentStatus.NOT_STARTED,
            min_angle=float('inf'),
            max_angle=float('-inf')
        )
    
    def _set_default_angles(self):
        """Set default angles to track for lower back flexion."""
        self.joint_angles = [
            "trunk_angle",
            "left_hip_angle",
            "right_hip_angle",
            "left_knee_angle",
            "right_knee_angle"
        ]
        
        self.segment_angles = [
            "trunk_segment",
            "left_thigh_segment",
            "right_thigh_segment"
        ]
    
    def _get_required_landmarks(self) -> List[int]:
        """Get required landmarks for lower back flexion test."""
        return [
            11,  # Left shoulder
            12,  # Right shoulder
            23,  # Left hip
            24,  # Right hip
            25,  # Left knee
            26   # Right knee
        ]
    
    def _get_movement_guidance(self) -> str:
        """Get guidance for lower back flexion."""
        return "Begin bending forward slowly"
    
    def _get_movement_plane(self) -> MovementPlane:
        """Get movement plane for lower back flexion."""
        return MovementPlane.SAGITTAL
    
    def _is_primary_angle(self, angle_name: str) -> bool:
        """Check if angle is primary for lower back flexion."""
        return angle_name == "trunk_angle"
    
    def _update_landmarks(self, landmarks: Dict[int, Tuple[float, float, float]]):
        """Update ROM data with landmark positions."""
        self.data.landmarks = {
            "left_shoulder": Point3D.from_tuple(landmarks[11]),
            "right_shoulder": Point3D.from_tuple(landmarks[12]),
            "left_hip": Point3D.from_tuple(landmarks[23]),
            "right_hip": Point3D.from_tuple(landmarks[24]),
            "left_knee": Point3D.from_tuple(landmarks[25]),
            "right_knee": Point3D.from_tuple(landmarks[26])
        }
    
    def calculate_angles(self, landmarks: Dict[int, Tuple[float, float, float]]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Calculate angles from landmarks.
        
        Args:
            landmarks: Dictionary of landmark coordinates
            
        Returns:
            Tuple of (joint_angles, segment_angles) dictionaries
        """
        joint_angles = {}
        segment_angles = {}
        
        # Calculate trunk angle (between shoulders, hips, and knees)
        if all(idx in landmarks for idx in [11, 12, 23, 24, 25, 26]):
            # Use midpoints for more stability
            shoulder_midpoint = MathUtils.get_midpoint(landmarks[11], landmarks[12])
            hip_midpoint = MathUtils.get_midpoint(landmarks[23], landmarks[24])
            knee_midpoint = MathUtils.get_midpoint(landmarks[25], landmarks[26])
            
            # Calculate trunk angle (joint angle)
            trunk_angle = MathUtils.calculate_angle(shoulder_midpoint, hip_midpoint, knee_midpoint)
            joint_angles["trunk_angle"] = trunk_angle
            
            # Calculate hip angles
            left_hip_angle = MathUtils.calculate_angle(landmarks[11], landmarks[23], landmarks[25])
            right_hip_angle = MathUtils.calculate_angle(landmarks[12], landmarks[24], landmarks[26])
            
            joint_angles["left_hip_angle"] = left_hip_angle
            joint_angles["right_hip_angle"] = right_hip_angle
            
            # Calculate knee angles
            if 27 in landmarks and 28 in landmarks:  # If ankle landmarks are available
                left_knee_angle = MathUtils.calculate_angle(landmarks[23], landmarks[25], landmarks[27])
                right_knee_angle = MathUtils.calculate_angle(landmarks[24], landmarks[26], landmarks[28])
                
                joint_angles["left_knee_angle"] = left_knee_angle
                joint_angles["right_knee_angle"] = right_knee_angle
            
            # Calculate segment angles (with horizontal)
            trunk_segment = MathUtils.calculate_segment_angle(hip_midpoint, shoulder_midpoint)
            left_thigh_segment = MathUtils.calculate_segment_angle(landmarks[23], landmarks[25])
            right_thigh_segment = MathUtils.calculate_segment_angle(landmarks[24], landmarks[26])
            
            segment_angles["trunk_segment"] = trunk_segment
            segment_angles["left_thigh_segment"] = left_thigh_segment
            segment_angles["right_thigh_segment"] = right_thigh_segment
        
        return joint_angles, segment_angles
    
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
        required_landmarks = self._get_required_landmarks()
        if not all(landmark_id in landmarks for landmark_id in required_landmarks):
            return False, "Cannot detect required body parts. Please step back to show your full body."
        
        # Check if person is facing the camera (using shoulder width)
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
        if shoulder_width < w * 0.15:
            messages.append("Turn to face the camera")
            is_valid = False
        
        # Check if person is too close or too far
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        
        # Estimate torso height
        torso_height = 0
        if 11 in landmarks and 23 in landmarks:
            torso_height = abs(landmarks[11][1] - landmarks[23][1])
        
        if torso_height < h * 0.2:
            messages.append("Step closer to the camera")
            is_valid = False
        elif torso_height > h * 0.4:
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
        hip_shoulder_angle = MathUtils.calculate_angle(
            left_shoulder,
            MathUtils.get_midpoint(left_hip, right_hip),
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
        Visualize the assessment on the frame with enhanced visualization.
        
        Args:
            frame: Input video frame
            rom_data: Current ROM data
            
        Returns:
            Frame with visualization
        """
        h, w, _ = frame.shape
        
        # Create a copy of the frame for visualization
        vis_frame = frame.copy()
        
        # Draw only relevant landmarks and connections for this test
        if self.visualizer and rom_data.landmarks:
            # Get landmark coordinates
            landmarks_to_draw = {}
            for name, point in rom_data.landmarks.items():
                landmarks_to_draw[name] = (int(point.x), int(point.y))
            
            # Draw selective connections - only draw what's needed for this test
            relevant_connections = [
                # Draw torso
                ("left_shoulder", "left_hip"),
                ("right_shoulder", "right_hip"),
                ("left_shoulder", "right_shoulder"),
                ("left_hip", "right_hip"),
                # Draw legs
                ("left_hip", "left_knee"),
                ("right_hip", "right_knee")
            ]
            
            # Draw connections
            for start_name, end_name in relevant_connections:
                if start_name in landmarks_to_draw and end_name in landmarks_to_draw:
                    self.visualizer.draw_connection(
                        vis_frame,
                        landmarks_to_draw[start_name],
                        landmarks_to_draw[end_name],
                        color="blue" if "shoulder" in start_name or "shoulder" in end_name else "green"
                    )
            
            # Draw landmarks
            for name, point in landmarks_to_draw.items():
                highlight = "shoulder" in name.lower() or "hip" in name.lower()
                self.visualizer.draw_landmark_point(
                    vis_frame, 
                    point[0], 
                    point[1], 
                    color="yellow" if highlight else "white",
                    size=8 if highlight else 6,
                    highlight=highlight
                )
            
            # Draw trunk angle
            if all(k in landmarks_to_draw for k in ["left_shoulder", "left_hip", "left_knee", 
                                                "right_shoulder", "right_hip", "right_knee"]):
                # Calculate midpoints
                shoulder_midpoint = (
                    (landmarks_to_draw["left_shoulder"][0] + landmarks_to_draw["right_shoulder"][0]) // 2,
                    (landmarks_to_draw["left_shoulder"][1] + landmarks_to_draw["right_shoulder"][1]) // 2
                )
                
                hip_midpoint = (
                    (landmarks_to_draw["left_hip"][0] + landmarks_to_draw["right_hip"][0]) // 2,
                    (landmarks_to_draw["left_hip"][1] + landmarks_to_draw["right_hip"][1]) // 2
                )
                
                knee_midpoint = (
                    (landmarks_to_draw["left_knee"][0] + landmarks_to_draw["right_knee"][0]) // 2,
                    (landmarks_to_draw["left_knee"][1] + landmarks_to_draw["right_knee"][1]) // 2
                )
                
                # Draw midpoints
                self.visualizer.draw_landmark_point(vis_frame, hip_midpoint[0], hip_midpoint[1], color="primary", size=10)
                self.visualizer.draw_landmark_point(vis_frame, shoulder_midpoint[0], shoulder_midpoint[1], color="primary", size=10)
                self.visualizer.draw_landmark_point(vis_frame, knee_midpoint[0], knee_midpoint[1], color="primary", size=10)
                
                # Draw angle visualization
                if rom_data.current_angle:
                    self.visualizer.draw_angle(
                        vis_frame,
                        shoulder_midpoint,
                        hip_midpoint,
                        knee_midpoint,
                        rom_data.current_angle.value,
                        color="primary",
                        thickness=3,
                        radius=40
                    )
        
        # Draw ROM information panel
        self.visualizer.draw_assessment_info(
            vis_frame,
            rom_data.status.value,
            rom_data.test_type,
            rom_data.current_angle.value if rom_data.current_angle else None,
            rom_data.min_angle,
            rom_data.max_angle,
            position="top_right"
        )
        
        # Draw status and guidance overlay
        if rom_data.status == AssessmentStatus.PREPARING:
            # Draw position guide with progress
            self.visualizer.draw_position_guide(
                vis_frame,
                self.is_ready,
                rom_data.guidance_message,
                (self.ready_time / self.config["ready_time_required"]) * 100
            )
        else:
            # Draw guidance overlay
            self.visualizer.draw_guidance_overlay(vis_frame, rom_data.guidance_message)
        
        # Add ROM gauge if ROM is being measured
        if rom_data.status == AssessmentStatus.IN_PROGRESS or rom_data.status == AssessmentStatus.COMPLETED:
            if rom_data.rom:
                # Get normal ROM value for this test
                normal_rom = 60  # Default for lower back flexion
                if rom_data.test_type == "extension":
                    normal_rom = 25
                elif "lateral_flexion" in rom_data.test_type:
                    normal_rom = 25
                elif "rotation" in rom_data.test_type:
                    normal_rom = 45
                
                # Draw ROM gauge
                self.visualizer.draw_rom_gauge(
                    vis_frame,
                    rom_data.rom,
                    normal_rom,
                    position=(20, h - 80),
                    width=w - 40,
                    height=20
                )
        
        # Add trajectory visualization if history is available
        if hasattr(self, 'motion_visualizer') and rom_data.history:
            # Update motion visualizer
            if not hasattr(self, '_last_updated_angle') or self._last_updated_angle != rom_data.current_angle:
                self._last_updated_angle = rom_data.current_angle
                if rom_data.current_angle:
                    self.motion_visualizer.update_angle("trunk_angle", rom_data.current_angle.value)
            
            # Create trajectory visualization
            traj_img = self.motion_visualizer.create_trajectory_visualization(
                selected_angles=["trunk_angle"],
                min_y=rom_data.min_angle - 10 if rom_data.min_angle != float('inf') else None,
                max_y=rom_data.max_angle + 10 if rom_data.max_angle != float('-inf') else None
            )
            
            # Place trajectory visualization at bottom of frame
            if traj_img is not None:
                traj_height, traj_width = traj_img.shape[:2]
                traj_x = (w - traj_width) // 2
                traj_y = h - traj_height - 120  # Above the guidance overlay
                
                vis_frame[traj_y:traj_y+traj_height, traj_x:traj_x+traj_width] = traj_img
        
        return vis_frame
    # def visualize_assessment(self, frame: np.ndarray, rom_data: ROMData) -> np.ndarray:
    #     """
    #     Visualize the assessment on the frame.
        
    #     Args:
    #         frame: Input video frame
    #         rom_data: Current ROM data
            
    #     Returns:
    #         Frame with visualization
    #     """
    #     h, w, _ = frame.shape
        
    #     # Draw relevant landmarks and connections based on test
    #     if self.visualizer and rom_data.landmarks:
    #         # Get landmark coordinates
    #         landmarks_to_draw = {}
    #         for name, point in rom_data.landmarks.items():
    #             landmarks_to_draw[name] = (int(point.x), int(point.y))
            
    #         # Draw connections for trunk
    #         if all(k in landmarks_to_draw for k in ["left_shoulder", "left_hip", "left_knee"]):
    #             self.visualizer.draw_connection(
    #                 frame,
    #                 landmarks_to_draw["left_shoulder"],
    #                 landmarks_to_draw["left_hip"],
    #                 color="blue"
    #             )
    #             self.visualizer.draw_connection(
    #                 frame,
    #                 landmarks_to_draw["left_hip"],
    #                 landmarks_to_draw["left_knee"],
    #                 color="blue"
    #             )
            
    #         if all(k in landmarks_to_draw for k in ["right_shoulder", "right_hip", "right_knee"]):
    #             self.visualizer.draw_connection(
    #                 frame,
    #                 landmarks_to_draw["right_shoulder"],
    #                 landmarks_to_draw["right_hip"],
    #                 color="blue"
    #             )
    #             self.visualizer.draw_connection(
    #                 frame,
    #                 landmarks_to_draw["right_hip"],
    #                 landmarks_to_draw["right_knee"],
    #                 color="blue"
    #             )
            
    #         # Draw landmarks
    #         for name, point in landmarks_to_draw.items():
    #             self.visualizer.draw_landmark_point(frame, point[0], point[1], color="green")
            
    #         # Draw trunk angle if available
    #         if "left_hip" in landmarks_to_draw and "right_hip" in landmarks_to_draw and \
    #             "left_shoulder" in landmarks_to_draw and "right_shoulder" in landmarks_to_draw:
    #             hip_midpoint = (
    #                 (landmarks_to_draw["left_hip"][0] + landmarks_to_draw["right_hip"][0]) // 2,
    #                 (landmarks_to_draw["left_hip"][1] + landmarks_to_draw["right_hip"][1]) // 2
    #             )
    #             shoulder_midpoint = (
    #                 (landmarks_to_draw["left_shoulder"][0] + landmarks_to_draw["right_shoulder"][0]) // 2,
    #                 (landmarks_to_draw["left_shoulder"][1] + landmarks_to_draw["right_shoulder"][1]) // 2
    #             )
                
    #             # Draw midpoints
    #             self.visualizer.draw_landmark_point(frame, hip_midpoint[0], hip_midpoint[1], color="yellow", size=6)
    #             self.visualizer.draw_landmark_point(frame, shoulder_midpoint[0], shoulder_midpoint[1], color="yellow", size=6)
                
    #             # Draw trunk line
    #             self.visualizer.draw_connection(frame, shoulder_midpoint, hip_midpoint, color="yellow", thickness=3)
                
    #             # Draw angle value
    #             if rom_data.current_angle:
    #                 angle_text = f"{rom_data.current_angle.value:.1f}°"
    #                 text_position = (hip_midpoint[0] + 20, hip_midpoint[1])
    #                 self.visualizer.put_text(frame, angle_text, text_position, color="white")
        
    #     # Draw status and guidance overlay
    #     self._draw_status_overlay(frame, rom_data)
        
    #     return frame
    
    def _draw_status_overlay(self, frame: np.ndarray, rom_data: ROMData) -> None:
        """
        Draw status overlay at the bottom of the frame.
        
        Args:
            frame: Input video frame
            rom_data: Current ROM data
        """
        if not self.visualizer:
            return
            
        h, w, _ = frame.shape
        
        # Draw ROM information
        if rom_data.current_angle:
            # Show current angle
            self.visualizer.put_text(
                frame,
                f"Current: {rom_data.current_angle.value:.1f}°",
                (20, 30),
                color="white"
            )
            
            # Show ROM if available
            if rom_data.min_angle is not None and rom_data.max_angle is not None:
                self.visualizer.put_text(
                    frame,
                    f"ROM: {rom_data.rom:.1f}° (min: {rom_data.min_angle:.1f}°, max: {rom_data.max_angle:.1f}°)",
                    (20, 60),
                    color="white"
                )
        
        # Draw status
        status_text = f"Status: {rom_data.status.value.replace('_', ' ').title()}"
        self.visualizer.put_text(frame, status_text, (20, 90), color="green")
        
        # Draw guidance overlay
        self.visualizer.draw_guidance_overlay(frame, rom_data.guidance_message)
        
        # Draw preparation progress bar if preparing
        if rom_data.status == AssessmentStatus.PREPARING:
            progress = (self.ready_time / self.config["ready_time_required"]) * (w - 40)
            cv2.rectangle(frame, (20, h - 30), (w - 20, h - 10), (255, 255, 255), 2)
            cv2.rectangle(frame, (20, h - 30), (int(20 + progress), h - 10), (0, 255, 0), -1)


class EnhancedLowerBackExtensionTest(EnhancedLowerBackFlexionTest):
    """Enhanced test for lower back extension with Sports2D features."""
    
    def _initialize_rom_data(self) -> ROMData:
        """Initialize ROM data specific to lower back extension test."""
        return ROMData(
            test_type=LowerBackTestType.EXTENSION,
            joint_type=JointType.LOWER_BACK,
            status=AssessmentStatus.NOT_STARTED,
            min_angle=float('inf'),
            max_angle=float('-inf')
        )
    
    def _get_movement_guidance(self) -> str:
        """Get guidance for lower back extension."""
        return "Slowly bend backward as far as comfortable"


class EnhancedLowerBackLateralFlexionTest(EnhancedLowerBackFlexionTest):
    """Enhanced test for lower back lateral flexion with Sports2D features."""
    
    def __init__(self, pose_detector=None, visualizer=None, config=None, side="left"):
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
    
    def _set_default_angles(self):
        """Set default angles to track for lateral flexion."""
        self.joint_angles = [
            "lateral_trunk_angle",
            f"{self.side}_shoulder_hip_angle",
        ]
        
        self.segment_angles = [
            "trunk_segment_lateral",
            f"{self.side}_side_segment"
        ]
    
    def _get_movement_guidance(self) -> str:
        """Get guidance for lateral flexion."""
        return f"Slowly bend to the {self.side} side"
    
    def _get_movement_plane(self) -> MovementPlane:
        """Get movement plane for lateral flexion."""
        return MovementPlane.FRONTAL
    
    def _is_primary_angle(self, angle_name: str) -> bool:
        """Check if angle is primary for lateral flexion."""
        return angle_name == "lateral_trunk_angle"
    
    def calculate_angles(self, landmarks: Dict[int, Tuple[float, float, float]]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Calculate angles for lateral flexion.
        
        Args:
            landmarks: Dictionary of landmark coordinates
            
        Returns:
            Tuple of (joint_angles, segment_angles) dictionaries
        """
        joint_angles = {}
        segment_angles = {}
        
        # Check if key landmarks are available
        if all(idx in landmarks for idx in [11, 12, 23, 24]):
            # Get relevant landmarks
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            
            # Calculate angle differently based on side
            if self.side == "left":
                # For left lateral flexion, use right landmarks as reference
                lateral_angle = MathUtils.calculate_angle(
                    right_shoulder,
                    right_hip,
                    (right_hip[0], right_hip[1] + 100, right_hip[2])  # Point below right hip
                )
                shoulder_hip_angle = MathUtils.calculate_angle(
                    left_shoulder,
                    left_hip,
                    (left_hip[0], left_hip[1] + 100, left_hip[2])
                )
                side_segment = MathUtils.calculate_segment_angle(
                    left_hip, 
                    left_shoulder, 
                    "vertical"
                )
            else:
                # For right lateral flexion, use left landmarks as reference
                lateral_angle = MathUtils.calculate_angle(
                    left_shoulder,
                    left_hip,
                    (left_hip[0], left_hip[1] + 100, left_hip[2])  # Point below left hip
                )
                shoulder_hip_angle = MathUtils.calculate_angle(
                    right_shoulder,
                    right_hip,
                    (right_hip[0], right_hip[1] + 100, right_hip[2])
                )
                side_segment = MathUtils.calculate_segment_angle(
                    right_hip, 
                    right_shoulder, 
                    "vertical"
                )
            
            # Adjust angles to get lateral flexion angle
            lateral_angle = 180 - lateral_angle
            
            joint_angles["lateral_trunk_angle"] = lateral_angle
            joint_angles[f"{self.side}_shoulder_hip_angle"] = shoulder_hip_angle
            
            # Calculate segment angles
            shoulder_midpoint = MathUtils.get_midpoint(left_shoulder, right_shoulder)
            hip_midpoint = MathUtils.get_midpoint(left_hip, right_hip)
            
            trunk_segment_lateral = MathUtils.calculate_segment_angle(
                hip_midpoint, 
                shoulder_midpoint, 
                "vertical"
            )
            
            segment_angles["trunk_segment_lateral"] = trunk_segment_lateral
            segment_angles[f"{self.side}_side_segment"] = side_segment
        
        return joint_angles, segment_angles
    
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


class EnhancedLowerBackRotationTest(EnhancedLowerBackFlexionTest):
    """Enhanced test for lower back rotation with Sports2D features."""
    
    def __init__(self, pose_detector=None, visualizer=None, config=None, side="left"):
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
    
    def _set_default_angles(self):
        """Set default angles to track for rotation."""
        self.joint_angles = [
            "rotation_angle",
            "shoulder_alignment",
            "hip_alignment"
        ]
        
        self.segment_angles = [
            "shoulder_segment",
            "hip_segment"
        ]
    
    def _get_movement_guidance(self) -> str:
        """Get guidance for rotation."""
        return f"Slowly rotate your upper body to the {self.side}"
    
    def _get_movement_plane(self) -> MovementPlane:
        """Get movement plane for rotation."""
        return MovementPlane.TRANSVERSE
    
    def _is_primary_angle(self, angle_name: str) -> bool:
        """Check if angle is primary for rotation."""
        return angle_name == "rotation_angle"
    
    def calculate_angles(self, landmarks: Dict[int, Tuple[float, float, float]]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Calculate angles for rotation.
        
        Args:
            landmarks: Dictionary of landmark coordinates
            
        Returns:
            Tuple of (joint_angles, segment_angles) dictionaries
        """
        joint_angles = {}
        segment_angles = {}
        
        # Check if key landmarks are available
        if all(idx in landmarks for idx in [11, 12, 23, 24]):
            # Get shoulder and hip vectors
            left_shoulder = np.array(landmarks[11])
            right_shoulder = np.array(landmarks[12])
            left_hip = np.array(landmarks[23])
            right_hip = np.array(landmarks[24])
            
            # Project to horizontal plane for rotation measurement
            shoulder_vector = np.array([right_shoulder[0] - left_shoulder[0], 0, right_shoulder[2] - left_shoulder[2]])
            hip_vector = np.array([right_hip[0] - left_hip[0], 0, right_hip[2] - left_hip[2]])
            
            # Normalize vectors
            shoulder_norm = np.linalg.norm(shoulder_vector)
            hip_norm = np.linalg.norm(hip_vector)
            
            if shoulder_norm > 0 and hip_norm > 0:
                shoulder_normalized = shoulder_vector / shoulder_norm
                hip_normalized = hip_vector / hip_norm
                
                # Calculate dot product for angle
                dot_product = np.clip(np.dot(shoulder_normalized, hip_normalized), -1.0, 1.0)
                rotation_angle = np.degrees(np.arccos(dot_product))
                
                # Adjust sign based on side
                if self.side == "right":
                    shoulder_hip_cross = np.cross(hip_normalized, shoulder_normalized)
                    if shoulder_hip_cross[1] < 0:  # Y-axis points up
                        rotation_angle = -rotation_angle
                else:  # left side
                    shoulder_hip_cross = np.cross(shoulder_normalized, hip_normalized)
                    if shoulder_hip_cross[1] < 0:
                        rotation_angle = -rotation_angle
                
                joint_angles["rotation_angle"] = abs(rotation_angle)
                
                # Calculate additional angles
                # Shoulder alignment (horizontal)
                shoulder_segment = MathUtils.calculate_segment_angle(
                    (left_shoulder[0], left_shoulder[1], 0),
                    (right_shoulder[0], right_shoulder[1], 0)
                )
                
                # Hip alignment (horizontal)
                hip_segment = MathUtils.calculate_segment_angle(
                    (left_hip[0], left_hip[1], 0),
                    (right_hip[0], right_hip[1], 0)
                )
                
                joint_angles["shoulder_alignment"] = shoulder_segment
                joint_angles["hip_alignment"] = hip_segment
                
                segment_angles["shoulder_segment"] = shoulder_segment
                segment_angles["hip_segment"] = hip_segment
        
        return joint_angles, segment_angles
    
    def check_position(self, landmarks: Dict[int, Tuple[float, float, float]], frame_shape: Tuple[int, int, int]) -> Tuple[bool, str]:
        """
        Check if the person is in the correct position for rotation.
        
        Args:
            landmarks: Dictionary of landmark coordinates
            frame_shape: Shape of the frame
            
        Returns:
            Tuple of (is_position_valid, guidance_message)
        """
        is_valid, generic_message = super().check_position(landmarks, frame_shape)
        
        if not is_valid:
            return False, generic_message
        
        # Additional check: person should be facing the camera
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
        
        if shoulder_width < frame_shape[1] * 0.15:
            return False, "Turn to face the camera directly"
        
        return True, f"Good position for {self.side} rotation"
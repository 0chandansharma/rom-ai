# rom/core/configurable_test.py
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from dataclasses import dataclass, field
import logging
import time
import json

from rom.core.base import EnhancedROMTest, ROMData, JointType, MovementPlane, AssessmentStatus, Point3D, Angle
from rom.utils.math_utils import MathUtils
from rom.config.config_manager import ConfigManager

logger = logging.getLogger("rom.configurable_test")

@dataclass
class AngleDefinition:
    """Definition of an angle to measure."""
    name: str
    points: List[str]
    type: str = "joint"  # "joint" or "segment"
    reference: str = "horizontal"  # For segment angles: "horizontal" or "vertical"
    is_primary: bool = False

class ConfigurableROMTest(EnhancedROMTest):
    """
    Highly configurable ROM test that can be customized with user preferences.
    
    This class provides a flexible framework for creating custom ROM tests
    with configurable body parts, angles, and visualization options.
    """
    
    def __init__(self, 
                pose_detector=None, 
                visualizer=None, 
                config=None, 
                test_type=None,
                config_manager=None):
        """
        Initialize configurable ROM test.
        
        Args:
            pose_detector: Pose detection system
            visualizer: Visualization system
            config: Configuration parameters
            test_type: Type of test (for loading test-specific config)
            config_manager: Configuration manager instance
        """
        # Initialize base configuration from config manager if provided
        self.config_manager = config_manager or ConfigManager()
        self.test_type = test_type
        
        # Get complete configuration for this test
        complete_config = self.config_manager.get_complete_config(test_type)
        
        # Merge with provided config (which takes precedence)
        if config:
            complete_config.update(config)
        
        # Initialize base class
        super().__init__(pose_detector, visualizer, complete_config)
        
        # Parse angle definitions
        self.angle_definitions = self._parse_angle_definitions(complete_config)
        
        # Set primary angle
        self.primary_angle = complete_config.get("primary_angle")
        
        # Set body parts to track
        self.body_parts = complete_config.get("body_parts", [])
        
        # Set target ROM value
        self.target_rom = complete_config.get("target_rom", 0)
        
        # Create motion visualizer if needed
        if visualizer and self.config.get("visualization", {}).get("show_trajectory", True):
            try:
                from rom.analysis.motion_visualizer import MotionVisualizer
                self.motion_visualizer = MotionVisualizer(
                    history_length=self.config.get("visualization", {}).get("trajectory_length", 100)
                )
            except ImportError:
                logger.warning("MotionVisualizer not available, trajectory visualization disabled")
        
        logger.info(f"Initialized ConfigurableROMTest with {len(self.angle_definitions)} angle definitions")
    
    def _parse_angle_definitions(self, config: Dict[str, Any]) -> List[AngleDefinition]:
        """
        Parse angle definitions from configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            List of angle definitions
        """
        angle_definitions = []
        
        # Process joint angles
        joint_angles = config.get("joint_angles", [])
        if isinstance(joint_angles, list):
            for angle in joint_angles:
                if isinstance(angle, str):
                    # Simple angle name, need to look up definition
                    # For now, just use placeholder
                    angle_definitions.append(AngleDefinition(
                        name=angle,
                        points=["left_shoulder", "left_hip", "left_knee"],
                        type="joint"
                    ))
                elif isinstance(angle, dict):
                    # Complete angle definition
                    angle_definitions.append(AngleDefinition(
                        name=angle.get("name", f"joint_angle_{len(angle_definitions)}"),
                        points=angle.get("points", []),
                        type="joint",
                        is_primary=angle.get("name") == config.get("primary_angle")
                    ))
        
        # Process segment angles
        segment_angles = config.get("segment_angles", [])
        if isinstance(segment_angles, list):
            for angle in segment_angles:
                if isinstance(angle, str):
                    # Simple angle name, need to look up definition
                    # For now, just use placeholder
                    angle_definitions.append(AngleDefinition(
                        name=angle,
                        points=["left_hip", "left_knee"],
                        type="segment",
                        reference="horizontal"
                    ))
                elif isinstance(angle, dict):
                    # Complete angle definition
                    angle_definitions.append(AngleDefinition(
                        name=angle.get("name", f"segment_angle_{len(angle_definitions)}"),
                        points=angle.get("points", []),
                        type="segment",
                        reference=angle.get("reference", "horizontal"),
                        is_primary=angle.get("name") == config.get("primary_angle")
                    ))
        
        return angle_definitions
    
    def _initialize_rom_data(self) -> ROMData:
        """Initialize ROM data for this test."""
        joint_type = JointType.LOWER_BACK
        if self.test_type:
            if "shoulder" in self.test_type:
                joint_type = JointType.SHOULDER
            elif "elbow" in self.test_type:
                joint_type = JointType.ELBOW
            elif "hip" in self.test_type:
                joint_type = JointType.HIP
            elif "knee" in self.test_type:
                joint_type = JointType.KNEE
            elif "ankle" in self.test_type:
                joint_type = JointType.ANKLE
            elif "neck" in self.test_type:
                joint_type = JointType.NECK
        
        return ROMData(
            test_type=self.test_type or "custom",
            joint_type=joint_type,
            status=AssessmentStatus.NOT_STARTED,
            min_angle=float('inf'),
            max_angle=float('-inf')
        )
    
    def _get_required_landmarks(self) -> List[int]:
        """
        Get required landmarks for this test.
        
        Returns:
            List of landmark IDs
        """
        required_landmarks = set()
        
        # Add landmarks from body parts
        for part in self.body_parts:
            if part in self.pose_detector.keypoint_mapping:
                required_landmarks.add(self.pose_detector.keypoint_mapping[part])
        
        # Add landmarks from angle definitions
        for angle_def in self.angle_definitions:
            for point in angle_def.points:
                if point in self.pose_detector.keypoint_mapping:
                    required_landmarks.add(self.pose_detector.keypoint_mapping[point])
        
        # If no landmarks specified, use a default set
        if not required_landmarks:
            # Include core landmarks for any test
            for part in ["left_shoulder", "right_shoulder", "left_hip", "right_hip", 
                        "left_knee", "right_knee", "left_ankle", "right_ankle"]:
                if part in self.pose_detector.keypoint_mapping:
                    required_landmarks.add(self.pose_detector.keypoint_mapping[part])
        
        return list(required_landmarks)
    
    def _is_primary_angle(self, angle_name: str) -> bool:
        """
        Check if angle is the primary one for this test.
        
        Args:
            angle_name: Name of the angle
            
        Returns:
            True if primary angle, False otherwise
        """
        if self.primary_angle:
            return angle_name == self.primary_angle
        
        # If no primary angle specified, use the first angle definition
        if self.angle_definitions:
            return angle_name == self.angle_definitions[0].name
        
        return False
    
    def _update_landmarks(self, landmarks: Dict[int, Tuple[float, float, float]]):
        """
        Update ROM data with landmark positions.
        
        Args:
            landmarks: Dictionary of landmark coordinates
        """
        updated_landmarks = {}
        
        # Add specified body parts
        for part in self.body_parts:
            if part in self.pose_detector.keypoint_mapping:
                idx = self.pose_detector.keypoint_mapping[part]
                if idx in landmarks:
                    updated_landmarks[part] = Point3D.from_tuple(landmarks[idx])
        
        # Add points from angle definitions
        for angle_def in self.angle_definitions:
            for point in angle_def.points:
                if point in self.pose_detector.keypoint_mapping:
                    idx = self.pose_detector.keypoint_mapping[point]
                    if idx in landmarks:
                        updated_landmarks[point] = Point3D.from_tuple(landmarks[idx])
        
        # If no landmarks to update, use all available landmarks
        if not updated_landmarks:
            for name, idx in self.pose_detector.keypoint_mapping.items():
                if idx in landmarks:
                    updated_landmarks[name] = Point3D.from_tuple(landmarks[idx])
        
        self.data.landmarks = updated_landmarks
    
    def calculate_angles(self, landmarks: Dict[int, Tuple[float, float, float]]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Calculate angles based on angle definitions.
        
        Args:
            landmarks: Dictionary of landmark coordinates
            
        Returns:
            Tuple of (joint_angles, segment_angles) dictionaries
        """
        joint_angles = {}
        segment_angles = {}
        
        # Process each angle definition
        for angle_def in self.angle_definitions:
            # Skip if not enough points in definition
            if len(angle_def.points) < 2:
                continue
            
            # Map point names to landmark IDs
            point_ids = []
            for point in angle_def.points:
                if point in self.pose_detector.keypoint_mapping:
                    point_ids.append(self.pose_detector.keypoint_mapping[point])
            
            # Skip if not all points are available
            if not all(pid in landmarks for pid in point_ids):
                continue
            
            # Calculate angle based on type
            if angle_def.type == "joint":
                # Need exactly 3 points for a joint angle
                if len(point_ids) != 3:
                    continue
                
                angle = MathUtils.calculate_angle(
                    landmarks[point_ids[0]],
                    landmarks[point_ids[1]],
                    landmarks[point_ids[2]]
                )
                
                joint_angles[angle_def.name] = angle
                
            elif angle_def.type == "segment":
                # Need exactly 2 points for a segment angle
                if len(point_ids) != 2:
                    continue
                
                angle = MathUtils.calculate_segment_angle(
                    landmarks[point_ids[0]],
                    landmarks[point_ids[1]],
                    angle_def.reference
                )
                
                segment_angles[angle_def.name] = angle
        
        return joint_angles, segment_angles
    
    def check_position(self, landmarks: Dict[int, Tuple[float, float, float]], frame_shape: Tuple[int, int, int]) -> Tuple[bool, str]:
        """
        Check if the person is in the correct position for the test.
        
        Args:
            landmarks: Dictionary of landmark coordinates
            frame_shape: Shape of the frame
            
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
        
        # Check if person is visible enough (based on shoulder width)
        left_shoulder_idx = self.pose_detector.keypoint_mapping.get("left_shoulder")
        right_shoulder_idx = self.pose_detector.keypoint_mapping.get("right_shoulder")
        
        if left_shoulder_idx in landmarks and right_shoulder_idx in landmarks:
            left_shoulder = landmarks[left_shoulder_idx]
            right_shoulder = landmarks[right_shoulder_idx]
            shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
            
            if shoulder_width < w * 0.15:
                messages.append("Move closer to camera")
                is_valid = False
            elif shoulder_width > w * 0.5:
                messages.append("Step back from camera")
                is_valid = False
        
        # Check if person is centered
        if left_shoulder_idx in landmarks and right_shoulder_idx in landmarks:
            center_x = (landmarks[left_shoulder_idx][0] + landmarks[right_shoulder_idx][0]) / 2
            
            if center_x < w * 0.3:
                messages.append("Move right")
                is_valid = False
            elif center_x > w * 0.7:
                messages.append("Move left")
                is_valid = False
        
        # Check if person is standing straight
        left_hip_idx = self.pose_detector.keypoint_mapping.get("left_hip")
        right_hip_idx = self.pose_detector.keypoint_mapping.get("right_hip")
        
        if (left_shoulder_idx in landmarks and right_shoulder_idx in landmarks and
            left_hip_idx in landmarks and right_hip_idx in landmarks):
            
            # Calculate hip-shoulder alignment
            hip_shoulder_angle = MathUtils.calculate_angle(
                landmarks[left_shoulder_idx],
                ((landmarks[left_hip_idx][0] + landmarks[right_hip_idx][0]) / 2,
                 (landmarks[left_hip_idx][1] + landmarks[right_hip_idx][1]) / 2,
                 0),
                landmarks[right_shoulder_idx]
            )
            
            # Tolerance from configuration
            position_tolerance = self.config.get("position_tolerance", 10)
            
            if not (90 - position_tolerance <= hip_shoulder_angle <= 90 + position_tolerance):
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
        # Skip visualization if no visualizer available
        if not self.visualizer:
            return frame
        
        h, w, _ = frame.shape
        
        # Create a copy of the frame for visualization
        vis_frame = frame.copy()
        
        # Get visualization settings
        vis_config = self.config.get("visualization", {})
        show_landmarks = vis_config.get("show_landmarks", True)
        show_connections = vis_config.get("show_connections", True)
        show_angles = vis_config.get("show_angles", True)
        show_info_panel = vis_config.get("show_info_panel", True)
        show_trajectory = vis_config.get("show_trajectory", True)
        display_mode = vis_config.get("display_mode", "body")
        
        # Draw landmarks and connections
        if (show_landmarks or show_connections) and rom_data.landmarks:
            # Get landmark coordinates
            landmarks_to_draw = {}
            for name, point in rom_data.landmarks.items():
                landmarks_to_draw[name] = (int(point.x), int(point.y))
            
            # Draw connections
            if show_connections:
                # Define connections based on body parts
                connections = []
                
                # Standard connections for common body parts
                standard_connections = [
                    ("left_shoulder", "right_shoulder"),
                    ("left_shoulder", "left_elbow"),
                    ("left_elbow", "left_wrist"),
                    ("right_shoulder", "right_elbow"),
                    ("right_elbow", "right_wrist"),
                    ("left_shoulder", "left_hip"),
                    ("right_shoulder", "right_hip"),
                    ("left_hip", "right_hip"),
                    ("left_hip", "left_knee"),
                    ("left_knee", "left_ankle"),
                    ("right_hip", "right_knee"),
                    ("right_knee", "right_ankle")
                ]
                
                # Add connections that are relevant for this test
                for start, end in standard_connections:
                    if start in landmarks_to_draw and end in landmarks_to_draw:
                        connections.append((start, end))
                
                # Draw each connection
                for start_name, end_name in connections:
                    self.visualizer.draw_connection(
                        vis_frame,
                        landmarks_to_draw[start_name],
                        landmarks_to_draw[end_name],
                        color="blue"
                    )
            
            # Draw landmarks
            if show_landmarks:
                for name, point in landmarks_to_draw.items():
                    highlight = self.primary_angle and name in next(
                        (angle.points for angle in self.angle_definitions 
                         if angle.name == self.primary_angle), 
                        []
                    )
                    
                    self.visualizer.draw_landmark_point(
                        vis_frame, 
                        point[0], 
                        point[1], 
                        color="yellow" if highlight else "white",
                        size=8 if highlight else 6,
                        highlight=highlight
                    )
        
        # Draw angles
        if show_angles:
            for angle_def in self.angle_definitions:
                # Check if all points are available
                if not all(point in rom_data.landmarks for point in angle_def.points):
                    continue
                
                # Get point coordinates
                points = [
                    (int(rom_data.landmarks[point].x), int(rom_data.landmarks[point].y))
                    for point in angle_def.points
                ]
                
                # Draw angle based on type
                if angle_def.type == "joint" and len(points) == 3:
                    # Find angle value
                    angle_value = 0
                    if rom_data.current_angle and rom_data.current_angle.value is not None and angle_def.is_primary:
                        angle_value = rom_data.current_angle.value
                    else:
                        # Calculate angle from points
                        angle_value = MathUtils.calculate_angle(
                            (points[0][0], points[0][1], 0),
                            (points[1][0], points[1][1], 0),
                            (points[2][0], points[2][1], 0)
                        )
                    
                    # Draw joint angle
                    self.visualizer.draw_angle(
                        vis_frame,
                        points[0],
                        points[1],
                        points[2],
                        angle_value,
                        color="primary" if angle_def.is_primary else "info",
                        radius=40,
                        display_mode="all" if display_mode in ["body", "both"] else "none"
                    )
                
                elif angle_def.type == "segment" and len(points) == 2:
                    # Calculate angle value
                    angle_value = MathUtils.calculate_segment_angle(
                        (points[0][0], points[0][1], 0),
                        (points[1][0], points[1][1], 0),
                        angle_def.reference
                    )
                    
                    # Draw segment angle
                    self.visualizer.draw_segment_angle(
                        vis_frame,
                        points[0],
                        points[1],
                        angle_value,
                        reference=angle_def.reference,
                        color="primary" if angle_def.is_primary else "info"
                    )
        
        # Draw information panel
        if show_info_panel:
            self.visualizer.draw_assessment_info(
                vis_frame,
                rom_data.status.value,
                rom_data.test_type,
                rom_data.current_angle.value if rom_data.current_angle else None,
                rom_data.min_angle,
                rom_data.max_angle,
                position="top_right"
            )
        
        # Draw guidance overlay
        if rom_data.status == AssessmentStatus.PREPARING:
            # Draw position guide with progress
            self.visualizer.draw_position_guide(
                vis_frame,
                self.is_ready,
                rom_data.guidance_message,
                (self.ready_time / self.config.get("ready_time_required", 20)) * 100
            )
        else:
            # Draw guidance overlay
            self.visualizer.draw_guidance_overlay(
                vis_frame, 
                rom_data.guidance_message
            )
        
        # Draw ROM gauge if measuring
        if (rom_data.status == AssessmentStatus.IN_PROGRESS or 
            rom_data.status == AssessmentStatus.COMPLETED) and rom_data.rom:
            
            # Draw ROM gauge
            self.visualizer.draw_rom_gauge(
                vis_frame,
                rom_data.rom,
                self.target_rom,
                position=(20, h - 80),
                width=w - 40,
                height=20
            )
        
        # Draw trajectory visualization
        if show_trajectory and hasattr(self, 'motion_visualizer') and rom_data.current_angle:
            # Update motion visualizer
            if not hasattr(self, '_last_updated_angle') or self._last_updated_angle != rom_data.current_angle:
                self._last_updated_angle = rom_data.current_angle
                if rom_data.current_angle:
                    self.motion_visualizer.update_angle(
                        self.primary_angle or "angle", 
                        rom_data.current_angle.value
                    )
            
            # Create trajectory visualization
            traj_img = self.motion_visualizer.create_trajectory_visualization(
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
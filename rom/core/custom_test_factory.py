# rom/core/custom_test_factory.py
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import cv2
import mediapipe as mp
from rom.core.base import EnhancedROMTest, ROMData, JointType, MovementPlane, AssessmentStatus, Point3D, Angle
from rom.utils.math_utils import MathUtils
from rom.tests.lower_back_test import EnhancedLowerBackFlexionTest

class CustomROMTest(EnhancedLowerBackFlexionTest):
    """
    Custom ROM test with user-defined body parts and angles.
    
    This class extends the lower back flexion test with customizable
    body parts and angle definitions.
    """
    
    def __init__(self, pose_detector=None, visualizer=None, config=None):
        """
        Initialize custom ROM test.
        
        Args:
            pose_detector: Pose detection system
            visualizer: Visualization system
            config: Configuration parameters with custom body parts and angles
        """
        # Initialize with base configuration
        super().__init__(pose_detector, visualizer, config)
        
        # Extract custom body parts and angle definitions
        self.body_parts = self.config.get("body_parts", [])
        self.custom_joint_angles = self.config.get("joint_angles", [])
        self.custom_segment_angles = self.config.get("segment_angles", [])
        self.primary_angle = self.config.get("primary_angle")
        
        # Create motion visualizer if not already done
        if visualizer and not hasattr(self, 'motion_visualizer'):
            from rom.analysis.motion_visualizer import MotionVisualizer
            self.motion_visualizer = MotionVisualizer()
    
    def _get_required_landmarks(self) -> List[int]:
        """
        Get required landmarks based on custom configuration.
        
        Returns:
            List of required landmark IDs
        """
        if not self.body_parts:
            return super()._get_required_landmarks()
        
        # Map body part names to landmark IDs
        landmark_ids = []
        keypoint_mapping = self.pose_detector.keypoint_mapping
        
        for part in self.body_parts:
            if part in keypoint_mapping:
                landmark_ids.append(keypoint_mapping[part])
        
        # Add IDs from angle definitions
        for angle_def in self.custom_joint_angles:
            if "points" in angle_def:
                for point in angle_def["points"]:
                    if point in keypoint_mapping:
                        landmark_ids.append(keypoint_mapping[point])
        
        for angle_def in self.custom_segment_angles:
            if "points" in angle_def:
                for point in angle_def["points"]:
                    if point in keypoint_mapping:
                        landmark_ids.append(keypoint_mapping[point])
        
        return list(set(landmark_ids))  # Unique IDs
    
    def calculate_angles(self, landmarks: Dict[int, Tuple[float, float, float]]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Calculate angles based on custom definitions.
        
        Args:
            landmarks: Dictionary of landmark coordinates
            
        Returns:
            Tuple of (joint_angles, segment_angles) dictionaries
        """
        joint_angles = {}
        segment_angles = {}
        
        # Process custom joint angles
        for angle_def in self.custom_joint_angles:
            if "name" not in angle_def or "points" not in angle_def:
                continue
            
            name = angle_def["name"]
            points = angle_def["points"]
            
            # Map point names to landmark IDs
            point_ids = []
            for point in points:
                if point in self.pose_detector.keypoint_mapping:
                    point_ids.append(self.pose_detector.keypoint_mapping[point])
            
            # Need exactly 3 points for a joint angle
            if len(point_ids) != 3 or not all(pid in landmarks for pid in point_ids):
                continue
            
            # Calculate angle
            angle = MathUtils.calculate_angle(
                landmarks[point_ids[0]],
                landmarks[point_ids[1]],
                landmarks[point_ids[2]]
            )
            
            joint_angles[name] = angle
        
        # Process custom segment angles
        for angle_def in self.custom_segment_angles:
            if "name" not in angle_def or "points" not in angle_def:
                continue
            
            name = angle_def["name"]
            points = angle_def["points"]
            reference = angle_def.get("reference", "horizontal")
            
            # Map point names to landmark IDs
            point_ids = []
            for point in points:
                if point in self.pose_detector.keypoint_mapping:
                    point_ids.append(self.pose_detector.keypoint_mapping[point])
            
            # Need exactly 2 points for a segment angle
            if len(point_ids) != 2 or not all(pid in landmarks for pid in point_ids):
                continue
            
            # Calculate angle
            angle = MathUtils.calculate_segment_angle(
                landmarks[point_ids[0]],
                landmarks[point_ids[1]],
                reference
            )
            
            segment_angles[name] = angle
        
        # If no custom angles defined, fall back to parent implementation
        if not joint_angles and not segment_angles:
            return super().calculate_angles(landmarks)
        
        return joint_angles, segment_angles
    
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
        return super()._is_primary_angle(angle_name)
    
    def _update_landmarks(self, landmarks: Dict[int, Tuple[float, float, float]]):
        """
        Update ROM data with custom landmark positions.
        
        Args:
            landmarks: Dictionary of landmark coordinates
        """
        updated_landmarks = {}
        
        # Add all specified body parts to landmarks
        for part_name in self.body_parts:
            if part_name in self.pose_detector.keypoint_mapping:
                idx = self.pose_detector.keypoint_mapping[part_name]
                if idx in landmarks:
                    updated_landmarks[part_name] = Point3D.from_tuple(landmarks[idx])
        
        # Add points from angle definitions
        for angle_def in self.custom_joint_angles + self.custom_segment_angles:
            if "points" in angle_def:
                for point in angle_def["points"]:
                    if point in self.pose_detector.keypoint_mapping:
                        idx = self.pose_detector.keypoint_mapping[point]
                        if idx in landmarks:
                            updated_landmarks[point] = Point3D.from_tuple(landmarks[idx])
        
        self.data.landmarks = updated_landmarks


class CustomTestFactory:
    """
    Factory for creating custom ROM tests.
    """
    
    @staticmethod
    def create_test(body_parts: List[str], 
                    joint_angles: List[Dict[str, Any]] = None,
                    segment_angles: List[Dict[str, Any]] = None,
                    primary_angle: str = None,
                    pose_detector=None, 
                    visualizer=None, 
                    base_config: Dict[str, Any] = None) -> EnhancedROMTest:
        """
        Create a custom ROM test with specified body parts and angles.
        
        Args:
            body_parts: List of body part names to track
            joint_angles: List of joint angle definitions
            segment_angles: List of segment angle definitions
            primary_angle: Name of the primary angle to track
            pose_detector: Pose detection system
            visualizer: Visualization system
            base_config: Base configuration parameters
            
        Returns:
            Customized ROM test instance
        """
        config = base_config or {}
        
        # Add custom configuration
        config["body_parts"] = body_parts
        config["joint_angles"] = joint_angles or []
        config["segment_angles"] = segment_angles or []
        config["primary_angle"] = primary_angle
        
        # Create custom test
        return CustomROMTest(pose_detector, visualizer, config)
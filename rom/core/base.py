# rom/core/base.py
from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Optional, Any
from enum import Enum
import numpy as np
import cv2
import time
import logging
from dataclasses import dataclass, field

from rom.utils.pose_detector import PoseDetector
from rom.utils.math_utils import MathUtils
from rom.core.data_processor import DataProcessor
from rom.utils.visualization import EnhancedVisualizer
# Setup logging
logger = logging.getLogger("rom.base")

class JointType(Enum):
    """Enum for types of joints measured in ROM assessments."""
    SHOULDER = "shoulder"
    ELBOW = "elbow"
    WRIST = "wrist"
    HIP = "hip"
    KNEE = "knee"
    ANKLE = "ankle"
    NECK = "neck"
    SPINE = "spine"
    LOWER_BACK = "lower_back"


class MovementPlane(Enum):
    """Enum for planes of movement."""
    SAGITTAL = "sagittal"  # Forward/backward movements
    FRONTAL = "frontal"    # Side-to-side movements
    TRANSVERSE = "transverse"  # Rotational movements


class AssessmentStatus(Enum):
    """Status of the assessment process."""
    NOT_STARTED = "not_started"
    PREPARING = "preparing"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Point3D:
    """3D point with x, y, z coordinates."""
    x: float
    y: float
    z: float = 0

    def as_tuple(self) -> Tuple[float, float, float]:
        """Return the point as a tuple."""
        return (self.x, self.y, self.z)

    def as_xy_tuple(self) -> Tuple[float, float]:
        """Return just the x, y coordinates as a tuple."""
        return (self.x, self.y)

    def as_dict(self) -> Dict[str, float]:
        """Return the point as a dictionary."""
        return {"x": self.x, "y": self.y, "z": self.z}

    @classmethod
    def from_tuple(cls, point: Tuple[float, float, float]) -> 'Point3D':
        """Create a Point3D from a tuple."""
        return cls(x=point[0], y=point[1], z=point[2] if len(point) > 2 else 0)


@dataclass
class Angle:
    """Represents an angle measurement with additional metadata."""
    value: float
    joint_type: JointType
    plane: MovementPlane
    timestamp: float  # Timestamp when the measurement was taken
    confidence: float = 1.0  # Confidence level of the measurement (0-1)


@dataclass
class ROMData:
    """Base class for Range of Motion data."""
    test_type: str
    joint_type: JointType
    status: AssessmentStatus = AssessmentStatus.NOT_STARTED
    current_angle: Optional[Angle] = None
    min_angle: Optional[float] = None
    max_angle: Optional[float] = None
    rom: Optional[float] = None  # Calculated ROM (max - min)
    landmarks: Dict[str, Point3D] = field(default_factory=dict)
    history: List[Angle] = field(default_factory=list)
    guidance_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert ROM data to a dictionary for API responses."""
        return {
            "test_type": self.test_type,
            "joint_type": self.joint_type.value,
            "status": self.status.value,
            "current_angle": self.current_angle.value if self.current_angle else None,
            "min_angle": self.min_angle,
            "max_angle": self.max_angle,
            "rom": self.rom,
            "landmarks": {k: v.as_dict() for k, v in self.landmarks.items()},
            "guidance_message": self.guidance_message,
            "metadata": self.metadata
        }

    def update_rom(self) -> None:
        """Update the ROM calculation based on history."""
        if not self.history:
            return

        angles = [angle.value for angle in self.history]
        self.min_angle = min(angles)
        self.max_angle = max(angles)
        self.rom = self.max_angle - self.min_angle


class EnhancedROMTest(ABC):
    """Enhanced base class for ROM tests with Sports2D features."""
    
    def __init__(self, 
                pose_detector: Optional[PoseDetector] = None,
                visualizer = EnhancedVisualizer(),
                config: Dict[str, Any] = None):
        """
        Initialize a ROM test.
        
        Args:
            pose_detector: The pose detection system
            visualizer: The visualization system
            config: Configuration parameters
        """
        # Default configuration
        self.default_config = {
            "ready_time_required": 20,  # Frames to consider position ready
            "angle_buffer_size": 100,  # Size of the angle history buffer
            "position_tolerance": 10,  # Degrees of tolerance for position checks
            "smoothing_window": 5,  # Window size for angle smoothing
            "filter_type": "butterworth",  # Type of filter to use
            "interpolate": True,  # Whether to interpolate missing data
            "max_gap_size": 10,  # Maximum gap size for interpolation
            "butterworth_cutoff": 6.0,  # Cutoff frequency for Butterworth filter
            "butterworth_order": 4,  # Order for Butterworth filter
            "fps": 30.0,  # Frame rate for filtering calculations
            "person_id": 0,  # Person ID to track
            "multiperson": False,  # Whether to track multiple persons
            "relevant_body_parts": [],  # Body parts relevant to this test
            "joint_angles": [],  # Joint angles to calculate
            "segment_angles": [],  # Segment angles to calculate
            "keypoint_likelihood_threshold": 0.3,  # Minimum confidence for keypoints
            "keypoint_number_threshold": 0.3  # Minimum fraction of keypoints detected
        }
        
        # Merge provided config with defaults
        self.config = {**self.default_config, **(config or {})}
        
        # Initialize components
        self.pose_detector = pose_detector or PoseDetector(
            keypoint_likelihood_threshold=self.config["keypoint_likelihood_threshold"],
            keypoint_number_threshold=self.config["keypoint_number_threshold"]
        )
        self.visualizer = visualizer
        
        # Initialize data processor
        self.data_processor = DataProcessor(
            filter_type=self.config["filter_type"],
            interpolate=self.config["interpolate"],
            max_gap_size=self.config["max_gap_size"],
            smoothing_window=self.config["smoothing_window"],
            butterworth_cutoff=self.config["butterworth_cutoff"],
            butterworth_order=self.config["butterworth_order"],
            fps=self.config["fps"]
        )
        
        # Initialize ROM data
        self.data = self._initialize_rom_data()
        
        # State variables
        self.ready_time = 0
        self.is_ready = False
        self.start_time = None
        self.end_time = None
        self.frame_count = 0
        
        # Define angles to track (joint and segment)
        self._init_angles_to_track()
    
    def _init_angles_to_track(self):
        """Initialize angles to track based on configuration."""
        self.joint_angles = self.config.get("joint_angles", [])
        self.segment_angles = self.config.get("segment_angles", [])
        
        # If no angles specified, use defaults based on test type
        if not self.joint_angles and not self.segment_angles:
            self._set_default_angles()
    
    def _set_default_angles(self):
        """Set default angles to track based on test type."""
        # To be implemented by subclasses
        pass
    
    @abstractmethod
    def _initialize_rom_data(self) -> ROMData:
        """Initialize ROM data specific to this test."""
        pass
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process a single frame and return the processed frame and ROM data.
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (processed_frame, rom_data_dict)
        """
        self.frame_count += 1
        h, w, _ = frame.shape
        frame_shape = (h, w, 3)
        # print("h, w : ",h, w)
        # Find pose landmarks
        all_landmarks = self.pose_detector.find_pose(frame)
        # print("all_landmarks", all_landmarks)
        if not all_landmarks:
            self.data.guidance_message = "No pose detected. Please make sure your full body is visible."
            self.data.status = AssessmentStatus.FAILED
            
            # Create a simple frame with message
            if self.visualizer:
                frame = self.visualizer.put_text(frame, self.data.guidance_message, (20, 50), color='red')
            
            return frame, self.data.to_dict()
        
        # Select landmarks based on person_id
        if self.config["multiperson"]:
            # Process all persons
            landmarks_dict = all_landmarks
        else:
            # Process only the specified person_id
            person_id = min(self.config["person_id"], len(all_landmarks) - 1)
            landmarks_dict = all_landmarks[person_id] if person_id < len(all_landmarks) else {}
        
        # Check if required landmarks are detected
        if not self._check_required_landmarks(landmarks_dict):
            self.data.guidance_message = "Cannot detect all required body parts. Please step back to show your full body."
            self.data.status = AssessmentStatus.FAILED
            
            print("frame-----before", len(frame))
            if self.visualizer:
                frame = self.visualizer.put_text(frame, self.data.guidance_message, (20, 50), color='red') #ERROR here TODO
            
            print("frame-----", len(frame))
            return frame, self.data.to_dict()
        
        # Check initial position
        is_valid_position, guidance_message = self.check_position(landmarks_dict, frame_shape)
        
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
                    self.data.guidance_message = self._get_movement_guidance()
            else:
                self.ready_time = 0
                self.is_ready = False
                self.data.guidance_message = guidance_message
        
        # Calculate angles
        joint_angles, segment_angles = self.calculate_angles(landmarks_dict)
        
        # Store landmarks in ROM data
        self._update_landmarks(landmarks_dict)
        
        # Process joint angles
        for angle_name, angle_value in joint_angles.items():
            self.data_processor.add_angle(angle_name, angle_value)
            
            # For the primary angle, also update ROM data
            if self._is_primary_angle(angle_name):
                if angle_value is not None:
                    self.data.current_angle = Angle(
                        value=angle_value,
                        joint_type=self.data.joint_type,
                        plane=self._get_movement_plane(),
                        timestamp=time.time()
                    )
                    
                    # Only record angles if assessment is in progress
                    if self.data.status == AssessmentStatus.IN_PROGRESS:
                        self.data.history.append(self.data.current_angle)
                        self.data.update_rom()
                        
                        # Check if assessment is completed
                        if self._is_assessment_completed():
                            self.data.status = AssessmentStatus.COMPLETED
                            self.end_time = time.time()
                            self.data.metadata["duration"] = self.end_time - self.start_time
                            self.data.guidance_message = "Assessment completed"
        
        # Process segment angles
        for angle_name, angle_value in segment_angles.items():
            self.data_processor.add_angle(angle_name, angle_value)
        
        # Visualize assessment
        # print("processed_frame_visu_before++++++++++++++++++++++++++++++++++++++++++++++", processed_frame)
        processed_frame = self.visualize_assessment(frame, self.data)
        # print("processed_frame_visu_after________________________________________________", processed_frame)
        return processed_frame, self.data.to_dict()
    
    def _check_required_landmarks(self, landmarks: Dict[int, Tuple[float, float, float]]) -> bool:
        """
        Check if all required landmarks are detected.
        
        Args:
            landmarks: Dictionary of landmark coordinates
            
        Returns:
            True if all required landmarks are detected, False otherwise
        """
        required_landmarks = self._get_required_landmarks()
        return all(landmark_id in landmarks for landmark_id in required_landmarks)
    
    def _get_required_landmarks(self) -> List[int]:
        """
        Get list of required landmark IDs for this test.
        
        Returns:
            List of landmark IDs
        """
        # Default implementation - override in subclasses
        return []
    
    def _get_movement_guidance(self) -> str:
        """
        Get guidance message for the movement phase.
        
        Returns:
            Guidance message
        """
        return "Begin the movement slowly"
    
    def _get_movement_plane(self) -> MovementPlane:
        """
        Get the movement plane for this test.
        
        Returns:
            Movement plane
        """
        return MovementPlane.SAGITTAL
    
    def _is_primary_angle(self, angle_name: str) -> bool:
        """
        Check if the angle is the primary one for this test.
        
        Args:
            angle_name: Name of the angle
            
        Returns:
            True if primary angle, False otherwise
        """
        # Default implementation - override in subclasses
        return True
    
    def _is_assessment_completed(self) -> bool:
        """
        Check if the assessment is completed.
        
        Returns:
            True if completed, False otherwise
        """
        # Default implementation based on angle stability
        if len(self.data.history) < 30:
            return False
        
        # Check if angle has stabilized (indicating max ROM)
        recent_angles = [angle.value for angle in self.data.history[-10:]]
        
        if max(recent_angles) - min(recent_angles) < 3:
            return True
        
        return False
    
    def _update_landmarks(self, landmarks: Dict[int, Tuple[float, float, float]]):
        """
        Update ROM data with landmark positions.
        
        Args:
            landmarks: Dictionary of landmark coordinates
        """
        # Override in subclasses to store relevant landmarks
        pass
    
    @abstractmethod
    def calculate_angles(self, landmarks: Dict[int, Tuple[float, float, float]]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Calculate angles from landmarks.
        
        Args:
            landmarks: Dictionary of landmark coordinates
            
        Returns:
            Tuple of (joint_angles, segment_angles) dictionaries
        """
        pass
    
    @abstractmethod
    def check_position(self, landmarks: Dict[int, Tuple[float, float, float]], frame_shape: Tuple[int, int, int]) -> Tuple[bool, str]:
        """
        Check if the person is in the correct position.
        
        Args:
            landmarks: Dictionary of landmark coordinates
            frame_shape: Shape of the frame (height, width, channels)
            
        Returns:
            Tuple of (is_position_valid, guidance_message)
        """
        pass
    
    @abstractmethod
    def visualize_assessment(self, frame: np.ndarray, rom_data: ROMData) -> np.ndarray:
        """
        Visualize the assessment on the frame.
        
        Args:
            frame: Input video frame
            rom_data: Current ROM data
            
        Returns:
            Frame with visualization
        """
        pass
    
    def reset(self) -> None:
        """Reset the test to initial state."""
        self.data = self._initialize_rom_data()
        self.ready_time = 0
        self.is_ready = False
        self.start_time = None
        self.end_time = None
        self.frame_count = 0
        self.data_processor.reset()
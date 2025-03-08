# rom/core/base.py
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, List, Optional
import numpy as np
import cv2
from dataclasses import dataclass, field
from enum import Enum


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


class ROMTest(ABC):
    """Base class for ROM tests."""

    def __init__(self, pose_detector, visualizer, config: Dict[str, Any] = None):
        """
        Initialize a ROM test.
        
        Args:
            pose_detector: The pose detection system
            visualizer: The visualization system
            config: Configuration parameters
        """
        self.pose_detector = pose_detector
        self.visualizer = visualizer
        self.config = config or {}
        self.data = self._initialize_rom_data()

    @abstractmethod
    def _initialize_rom_data(self) -> ROMData:
        """Initialize ROM data specific to this test."""
        pass

    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process a single frame and return the processed frame and ROM data.
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (processed_frame, rom_data_dict)
        """
        pass

    @abstractmethod
    def calculate_angles(self, landmarks: Dict[int, Tuple[float, float, float]]) -> Dict[str, Angle]:
        """
        Calculate angles from landmarks.
        
        Args:
            landmarks: Dictionary of landmark coordinates
            
        Returns:
            Dictionary of calculated angles
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


class MathUtils:
    """Utility class for mathematical calculations used in ROM tests."""

    @staticmethod
    def calculate_angle(p1: Tuple[float, float, float], 
                        p2: Tuple[float, float, float], 
                        p3: Tuple[float, float, float]) -> float:
        """
        Calculate the angle between three points with p2 as the vertex.
        
        Args:
            p1, p2, p3: 3D points (x, y, z)
            
        Returns:
            Angle in degrees
        """
        # Convert to numpy arrays and take only x, y coordinates
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        # Normalize vectors
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm == 0 or v2_norm == 0:
            return 0.0
        
        v1_normalized = v1 / v1_norm
        v2_normalized = v2 / v2_norm
        
        # Calculate dot product and clip to avoid numerical errors
        dot_product = np.clip(np.dot(v1_normalized, v2_normalized), -1.0, 1.0)
        
        # Return angle in degrees
        return np.degrees(np.arccos(dot_product))

    @staticmethod
    def calculate_distance(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
        """
        Calculate the Euclidean distance between two points.
        
        Args:
            p1, p2: 3D points (x, y, z)
            
        Returns:
            Distance
        """
        return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)

    @staticmethod
    def smooth_angle(angle_history: List[float], window_size: int = 5) -> float:
        """
        Apply smoothing to angle measurements.
        
        Args:
            angle_history: List of angle measurements
            window_size: Size of the smoothing window
            
        Returns:
            Smoothed angle
        """
        if not angle_history:
            return 0.0
        
        if len(angle_history) < window_size:
            return angle_history[-1]
        
        # Apply simple moving average
        return sum(angle_history[-window_size:]) / window_size
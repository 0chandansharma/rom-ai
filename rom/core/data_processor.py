# rom/core/data_processor.py
import numpy as np
from typing import Dict, Tuple, List, Union, Optional, Any
import time
from collections import deque

from rom.utils.math_utils import MathUtils

class AngleData:
    """Class to store and process angle data for ROM assessment."""
    
    def __init__(self, max_history: int = 100):
        """
        Initialize angle data storage.
        
        Args:
            max_history: Maximum number of angles to store in history
        """
        self.angle_history = deque(maxlen=max_history)
        self.timestamp_history = deque(maxlen=max_history)
        self.min_angle = float('inf')
        self.max_angle = float('-inf')
        self.current_angle = None
        self.rom = 0
        self.start_time = None
        self.last_update_time = None
    
    def add_angle(self, angle: float):
        """
        Add an angle measurement to the history.
        
        Args:
            angle: Angle value in degrees
        """
        if self.start_time is None:
            self.start_time = time.time()
        
        self.last_update_time = time.time()
        self.current_angle = angle
        
        if not np.isnan(angle):
            self.angle_history.append(angle)
            self.timestamp_history.append(time.time())
            
            # Update min/max angles
            self.min_angle = min(self.min_angle, angle) if self.min_angle != float('inf') else angle
            self.max_angle = max(self.max_angle, angle) if self.max_angle != float('-inf') else angle
            
            # Update ROM
            self.rom = self.max_angle - self.min_angle
    
    def get_smoothed_angle(self, window_size: int = 5) -> float:
        """
        Get smoothed current angle.
        
        Args:
            window_size: Smoothing window size
            
        Returns:
            Smoothed angle value
        """
        return MathUtils.smooth_angle(list(self.angle_history), window_size)
    
    def get_velocity(self, window_size: int = 5) -> float:
        """
        Calculate angular velocity.
        
        Args:
            window_size: Window size for velocity calculation
            
        Returns:
            Angular velocity in degrees per second
        """
        if len(self.angle_history) < 2 or len(self.timestamp_history) < 2:
            return 0.0
        
        # Take last window_size points
        angles = list(self.angle_history)[-window_size:]
        timestamps = list(self.timestamp_history)[-window_size:]
        
        if len(angles) < 2:
            return 0.0
        
        # Simple finite difference
        delta_angle = angles[-1] - angles[0]
        delta_time = timestamps[-1] - timestamps[0]
        
        if delta_time == 0:
            return 0.0
            
        return delta_angle / delta_time
    
    def is_stable(self, threshold: float = 2.0, window_size: int = 5) -> bool:
        """
        Check if angle is stable (not changing significantly).
        
        Args:
            threshold: Maximum allowed variation in degrees
            window_size: Window size for stability check
            
        Returns:
            True if angle is stable, False otherwise
        """
        if len(self.angle_history) < window_size:
            return False
        
        # Get last window_size angles
        recent_angles = list(self.angle_history)[-window_size:]
        
        # Check if max variation is below threshold
        angle_range = max(recent_angles) - min(recent_angles)
        return angle_range < threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert angle data to dictionary.
        
        Returns:
            Dictionary with angle data
        """
        return {
            "current_angle": self.current_angle,
            "min_angle": self.min_angle if self.min_angle != float('inf') else None,
            "max_angle": self.max_angle if self.max_angle != float('-inf') else None,
            "rom": self.rom,
            "history_length": len(self.angle_history),
            "is_stable": self.is_stable(),
            "duration": time.time() - self.start_time if self.start_time else None
        }
    
    def reset(self):
        """Reset all angle data."""
        self.angle_history.clear()
        self.timestamp_history.clear()
        self.min_angle = float('inf')
        self.max_angle = float('-inf')
        self.current_angle = None
        self.rom = 0
        self.start_time = None
        self.last_update_time = None


class DataProcessor:
    """Process and analyze pose data for ROM assessment."""
    
    def __init__(self, 
                filter_type: str = 'butterworth', 
                interpolate: bool = True,
                max_gap_size: int = 10,
                smoothing_window: int = 5,
                butterworth_cutoff: float = 6.0,
                butterworth_order: int = 4,
                fps: float = 30.0):
        """
        Initialize data processor.
        
        Args:
            filter_type: 'butterworth', 'moving_average', or 'none'
            interpolate: Whether to interpolate missing data
            max_gap_size: Maximum gap size for interpolation
            smoothing_window: Window size for smoothing
            butterworth_cutoff: Cutoff frequency for Butterworth filter
            butterworth_order: Order for Butterworth filter
            fps: Frame rate for filtering calculations
        """
        self.filter_type = filter_type
        self.interpolate = interpolate
        self.max_gap_size = max_gap_size
        self.smoothing_window = smoothing_window
        self.butterworth_cutoff = butterworth_cutoff
        self.butterworth_order = butterworth_order
        self.fps = fps
        
        # Store angle data for different angles
        self.angle_data = {}
    
    def add_angle(self, angle_name: str, angle_value: float):
        """
        Add an angle measurement.
        
        Args:
            angle_name: Name of the angle
            angle_value: Angle value in degrees
        """
        if angle_name not in self.angle_data:
            self.angle_data[angle_name] = AngleData()
        
        self.angle_data[angle_name].add_angle(angle_value)
    
    def get_processed_angle(self, angle_name: str) -> float:
        """
        Get processed (smoothed/filtered) angle value.
        
        Args:
            angle_name: Name of the angle
            
        Returns:
            Processed angle value
        """
        if angle_name not in self.angle_data:
            return float('nan')
        
        if self.filter_type == 'moving_average':
            return self.angle_data[angle_name].get_smoothed_angle(self.smoothing_window)
        elif self.filter_type == 'butterworth':
            # Apply Butterworth filter to history
            history = list(self.angle_data[angle_name].angle_history)
            if len(history) < self.butterworth_order * 2:
                return self.angle_data[angle_name].current_angle
            
            filtered_history = MathUtils.filter_butterworth(
                history, 
                self.butterworth_cutoff, 
                self.fps, 
                self.butterworth_order
            )
            
            return filtered_history[-1] if filtered_history else float('nan')
        else:
            return self.angle_data[angle_name].current_angle
    
    def get_all_processed_angles(self) -> Dict[str, float]:
        """
        Get all processed angles.
        
        Returns:
            Dictionary mapping angle names to processed values
        """
        return {name: self.get_processed_angle(name) for name in self.angle_data}
    
    def get_angle_data(self, angle_name: str) -> Dict[str, Any]:
        """
        Get angle data for a specific angle.
        
        Args:
            angle_name: Name of the angle
            
        Returns:
            Dictionary with angle data
        """
        if angle_name not in self.angle_data:
            return {}
        
        return self.angle_data[angle_name].to_dict()
    
    def get_all_angle_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Get data for all angles.
        
        Returns:
            Dictionary mapping angle names to angle data
        """
        return {name: self.get_angle_data(name) for name in self.angle_data}
    
    def reset(self, angle_name: Optional[str] = None):
        """
        Reset angle data.
        
        Args:
            angle_name: Name of angle to reset (or all if None)
        """
        if angle_name:
            if angle_name in self.angle_data:
                self.angle_data[angle_name].reset()
        else:
            for data in self.angle_data.values():
                data.reset()
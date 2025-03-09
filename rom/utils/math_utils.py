# rom/utils/math_utils.py
import numpy as np
from typing import Dict, Tuple, List, Union, Optional
import math
from collections import deque

class MathUtils:
    """Enhanced mathematical utilities for ROM assessment."""
    
    @staticmethod
    def calculate_angle(p1: Tuple[float, float, float], 
                        p2: Tuple[float, float, float], 
                        p3: Tuple[float, float, float],
                        angle_type: str = 'inner') -> float:
        """
        Calculate angle between three points with p2 as the vertex.
        
        Args:
            p1, p2, p3: Points as (x, y, z) tuples
            angle_type: 'inner' for inner angle, 'outer' for outer angle
            
        Returns:
            Angle in degrees
        """
        # Convert to numpy arrays and take only x, y coordinates for 2D angle
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
        
        # Calculate angle
        angle = np.degrees(np.arccos(dot_product))
        
        # Return inner or outer angle as requested
        if angle_type == 'outer':
            return 360 - angle
        else:
            return angle

    @staticmethod
    def calculate_segment_angle(p1: Tuple[float, float, float], 
                               p2: Tuple[float, float, float],
                               reference: str = 'horizontal') -> float:
        """
        Calculate segment angle with respect to reference.
        
        Args:
            p1, p2: Points defining the segment
            reference: 'horizontal' or 'vertical'
            
        Returns:
            Angle in degrees
        """
        # Calculate segment vector
        segment = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        
        # Calculate angle with reference
        if reference == 'horizontal':
            reference_vector = np.array([1, 0])
        else:  # vertical
            reference_vector = np.array([0, 1])
        
        # Normalize segment
        segment_norm = np.linalg.norm(segment)
        if segment_norm == 0:
            return 0.0
        
        segment_normalized = segment / segment_norm
        
        # Calculate dot product and angle
        dot_product = np.clip(np.dot(segment_normalized, reference_vector), -1.0, 1.0)
        angle = np.degrees(np.arccos(dot_product))
        
        # Determine direction (clockwise or counterclockwise)
        cross_product = np.cross([segment_normalized[0], segment_normalized[1], 0], 
                                [reference_vector[0], reference_vector[1], 0])
        if cross_product[2] < 0:
            angle = 360 - angle
            
        return angle

    @staticmethod
    def calculate_joint_angle(landmarks: Dict[int, Tuple[float, float, float]], 
                             joint_points: Tuple[int, int, int]) -> float:
        """
        Calculate joint angle from landmarks.
        
        Args:
            landmarks: Dictionary of landmark coordinates
            joint_points: Tuple of (p1, joint, p2) indices
            
        Returns:
            Angle in degrees
        """
        p1_idx, joint_idx, p2_idx = joint_points
        
        if all(idx in landmarks for idx in joint_points):
            return MathUtils.calculate_angle(
                landmarks[p1_idx],
                landmarks[joint_idx],
                landmarks[p2_idx]
            )
        else:
            return float('nan')

    @staticmethod
    def calculate_distance(p1: Tuple[float, float, float], 
                          p2: Tuple[float, float, float]) -> float:
        """
        Calculate Euclidean distance between two points.
        
        Args:
            p1, p2: Points as (x, y, z) tuples
            
        Returns:
            Distance
        """
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

    @staticmethod
    def get_midpoint(p1: Tuple[float, float, float], 
                    p2: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Calculate midpoint between two points.
        
        Args:
            p1, p2: Points as (x, y, z) tuples
            
        Returns:
            Midpoint as (x, y, z) tuple
        """
        return tuple((a + b) / 2 for a, b in zip(p1, p2))

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
        
        valid_angles = [a for a in angle_history if not math.isnan(a)]
        
        if not valid_angles:
            return float('nan')
        
        if len(valid_angles) < window_size:
            return valid_angles[-1]
        
        # Apply simple moving average
        return sum(valid_angles[-window_size:]) / window_size

    @staticmethod
    def interpolate_missing(data: List[float], max_gap: int = 5) -> List[float]:
        """
        Interpolate missing values in a time series.
        
        Args:
            data: List of data points (can contain NaN)
            max_gap: Maximum gap size to interpolate
            
        Returns:
            Interpolated data
        """
        # Convert to numpy array for easier manipulation
        data_array = np.array(data)
        mask = np.isnan(data_array)
        
        # Find sequences of NaNs
        mask_indices = np.where(mask)[0]
        if len(mask_indices) == 0:
            return data
        
        # Find gaps (consecutive NaN indices)
        consecutive_groups = []
        current_group = [mask_indices[0]]
        
        for i in range(1, len(mask_indices)):
            if mask_indices[i] == mask_indices[i-1] + 1:
                current_group.append(mask_indices[i])
            else:
                consecutive_groups.append(current_group)
                current_group = [mask_indices[i]]
        
        if current_group:
            consecutive_groups.append(current_group)
        
        # Interpolate gaps smaller than max_gap
        result = data_array.copy()
        
        for group in consecutive_groups:
            if len(group) <= max_gap:
                # Get values before and after gap
                before_idx = group[0] - 1
                after_idx = group[-1] + 1
                
                if before_idx >= 0 and after_idx < len(data_array):
                    before_val = data_array[before_idx]
                    after_val = data_array[after_idx]
                    
                    if not np.isnan(before_val) and not np.isnan(after_val):
                        # Linear interpolation
                        for i, idx in enumerate(group):
                            alpha = (i + 1) / (len(group) + 1)
                            result[idx] = before_val + alpha * (after_val - before_val)
        
        return result.tolist()

    @staticmethod
    def filter_butterworth(data: List[float], cutoff: float, fs: float, order: int = 4) -> List[float]:
        """
        Apply Butterworth low-pass filter to data.
        
        Args:
            data: List of data points
            cutoff: Cutoff frequency
            fs: Sampling frequency
            order: Filter order
            
        Returns:
            Filtered data
        """
        try:
            from scipy import signal
            
            # Remove NaN values for filtering
            data_array = np.array(data)
            nan_mask = np.isnan(data_array)
            data_no_nan = data_array[~nan_mask]
            
            if len(data_no_nan) < order + 1:
                return data
            
            # Design filter
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
            
            # Apply filter
            filtered_data = signal.filtfilt(b, a, data_no_nan)
            
            # Reconstruct data with NaNs
            result = data_array.copy()
            result[~nan_mask] = filtered_data
            
            return result.tolist()
        except ImportError:
            print("SciPy not available, returning unfiltered data")
            return data
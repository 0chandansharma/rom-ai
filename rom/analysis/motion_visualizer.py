# rom/analysis/motion_visualizer.py
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io
from typing import Dict, List, Tuple, Any, Optional

class MotionVisualizer:
    """Visualize motion and angle trajectories in real-time."""
    
    def __init__(self, width: int = 800, height: int = 150, history_length: int = 100):
        """
        Initialize motion visualizer.
        
        Args:
            width: Width of the visualization area
            height: Height of the visualization area
            history_length: Number of frames to keep in history
        """
        self.width = width
        self.height = height
        self.history_length = history_length
        self.angle_histories = {}
        self.color_map = {
            0: (0, 0, 255),    # Red
            1: (0, 255, 0),    # Green
            2: (255, 0, 0),    # Blue
            3: (0, 255, 255),  # Yellow
            4: (255, 0, 255),  # Magenta
            5: (255, 255, 0)   # Cyan
        }
    
    def update_angle(self, angle_name: str, angle_value: float) -> None:
        """
        Update angle history with new value.
        
        Args:
            angle_name: Name of the angle
            angle_value: New angle value
        """
        if angle_name not in self.angle_histories:
            self.angle_histories[angle_name] = []
        
        history = self.angle_histories[angle_name]
        history.append(angle_value)
        
        # Trim to keep only the last N values
        if len(history) > self.history_length:
            self.angle_histories[angle_name] = history[-self.history_length:]
    
    def create_trajectory_visualization(self, 
                                      selected_angles: Optional[List[str]] = None, 
                                      min_y: Optional[float] = None,
                                      max_y: Optional[float] = None,
                                      show_grid: bool = True) -> np.ndarray:
        """
        Create visualization of angle trajectories.
        
        Args:
            selected_angles: List of angle names to visualize (all if None)
            min_y: Minimum y-axis value
            max_y: Maximum y-axis value
            show_grid: Whether to show grid lines
            
        Returns:
            Visualization image
        """
        # Create blank image
        vis_img = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        
        # Draw background
        if show_grid:
            # Draw horizontal grid lines
            for i in range(1, 10):
                y = int(self.height * i / 10)
                cv2.line(vis_img, (0, y), (self.width, y), (220, 220, 220), 1)
            
            # Draw vertical grid lines
            for i in range(1, 10):
                x = int(self.width * i / 10)
                cv2.line(vis_img, (x, 0), (x, self.height), (220, 220, 220), 1)
        
        # Determine angles to plot
        if selected_angles:
            angles_to_plot = [name for name in selected_angles if name in self.angle_histories]
        else:
            angles_to_plot = list(self.angle_histories.keys())
        
        if not angles_to_plot:
            return vis_img
        
        # Determine y-axis limits
        if min_y is None or max_y is None:
            all_values = []
            for name in angles_to_plot:
                all_values.extend(self.angle_histories[name])
            
            if not all_values:
                return vis_img
                
            min_val = min(all_values)
            max_val = max(all_values)
            
            # Add padding
            range_val = max_val - min_val
            if range_val < 10:
                range_val = 10
            
            if min_y is None:
                min_y = min_val - range_val * 0.1
            
            if max_y is None:
                max_y = max_val + range_val * 0.1
        
        # Draw trajectories
        for i, angle_name in enumerate(angles_to_plot):
            color = self.color_map.get(i % len(self.color_map), (0, 0, 0))
            values = self.angle_histories[angle_name]
            
            if len(values) < 2:
                continue
            
            # Scale to image coordinates
            y_scale = self.height / (max_y - min_y)
            x_scale = self.width / (self.history_length - 1)
            
            points = []
            for j, val in enumerate(values):
                x = int(j * x_scale)
                y = int(self.height - (val - min_y) * y_scale)
                points.append((x, y))
            
            # Draw polyline
            cv2.polylines(vis_img, [np.array(points)], False, color, 2)
            
            # Draw current value
            last_x, last_y = points[-1]
            cv2.circle(vis_img, (last_x, last_y), 4, color, -1)
            
            # Add label
            short_name = angle_name[-15:] if len(angle_name) > 15 else angle_name
            cv2.putText(
                vis_img,
                f"{short_name}: {values[-1]:.1f}°",
                (last_x + 5, last_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1
            )
        
        # Draw axes labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(vis_img, f"{min_y:.0f}°", (5, self.height - 5), font, 0.4, (0, 0, 0), 1)
        cv2.putText(vis_img, f"{max_y:.0f}°", (5, 15), font, 0.4, (0, 0, 0), 1)
        
        return vis_img
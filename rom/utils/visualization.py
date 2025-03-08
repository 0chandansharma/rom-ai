# rom/utils/visualization.py
import cv2
import numpy as np
from typing import Dict, Tuple, List, Any, Optional
import math


class PoseVisualizer:
    """Enhanced visualization utilities for ROM assessment."""
    
    def __init__(self, theme="dark"):
        """
        Initialize visualizer with color theme.
        
        Args:
            theme: Color theme ("dark" or "light")
        """
        self.theme = theme
        self._set_color_theme(theme)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
    def _set_color_theme(self, theme: str) -> None:
        """
        Set color theme for visualization.
        
        Args:
            theme: Color theme ("dark" or "light")
        """
        # Basic colors (BGR format)
        if theme == "dark":
            self.colors = {
                'background': (40, 44, 52),
                'primary': (0, 165, 255),    # Orange
                'secondary': (255, 196, 0),  # Cyan
                'success': (0, 222, 60),     # Green
                'warning': (0, 170, 255),    # Yellow
                'danger': (48, 59, 255),     # Red
                'info': (255, 191, 0),       # Light blue
                'white': (255, 255, 255),
                'black': (0, 0, 0),
                'gray': (128, 128, 128),
                'light_gray': (192, 192, 192),
                'dark_gray': (64, 64, 64),
                'blue': (255, 0, 0),
                'green': (0, 255, 0),
                'red': (0, 0, 255),
                'yellow': (0, 255, 255),
                'magenta': (255, 0, 255),
                'cyan': (255, 255, 0),
                'purple': (128, 0, 128),
                'overlay': (0, 0, 0, 0.7)    # Black with alpha for overlays
            }
        else:  # light theme
            self.colors = {
                'background': (248, 249, 250),
                'primary': (0, 165, 255),    # Orange
                'secondary': (255, 196, 0),  # Cyan
                'success': (0, 200, 60),     # Green
                'warning': (0, 170, 255),    # Yellow
                'danger': (48, 59, 255),     # Red
                'info': (209, 140, 0),       # Darker blue
                'white': (255, 255, 255),
                'black': (0, 0, 0),
                'gray': (128, 128, 128),
                'light_gray': (192, 192, 192),
                'dark_gray': (64, 64, 64),
                'blue': (255, 0, 0),
                'green': (0, 255, 0),
                'red': (0, 0, 255),
                'yellow': (0, 255, 255),
                'magenta': (255, 0, 255),
                'cyan': (255, 255, 0),
                'purple': (128, 0, 128),
                'overlay': (255, 255, 255, 0.7)  # White with alpha for overlays
            }
    
    def draw_landmark_point(self, frame: np.ndarray, x: int, y: int, 
                           color: str = 'primary', size: int = 8) -> None:
        """
        Draw a landmark point on the frame.
        
        Args:
            frame: Input video frame
            x, y: Coordinates
            color: Color name from theme
            size: Size of the marker
        """
        if color in self.colors:
            color_value = self.colors[color]
        else:
            color_value = self.colors['primary']
            
        cv2.circle(frame, (x, y), size, color_value, -1)  # Filled circle
        cv2.circle(frame, (x, y), size+2, self.colors['white'], 2)  # Border
    
    def draw_connection(self, frame: np.ndarray, start_point: Tuple[int, int], 
                        end_point: Tuple[int, int], color: str = 'white', 
                        thickness: int = 2) -> None:
        """
        Draw a connection line between two points.
        
        Args:
            frame: Input video frame
            start_point: Starting point coordinates
            end_point: Ending point coordinates
            color: Color name from theme
            thickness: Line thickness
        """
        if color in self.colors:
            color_value = self.colors[color]
        else:
            color_value = self.colors['white']
            
        cv2.line(frame, start_point, end_point, color_value, thickness)
    
    def put_text(self, frame: np.ndarray, text: str, position: Tuple[int, int], 
                color: str = 'white', scale: float = 0.8, thickness: int = 2) -> None:
        """
        Put text on the frame with better readability.
        
        Args:
            frame: Input video frame
            text: Text to display
            position: (x, y) position for text
            color: Color name from theme
            scale: Font scale
            thickness: Text thickness
        """
        if color in self.colors:
            color_value = self.colors[color]
        else:
            color_value = self.colors['white']
        
        # Add a dark background for better readability
        (text_width, text_height), baseline = cv2.getTextSize(text, self.font, scale, thickness)
        cv2.rectangle(
            frame, 
            (position[0] - 5, position[1] - text_height - 5), 
            (position[0] + text_width + 5, position[1] + 5), 
            self.colors['black'], 
            -1
        )
        
        # Draw the text
        cv2.putText(
            frame, 
            text, 
            position, 
            self.font, 
            scale, 
            color_value, 
            thickness, 
            cv2.LINE_AA
        )
    
    def draw_angle(self, frame: np.ndarray, p1: Tuple[float, float], p2: Tuple[float, float], 
                  p3: Tuple[float, float], angle_value: float, color: str = 'info',
                  thickness: int = 2, radius: int = 30) -> None:
        """
        Draw an angle between three points with visualization of the angle.
        
        Args:
            frame: Input video frame
            p1, p2, p3: Three points forming the angle (p2 is the vertex)
            angle_value: Angle value in degrees
            color: Color name from theme
            thickness: Line thickness
            radius: Radius of the angle arc
        """
        if color in self.colors:
            color_value = self.colors[color]
        else:
            color_value = self.colors['info']
        
        # Draw the lines connecting the points
        x1, y1 = int(p1[0]), int(p1[1])
        x2, y2 = int(p2[0]), int(p2[1])
        x3, y3 = int(p3[0]), int(p3[1])
        
        self.draw_connection(frame, (x1, y1), (x2, y2), color, thickness)
        self.draw_connection(frame, (x2, y2), (x3, y3), color, thickness)
        
        # Calculate the angle for drawing the arc
        v1 = np.array([x1 - x2, y1 - y2])
        v2 = np.array([x3 - x2, y3 - y2])
        
        # Normalize vectors
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm == 0 or v2_norm == 0:
            return
        
        # Calculate start and end angles for arc
        start_angle = math.atan2(v1[1], v1[0])
        end_angle = math.atan2(v2[1], v2[0])
        
        # Ensure the arc takes the shorter path
        if abs(end_angle - start_angle) > math.pi:
            if end_angle > start_angle:
                start_angle += 2 * math.pi
            else:
                end_angle += 2 * math.pi
        
        # Draw arc representing the angle
        cv2.ellipse(
            frame, 
            (x2, y2), 
            (radius, radius), 
            0, 
            math.degrees(start_angle), 
            math.degrees(end_angle), 
            color_value, 
            2, 
            cv2.LINE_AA
        )
        
        # Draw angle text
        text_position = (
            int(x2 + radius * 1.5 * math.cos((start_angle + end_angle) / 2)),
            int(y2 + radius * 1.5 * math.sin((start_angle + end_angle) / 2))
        )
        
        self.put_text(
            frame, 
            f"{angle_value:.1f}°", 
            text_position, 
            color, 
            0.7, 
            2
        )
    
    def draw_rom_gauge(self, frame: np.ndarray, current_rom: float, target_rom: float, 
                      position: Tuple[int, int], width: int = 200, height: int = 30) -> None:
        """
        Draw a gauge showing ROM progress toward a target.
        
        Args:
            frame: Input video frame
            current_rom: Current ROM measurement
            target_rom: Target ROM to achieve
            position: (x, y) position for the gauge
            width: Width of the gauge
            height: Height of the gauge
        """
        x, y = position
        
        # Calculate progress percentage (capped at 100%)
        progress = min(current_rom / target_rom * 100, 100) if target_rom > 0 else 0
        progress_width = int((progress / 100) * width)
        
        # Draw background
        cv2.rectangle(frame, (x, y), (x + width, y + height), self.colors['dark_gray'], -1)
        
        # Draw progress
        if progress <= 33:
            color = self.colors['danger']
        elif progress <= 66:
            color = self.colors['warning']
        else:
            color = self.colors['success']
            
        cv2.rectangle(frame, (x, y), (x + progress_width, y + height), color, -1)
        
        # Draw border
        cv2.rectangle(frame, (x, y), (x + width, y + height), self.colors['white'], 2)
        
        # Draw text
        text = f"{current_rom:.1f}° / {target_rom:.1f}°"
        text_size, _ = cv2.getTextSize(text, self.font, 0.6, 2)
        text_x = x + (width - text_size[0]) // 2
        text_y = y + height + 20
        
        self.put_text(frame, text, (text_x, text_y), 'white', 0.6, 2)
    
    def draw_guidance_overlay(self, frame: np.ndarray, message: str) -> np.ndarray:
        """
        Draw a guidance overlay at the bottom of the frame.
        
        Args:
            frame: Input video frame
            message: Guidance message to display
            
        Returns:
            Frame with overlay
        """
        h, w, _ = frame.shape
        overlay_height = 100
        
        # Create a copy of the frame
        overlay = frame.copy()
        
        # Draw semi-transparent black rectangle at the bottom
        cv2.rectangle(overlay, (0, h - overlay_height), (w, h), self.colors['black'], -1)
        
        # Blend the overlay with the original frame
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Add the guidance message
        self.put_text(
            frame,
            message,
            (20, h - overlay_height // 2),
            'white',
            0.8,
            2
        )
        
        return frame
    
    def draw_assessment_info(self, frame: np.ndarray, 
                            status: str, 
                            test_type: str, 
                            current_angle: Optional[float] = None,
                            min_angle: Optional[float] = None,
                            max_angle: Optional[float] = None) -> np.ndarray:
        """
        Draw assessment information on the top left corner.
        
        Args:
            frame: Input video frame
            status: Assessment status
            test_type: Type of test being performed
            current_angle: Current angle measurement
            min_angle: Minimum angle recorded
            max_angle: Maximum angle recorded
            
        Returns:
            Frame with information
        """
        # Format test type for display
        display_test = test_type.replace('_', ' ').title()
        
        # Draw background panel
        h, w, _ = frame.shape
        panel_width = 300
        panel_height = 120
        panel_x = 20
        panel_y = 20
        
        # Draw semi-transparent panel
        overlay = frame.copy()
        cv2.rectangle(
            overlay, 
            (panel_x, panel_y), 
            (panel_x + panel_width, panel_y + panel_height), 
            self.colors['black'], 
            -1
        )
        
        # Blend the overlay with the original frame
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw panel border
        cv2.rectangle(
            frame, 
            (panel_x, panel_y), 
            (panel_x + panel_width, panel_y + panel_height), 
            self.colors['white'], 
            2
        )
        
        # Draw assessment info
        y_offset = panel_y + 30
        self.put_text(frame, f"Test: {display_test}", (panel_x + 10, y_offset), 'info', 0.7, 2)
        
        y_offset += 25
        status_color = 'success' if status == 'completed' else 'warning'
        self.put_text(
            frame, 
            f"Status: {status.replace('_', ' ').title()}", 
            (panel_x + 10, y_offset), 
            status_color, 
            0.7, 
            2
        )
        
        if current_angle is not None:
            y_offset += 25
            self.put_text(
                frame, 
                f"Current: {current_angle:.1f}°", 
                (panel_x + 10, y_offset), 
                'white', 
                0.7, 
                2
            )
        
        if min_angle is not None and max_angle is not None:
            y_offset += 25
            rom = max_angle - min_angle
            self.put_text(
                frame, 
                f"ROM: {rom:.1f}° ({min_angle:.1f}° - {max_angle:.1f}°)", 
                (panel_x + 10, y_offset), 
                'primary', 
                0.7, 
                2
            )
        
        return frame
    
    def draw_position_guide(self, frame: np.ndarray, 
                           is_correct_position: bool, 
                           message: str, 
                           ready_progress: float = 0) -> np.ndarray:
        """
        Draw position guidance with progress indicator.
        
        Args:
            frame: Input video frame
            is_correct_position: Whether the position is correct
            message: Guidance message
            ready_progress: Progress toward ready position (0-100)
            
        Returns:
            Frame with position guide
        """
        h, w, _ = frame.shape
        
        # Draw semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 100), (w, h), self.colors['black'], -1)
        
        # Blend the overlay with the original frame
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw position message
        color = 'success' if is_correct_position else 'warning'
        self.put_text(
            frame,
            f"Position: {message}",
            (20, h - 60),
            color,
            0.8,
            2
        )
        
        # Draw progress bar if in correct position
        if is_correct_position and ready_progress > 0:
            progress_width = int((ready_progress / 100) * (w - 40))
            
            # Draw background
            cv2.rectangle(frame, (20, h - 30), (w - 20, h - 10), self.colors['dark_gray'], -1)
            
            # Draw progress
            cv2.rectangle(frame, (20, h - 30), (20 + progress_width, h - 10), self.colors['success'], -1)
            
            # Draw border
            cv2.rectangle(frame, (20, h - 30), (w - 20, h - 10), self.colors['white'], 2)
            
        return frame
# rom/utils/visualization.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Any, Optional, Union
import time
import math
import io
from PIL import Image


class EnhancedVisualizer:
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
                           color: str = 'primary', size: int = 8,
                           highlight: bool = False) -> None:
        """
        Draw a landmark point on the frame.
        
        Args:
            frame: Input video frame
            x, y: Coordinates
            color: Color name from theme
            size: Size of the marker
            highlight: Whether to highlight the point
        """
        if color in self.colors:
            color_value = self.colors[color]
        else:
            color_value = self.colors['primary']
            
        cv2.circle(frame, (x, y), size, color_value, -1)  # Filled circle
        
        if highlight:
            # Draw a pulsing highlight effect
            pulse_size = size + 4 + int(2 * math.sin(time.time() * 5))
            cv2.circle(frame, (x, y), pulse_size, color_value, 2)
        else:
            cv2.circle(frame, (x, y), size+2, self.colors['white'], 2)  # Border
    
    def draw_connection(self, frame: np.ndarray, start_point: Tuple[int, int], 
                        end_point: Tuple[int, int], color: str = 'white', 
                        thickness: int = 2, style: str = 'solid') -> None:
        """
        Draw a connection line between two points.
        
        Args:
            frame: Input video frame
            start_point: Starting point coordinates
            end_point: Ending point coordinates
            color: Color name from theme
            thickness: Line thickness
            style: Line style ('solid', 'dashed', 'dotted')
        """
        if color in self.colors:
            color_value = self.colors[color]
        else:
            color_value = self.colors['white']
        
        if style == 'solid':
            cv2.line(frame, start_point, end_point, color_value, thickness)
        
        elif style == 'dashed':
            # Draw dashed line
            pt1 = np.array(start_point)
            pt2 = np.array(end_point)
            dist = np.linalg.norm(pt2 - pt1)
            
            if dist < 1:
                return
                
            pts = []
            gap = 7
            dash = 10
            
            if dist < dash:
                cv2.line(frame, start_point, end_point, color_value, thickness)
                return
                
            dist_covered = 0
            curr_pos = pt1
            direction = (pt2 - pt1) / dist
            
            while dist_covered < dist:
                # Draw dash
                if dist_covered + dash < dist:
                    new_pos = curr_pos + direction * dash
                    cv2.line(frame, tuple(curr_pos.astype(int)), tuple(new_pos.astype(int)), color_value, thickness)
                    curr_pos = new_pos
                    dist_covered += dash
                else:
                    cv2.line(frame, tuple(curr_pos.astype(int)), tuple(pt2.astype(int)), color_value, thickness)
                    break
                
                # Skip gap
                curr_pos = curr_pos + direction * gap
                dist_covered += gap
                
                if dist_covered >= dist:
                    break
        
        elif style == 'dotted':
            # Draw dotted line
            pt1 = np.array(start_point)
            pt2 = np.array(end_point)
            dist = np.linalg.norm(pt2 - pt1)
            
            if dist < 1:
                return
                
            num_dots = max(int(dist / 5), 2)
            for i in range(num_dots):
                alpha = i / (num_dots - 1)
                dot_pt = tuple((pt1 * (1 - alpha) + pt2 * alpha).astype(int))
                cv2.circle(frame, dot_pt, thickness, color_value, -1)
    
    def put_text(self, frame: np.ndarray, text: str, position: Tuple[int, int], 
                color: str = 'white', scale: float = 0.8, thickness: int = 2,
                background: bool = True, align: str = 'left') -> None:
        """
        Put text on the frame with better readability.
        
        Args:
            frame: Input video frame
            text: Text to display
            position: (x, y) position for text
            color: Color name from theme
            scale: Font scale
            thickness: Text thickness
            background: Whether to add background box
            align: Text alignment ('left', 'center', 'right')
        """
        if color in self.colors:
            color_value = self.colors[color]
        else:
            color_value = self.colors['white']
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, self.font, scale, thickness)
        
        # Adjust position based on alignment
        x, y = position
        if align == 'center':
            x = x - text_width // 2
        elif align == 'right':
            x = x - text_width
        
        # Add a dark background for better readability
        if background:
            cv2.rectangle(
                frame, 
                (x - 5, y - text_height - 5), 
                (x + text_width + 5, y + 5), 
                self.colors['black'], 
                -1
            )
        
        # Draw the text
        cv2.putText(
            frame, 
            text, 
            (x, y), 
            self.font, 
            scale, 
            color_value, 
            thickness, 
            cv2.LINE_AA
        )
    
    def draw_angle(self, frame: np.ndarray, p1: Tuple[float, float], p2: Tuple[float, float], 
                  p3: Tuple[float, float], angle_value: float, color: str = 'info',
                  thickness: int = 2, radius: int = 30, display_mode: str = 'all') -> None:
        """
        Draw an angle between three points with visualization of the angle.
        
        Args:
            frame: Input video frame
            p1, p2, p3: Three points forming the angle (p2 is the vertex)
            angle_value: Angle value in degrees
            color: Color name from theme
            thickness: Line thickness
            radius: Radius of the angle arc
            display_mode: Display mode ('all', 'arc', 'text', 'lines')
        """
        if color in self.colors:
            color_value = self.colors[color]
        else:
            color_value = self.colors['info']
        
        # Draw the lines connecting the points
        x1, y1 = int(p1[0]), int(p1[1])
        x2, y2 = int(p2[0]), int(p2[1])
        x3, y3 = int(p3[0]), int(p3[1])
        
        if display_mode in ['all', 'lines']:
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
        if display_mode in ['all', 'arc']:
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
        if display_mode in ['all', 'text']:
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
    
    def draw_segment_angle(self, frame: np.ndarray, p1: Tuple[float, float], p2: Tuple[float, float],
                          angle_value: float, reference: str = 'horizontal',
                          color: str = 'info', thickness: int = 2, length: int = 30) -> None:
        """
        Draw a segment angle with respect to reference.
        
        Args:
            frame: Input video frame
            p1, p2: Points defining the segment
            angle_value: Angle value in degrees
            reference: Reference direction ('horizontal' or 'vertical')
            color: Color name from theme
            thickness: Line thickness
            length: Length of the reference line
        """
        if color in self.colors:
            color_value = self.colors[color]
        else:
            color_value = self.colors['info']
        
        # Convert points to integers
        p1 = (int(p1[0]), int(p1[1]))
        p2 = (int(p2[0]), int(p2[1]))
        
        # Draw the segment
        cv2.line(frame, p1, p2, color_value, thickness)
        
        # Calculate midpoint for reference line
        mid_x = (p1[0] + p2[0]) // 2
        mid_y = (p1[1] + p2[1]) // 2
        
        # Draw reference line
        if reference == 'horizontal':
            ref_p1 = (mid_x - length, mid_y)
            ref_p2 = (mid_x + length, mid_y)
        else:  # vertical
            ref_p1 = (mid_x, mid_y - length)
            ref_p2 = (mid_x, mid_y + length)
        
        cv2.line(frame, ref_p1, ref_p2, self.colors['gray'], 1, cv2.LINE_DASHED)
        
        # Draw angle arc
        radius = 15
        segment_angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        ref_angle = 0 if reference == 'horizontal' else -math.pi/2
        
        # Calculate start and end angles for arc
        if segment_angle < ref_angle:
            start_angle = segment_angle
            end_angle = ref_angle
        else:
            start_angle = ref_angle
            end_angle = segment_angle
        
        cv2.ellipse(
            frame, 
            (mid_x, mid_y), 
            (radius, radius), 
            0, 
            math.degrees(start_angle), 
            math.degrees(end_angle), 
            color_value, 
            thickness
        )
        
        # Draw angle text
        text_x = mid_x + int(radius * 1.5 * math.cos((start_angle + end_angle) / 2))
        text_y = mid_y + int(radius * 1.5 * math.sin((start_angle + end_angle) / 2))
        
        self.put_text(
            frame, 
            f"{angle_value:.1f}°", 
            (text_x, text_y), 
            color, 
            0.6, 
            1
        )
    
    def draw_rom_gauge(self, frame: np.ndarray, current_rom: float, target_rom: float, 
                      position: Tuple[int, int], width: int = 200, height: int = 30,
                      color_scheme: str = 'gradient') -> None:
        """
        Draw a gauge showing ROM progress toward a target.
        
        Args:
            frame: Input video frame
            current_rom: Current ROM measurement
            target_rom: Target ROM to achieve
            position: (x, y) position for the gauge
            width: Width of the gauge
            height: Height of the gauge
            color_scheme: Color scheme ('gradient' or 'threshold')
        """
        x, y = position
        
        # Calculate progress percentage (capped at 100%)
        progress = min(current_rom / target_rom * 100, 100) if target_rom > 0 else 0
        progress_width = int((progress / 100) * width)
        
        # Draw background
        cv2.rectangle(frame, (x, y), (x + width, y + height), self.colors['dark_gray'], -1)
        
        # Draw progress based on color scheme
        if color_scheme == 'gradient':
            # Gradient from red to green based on progress
            if progress <= 33:
                r = 255
                g = int(255 * (progress / 33))
                b = 0
            elif progress <= 66:
                r = int(255 * (1 - (progress - 33) / 33))
                g = 255
                b = 0
            else:
                r = 0
                g = 255
                b = int(255 * (progress - 66) / 34)
            
            color = (b, g, r)  # BGR format
            
        else:  # threshold based
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
    
    def draw_guidance_overlay(self, frame: np.ndarray, message: str,
                             position: str = 'bottom') -> np.ndarray:
        """
        Draw a guidance overlay at the specified position.
        
        Args:
            frame: Input video frame
            message: Guidance message to display
            position: Position ('bottom', 'top', 'center')
            
        Returns:
            Frame with overlay
        """
        h, w, _ = frame.shape
        
        if position == 'bottom':
            overlay_height = 100
            y_start = h - overlay_height
            y_end = h
            text_y = h - overlay_height // 2
        elif position == 'top':
            overlay_height = 100
            y_start = 0
            y_end = overlay_height
            text_y = overlay_height // 2
        else:  # center
            overlay_height = 150
            y_start = (h - overlay_height) // 2
            y_end = y_start + overlay_height
            text_y = h // 2
        
        # Create a copy of the frame
        overlay = frame.copy()
        
        # Draw semi-transparent black rectangle
        cv2.rectangle(overlay, (0, y_start), (w, y_end), self.colors['black'], -1)
        
        # Blend the overlay with the original frame
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Add the guidance message
        self.put_text(
            frame,
            message,
            (20, text_y),
            'white',
            0.8,
            2,
            background=False
        )
        
        return frame
    
    def draw_assessment_info(self, frame: np.ndarray, 
                            status: str, 
                            test_type: str, 
                            current_angle: Optional[float] = None,
                            min_angle: Optional[float] = None,
                            max_angle: Optional[float] = None,
                            position: str = 'top_left') -> np.ndarray:
        """
        Draw assessment information panel.
        
        Args:
            frame: Input video frame
            status: Assessment status
            test_type: Type of test being performed
            current_angle: Current angle measurement
            min_angle: Minimum angle recorded
            max_angle: Maximum angle recorded
            position: Panel position ('top_left', 'top_right')
            
        Returns:
            Frame with information
        """
        # Format test type for display
        display_test = test_type.replace('_', ' ').title()
        
        # Draw background panel
        h, w, _ = frame.shape
        panel_width = 300
        panel_height = 120
        
        if position == 'top_left':
            panel_x = 20
            panel_y = 20
        else:  # top_right
            panel_x = w - panel_width - 20
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
        self.put_text(frame, f"Test: {display_test}", (panel_x + 10, y_offset), 'info', 0.7, 2, background=False)
        
        y_offset += 25
        status_color = 'success' if status == 'completed' else 'warning'
        self.put_text(
            frame, 
            f"Status: {status.replace('_', ' ').title()}", 
            (panel_x + 10, y_offset), 
            status_color, 
            0.7, 
            2,
            background=False
        )
        
        if current_angle is not None:
            y_offset += 25
            self.put_text(
                frame, 
                f"Current: {current_angle:.1f}°", 
                (panel_x + 10, y_offset), 
                'white', 
                0.7, 
                2,
                background=False
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
                2,
                background=False
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
            2,
            background=False
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
    
    def draw_selective_pose(self, frame: np.ndarray, 
                           landmarks: Dict[int, Tuple[float, float, float]],
                           relevant_parts: Optional[List[str]] = None,
                           keypoint_mapping: Optional[Dict[str, int]] = None,
                           connection_mapping: Optional[Dict[str, List[Tuple[str, str]]]] = None,
                           color_scheme: str = 'role',  # 'role', 'confidence', or 'uniform'
                           confidence_values: Optional[Dict[int, float]] = None) -> np.ndarray:
        """
        Draw selected pose landmarks and connections based on relevant parts.
        
        Args:
            frame: Input video frame
            landmarks: Dictionary of landmark coordinates
            relevant_parts: List of relevant body part names (if None, draw all)
            keypoint_mapping: Mapping of landmark names to indices
            connection_mapping: Mapping of body parts to connections
            color_scheme: Color scheme for landmarks and connections
            confidence_values: Dictionary of landmark indices to confidence values
            
        Returns:
            Frame with pose visualization
        """
        if not landmarks:
            return frame
        
        # Default keypoint mapping if not provided
        if keypoint_mapping is None:
            keypoint_mapping = {
                "nose": 0,
                "left_eye": 1,
                "right_eye": 2,
                "left_ear": 3,
                "right_ear": 4,
                "left_shoulder": 5,
                "right_shoulder": 6,
                "left_elbow": 7,
                "right_elbow": 8,
                "left_wrist": 9,
                "right_wrist": 10,
                "left_hip": 11,
                "right_hip": 12,
                "left_knee": 13,
                "right_knee": 14,
                "left_ankle": 15,
                "right_ankle": 16
            }
        
        # Default connection mapping if not provided
        if connection_mapping is None:
            connection_mapping = {
                "head": [("nose", "left_eye"), ("nose", "right_eye"), ("left_eye", "left_ear"), ("right_eye", "right_ear")],
                "torso": [("left_shoulder", "right_shoulder"), ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"), ("left_hip", "right_hip")],
                "left_arm": [("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist")],
                "right_arm": [("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist")],
                "left_leg": [("left_hip", "left_knee"), ("left_knee", "left_ankle")],
                "right_leg": [("right_hip", "right_knee"), ("right_knee", "right_ankle")]
            }
        
        # Determine which parts to draw
        parts_to_draw = relevant_parts if relevant_parts else list(connection_mapping.keys())
        
        # Determine colors based on scheme
        colors_by_part = {}
        if color_scheme == 'role':
            # Color by body part role
            colors_by_part = {
                "head": 'yellow',
                "torso": 'primary',
                "left_arm": 'green',
                "right_arm": 'red',
                "left_leg": 'blue',
                "right_leg": 'purple'
            }
        else:
            # Uniform color
            uniform_color = 'white' if color_scheme == 'uniform' else 'green'
            for part in connection_mapping:
                colors_by_part[part] = uniform_color
        
        # Draw connections first (behind points)
        for part in parts_to_draw:
            if part in connection_mapping:
                color = colors_by_part.get(part, 'white')
                for connection in connection_mapping[part]:
                    start_name, end_name = connection
                    if start_name in keypoint_mapping and end_name in keypoint_mapping:
                        start_idx = keypoint_mapping[start_name]
                        end_idx = keypoint_mapping[end_name]
                        if start_idx in landmarks and end_idx in landmarks:
                            start_point = (int(landmarks[start_idx][0]), int(landmarks[start_idx][1]))
                            end_point = (int(landmarks[end_idx][0]), int(landmarks[end_idx][1]))
                            self.draw_connection(frame, start_point, end_point, color)
        
        # Draw landmarks
        landmark_names_to_draw = set()
        for part in parts_to_draw:
            if part in connection_mapping:
                for connection in connection_mapping[part]:
                    landmark_names_to_draw.add(connection[0])
                    landmark_names_to_draw.add(connection[1])
        
        for landmark_name in landmark_names_to_draw:
            if landmark_name in keypoint_mapping:
                idx = keypoint_mapping[landmark_name]
                if idx in landmarks:
                    x, y = int(landmarks[idx][0]), int(landmarks[idx][1])
                    
                    # Determine color and size based on confidence if applicable
                    # Determine color and size based on confidence if applicable
                    color = 'white'
                    size = 8
                    if color_scheme == 'confidence' and confidence_values and idx in confidence_values:
                        conf = confidence_values[idx]
                        if conf > 0.75:
                            color = 'green'
                            size = 8
                        elif conf > 0.5:
                            color = 'yellow'
                            size = 7
                        else:
                            color = 'red'
                            size = 6
                    
                    # Draw the landmark
                    self.draw_landmark_point(frame, x, y, color, size)
        
        return frame
    
    def create_report_image(self, rom_data: Dict[str, Any], 
                          angles_history: Dict[str, List[float]],
                          include_plots: bool = True,
                          width: int = 800, 
                          height: int = 600) -> np.ndarray:
        """
        Create a report image with ROM data and plots.
        
        Args:
            rom_data: ROM data dictionary
            angles_history: Dictionary of angle histories
            include_plots: Whether to include plots
            width: Width of the report image
            height: Height of the report image
            
        Returns:
            Report image as numpy array
        """
        # Create blank image
        report_img = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Draw header
        cv2.rectangle(report_img, (0, 0), (width, 80), self.colors['primary'], -1)
        
        self.put_text(
            report_img,
            f"ROM Assessment Report: {rom_data.get('test_type', '').replace('_', ' ').title()}",
            (width // 2, 50),
            'white',
            1.2,
            3,
            align='center',
            background=False
        )
        
        # Draw ROM results
        y_offset = 120
        self.put_text(
            report_img,
            "Assessment Results",
            (30, y_offset),
            'primary',
            1.0,
            2,
            background=False
        )
        
        y_offset += 40
        joint_type = rom_data.get('joint_type', '')
        status = rom_data.get('status', '')
        rom_value = rom_data.get('rom', 0)
        min_angle = rom_data.get('min_angle', 0)
        max_angle = rom_data.get('max_angle', 0)
        duration = rom_data.get('metadata', {}).get('duration', 0)
        
        info_items = [
            f"Joint: {joint_type}",
            f"Status: {status.replace('_', ' ').title()}",
            f"Range of Motion: {rom_value:.1f}°",
            f"Minimum Angle: {min_angle:.1f}°",
            f"Maximum Angle: {max_angle:.1f}°"
        ]
        
        if duration:
            info_items.append(f"Duration: {duration:.1f} seconds")
        
        for item in info_items:
            self.put_text(
                report_img,
                item,
                (50, y_offset),
                'black',
                0.8,
                2,
                background=False
            )
            y_offset += 30
        
        # Draw plots if requested and data available
        if include_plots and angles_history:
            # Find primary angle history (typically has the most entries)
            primary_angle_name = max(angles_history, key=lambda k: len(angles_history[k]))
            primary_angle_data = angles_history[primary_angle_name]
            
            if primary_angle_data:
                y_offset += 20
                self.put_text(
                    report_img,
                    "Angle Trajectory",
                    (30, y_offset),
                    'primary',
                    1.0,
                    2,
                    background=False
                )
                
                y_offset += 40
                
                # Create plot using matplotlib
                plt.figure(figsize=(7, 3))
                plt.plot(primary_angle_data, 'b-', linewidth=2)
                plt.grid(True, alpha=0.3)
                plt.title(f"{primary_angle_name.replace('_', ' ').title()} Angle")
                plt.xlabel("Frame")
                plt.ylabel("Angle (°)")
                plt.tight_layout()
                
                # Convert matplotlib plot to image
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100)
                plt.close()
                buf.seek(0)
                
                # Read the image into OpenCV format
                plot_img = np.asarray(bytearray(buf.read()), dtype=np.uint8)
                plot_img = cv2.imdecode(plot_img, cv2.IMREAD_COLOR)
                
                # Resize if needed
                plot_height, plot_width = plot_img.shape[:2]
                max_plot_width = width - 100
                if plot_width > max_plot_width:
                    ratio = max_plot_width / plot_width
                    new_height = int(plot_height * ratio)
                    plot_img = cv2.resize(plot_img, (max_plot_width, new_height))
                
                # Insert plot into report
                x_offset = (width - plot_img.shape[1]) // 2
                y_plot = y_offset
                h_plot, w_plot, _ = plot_img.shape
                report_img[y_plot:y_plot+h_plot, x_offset:x_offset+w_plot] = plot_img
                
                y_offset += h_plot + 50
        
        # Draw footer
        cv2.rectangle(report_img, (0, height - 50), (width, height), self.colors['light_gray'], -1)
        self.put_text(
            report_img,
            f"Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}",
            (width // 2, height - 20),
            'black',
            0.7,
            1,
            align='center',
            background=False
        )
        
        return report_img
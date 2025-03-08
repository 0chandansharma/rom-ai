# rom/utils/pose_detector.py
import mediapipe as mp
import cv2
import numpy as np
from typing import Dict, Tuple, List, Optional, Any


class PoseDetector:
    """Enhanced pose detector using MediaPipe with additional features."""
    
    def __init__(self, 
                 static_image_mode=False, 
                 model_complexity=1, 
                 smooth_landmarks=True, 
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        """
        Initialize the pose detector with MediaPipe.
        
        Args:
            static_image_mode: Whether to treat input as static images (slower but more accurate)
            model_complexity: Model complexity (0, 1, or 2)
            smooth_landmarks: Whether to filter landmarks to reduce jitter
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize MediaPipe Pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Landmark indices for common body parts for reference
        self.LANDMARK_INDICES = {
            "nose": 0,
            "left_eye_inner": 1,
            "left_eye": 2,
            "left_eye_outer": 3,
            "right_eye_inner": 4,
            "right_eye": 5,
            "right_eye_outer": 6,
            "left_ear": 7,
            "right_ear": 8,
            "mouth_left": 9,
            "mouth_right": 10,
            "left_shoulder": 11,
            "right_shoulder": 12,
            "left_elbow": 13,
            "right_elbow": 14,
            "left_wrist": 15,
            "right_wrist": 16,
            "left_pinky": 17,
            "right_pinky": 18,
            "left_index": 19,
            "right_index": 20,
            "left_thumb": 21,
            "right_thumb": 22,
            "left_hip": 23,
            "right_hip": 24,
            "left_knee": 25,
            "right_knee": 26,
            "left_ankle": 27,
            "right_ankle": 28,
            "left_heel": 29,
            "right_heel": 30,
            "left_foot_index": 31,
            "right_foot_index": 32
        }

    def find_pose(self, frame: np.ndarray) -> Optional[Any]:
        """
        Process frame and find pose landmarks.
        
        Args:
            frame: Input video frame
            
        Returns:
            Pose landmarks or None if no pose detected
        """
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe
        results = self.pose.process(rgb_frame)
        
        return results.pose_landmarks

    def get_landmark_coordinates(self, frame: np.ndarray, landmarks: Any) -> Dict[int, Tuple[float, float, float]]:
        """
        Convert normalized landmarks to pixel coordinates.
        
        Args:
            frame: Input video frame
            landmarks: Pose landmarks from MediaPipe
            
        Returns:
            Dictionary of landmark indices to (x, y, z) coordinates
        """
        h, w, _ = frame.shape
        coordinates = {}
        
        if landmarks:
            for idx, landmark in enumerate(landmarks.landmark):
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                z = landmark.z
                visibility = landmark.visibility if hasattr(landmark, 'visibility') else 1.0
                
                # Only include landmarks with reasonable visibility
                if visibility > 0.5:
                    coordinates[idx] = (x, y, z)
                
        return coordinates
    
    def draw_pose_landmarks(self, frame: np.ndarray, landmarks: Any) -> np.ndarray:
        """
        Draw all pose landmarks on the frame.
        
        Args:
            frame: Input video frame
            landmarks: Pose landmarks from MediaPipe
            
        Returns:
            Frame with landmarks drawn
        """
        if landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        return frame
    
    def get_joint_angles(self, landmarks: Dict[int, Tuple[float, float, float]]) -> Dict[str, float]:
        """
        Calculate common joint angles from landmarks.
        
        Args:
            landmarks: Dictionary of landmark coordinates
            
        Returns:
            Dictionary of joint names to angles
        """
        angles = {}
        
        # Calculate some common joint angles
        
        # Elbow angles
        if all(idx in landmarks for idx in [11, 13, 15]):  # Left elbow
            shoulder, elbow, wrist = landmarks[11], landmarks[13], landmarks[15]
            angles["left_elbow"] = self._calculate_angle(shoulder, elbow, wrist)
            
        if all(idx in landmarks for idx in [12, 14, 16]):  # Right elbow
            shoulder, elbow, wrist = landmarks[12], landmarks[14], landmarks[16]
            angles["right_elbow"] = self._calculate_angle(shoulder, elbow, wrist)
        
        # Knee angles
        if all(idx in landmarks for idx in [23, 25, 27]):  # Left knee
            hip, knee, ankle = landmarks[23], landmarks[25], landmarks[27]
            angles["left_knee"] = self._calculate_angle(hip, knee, ankle)
            
        if all(idx in landmarks for idx in [24, 26, 28]):  # Right knee
            hip, knee, ankle = landmarks[24], landmarks[26], landmarks[28]
            angles["right_knee"] = self._calculate_angle(hip, knee, ankle)
        
        # Hip angles
        if all(idx in landmarks for idx in [11, 23, 25]):  # Left hip
            shoulder, hip, knee = landmarks[11], landmarks[23], landmarks[25]
            angles["left_hip"] = self._calculate_angle(shoulder, hip, knee)
            
        if all(idx in landmarks for idx in [12, 24, 26]):  # Right hip
            shoulder, hip, knee = landmarks[12], landmarks[24], landmarks[26]
            angles["right_hip"] = self._calculate_angle(shoulder, hip, knee)
        
        # Calculate back angle (angle between shoulders, mid-hip, and mid-knee)
        if all(idx in landmarks for idx in [11, 12, 23, 24, 25, 26]):
            shoulder_midpoint = (
                (landmarks[11][0] + landmarks[12][0]) / 2,
                (landmarks[11][1] + landmarks[12][1]) / 2,
                (landmarks[11][2] + landmarks[12][2]) / 2
            )
            hip_midpoint = (
                (landmarks[23][0] + landmarks[24][0]) / 2,
                (landmarks[23][1] + landmarks[24][1]) / 2,
                (landmarks[23][2] + landmarks[24][2]) / 2
            )
            knee_midpoint = (
                (landmarks[25][0] + landmarks[26][0]) / 2,
                (landmarks[25][1] + landmarks[26][1]) / 2,
                (landmarks[25][2] + landmarks[26][2]) / 2
            )
            angles["trunk"] = self._calculate_angle(shoulder_midpoint, hip_midpoint, knee_midpoint)
            
        return angles
    
    def _calculate_angle(self, p1: Tuple[float, float, float], 
                         p2: Tuple[float, float, float], 
                         p3: Tuple[float, float, float]) -> float:
        """
        Calculate the angle between three points with p2 as the vertex.
        
        Args:
            p1, p2, p3: 3D points (x, y, z)
            
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
        
        # Return angle in degrees
        return np.degrees(np.arccos(dot_product))
    
    def estimate_pose_stability(self, landmark_history: List[Dict[int, Tuple[float, float, float]]], 
                               key_points: List[int] = None) -> float:
        """
        Estimate the stability of the pose over time.
        
        Args:
            landmark_history: List of landmark dictionaries from previous frames
            key_points: List of landmark indices to consider (if None, use all common points)
            
        Returns:
            Stability score between 0 (unstable) and 1 (stable)
        """
        if not landmark_history or len(landmark_history) < 2:
            return 0.0
        
        if key_points is None:
            # Use common landmarks that are usually visible
            key_points = [
                11, 12,  # Shoulders
                23, 24,  # Hips
                13, 14,  # Elbows
                25, 26   # Knees
            ]
        
        # Calculate the average movement of key points between frames
        movements = []
        for i in range(1, len(landmark_history)):
            prev_landmarks = landmark_history[i-1]
            curr_landmarks = landmark_history[i]
            
            frame_movements = []
            for point_idx in key_points:
                if point_idx in prev_landmarks and point_idx in curr_landmarks:
                    prev_point = prev_landmarks[point_idx]
                    curr_point = curr_landmarks[point_idx]
                    
                    # Calculate Euclidean distance
                    distance = np.sqrt(
                        (prev_point[0] - curr_point[0])**2 + 
                        (prev_point[1] - curr_point[1])**2
                    )
                    frame_movements.append(distance)
            
            if frame_movements:
                movements.append(np.mean(frame_movements))
        
        if not movements:
            return 0.0
        
        # Calculate stability score (inversely proportional to movement)
        avg_movement = np.mean(movements)
        stability = np.exp(-avg_movement / 10.0)  # Exponential decay based on movement
        
        return min(max(stability, 0.0), 1.0)  # Clamp between 0 and 1
    
    def get_body_orientation(self, landmarks: Dict[int, Tuple[float, float, float]]) -> str:
        """
        Determine if the person is facing the camera, facing left, or facing right.
        
        Args:
            landmarks: Dictionary of landmark coordinates
            
        Returns:
            Orientation as string: "front", "left", "right", or "unknown"
        """
        if 11 not in landmarks or 12 not in landmarks:  # Need shoulders
            return "unknown"
        
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        
        # Check which shoulder is more visible (z coordinate closer to camera)
        shoulder_z_diff = left_shoulder[2] - right_shoulder[2]
        
        # Also check shoulder x distance to detect if person is turned
        shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
        
        if abs(shoulder_z_diff) < 0.1 and shoulder_width > 100:
            return "front"  # Facing camera
        elif shoulder_z_diff > 0.1:
            return "right"  # Facing right
        elif shoulder_z_diff < -0.1:
            return "left"   # Facing left
        else:
            return "unknown"
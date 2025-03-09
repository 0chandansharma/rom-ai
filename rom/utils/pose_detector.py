# rom/utils/pose_detector.py
import mediapipe as mp
import cv2
import numpy as np
from typing import Dict, Tuple, List, Optional, Any
import logging

logger = logging.getLogger("rom.pose_detector")

class PoseDetector:
    """Enhanced pose detector with Sports2D capabilities."""
    
    def __init__(self, 
                 model_type='HALPE_26',
                 det_frequency=4,
                 tracking_mode='sports2d',
                 keypoint_likelihood_threshold=0.3,
                 average_likelihood_threshold=0.5,
                 keypoint_number_threshold=0.3,
                 static_image_mode=False,
                 model_complexity=1):
        """
        Initialize the pose detector with enhanced features.
        
        Args:
            model_type: Type of pose model ('HALPE_26', 'COCO_133', 'COCO_17')
            det_frequency: Run detection every N frames (tracking in between)
            tracking_mode: 'sports2d' or 'deepsort'
            keypoint_likelihood_threshold: Minimum confidence for keypoints
            average_likelihood_threshold: Minimum average confidence for person
            keypoint_number_threshold: Minimum fraction of keypoints detected
            static_image_mode: Whether to treat input as static images
            model_complexity: Model complexity (0, 1, or 2)
        """
        self.model_type = model_type.upper()
        self.det_frequency = det_frequency
        self.tracking_mode = tracking_mode
        self.keypoint_likelihood_threshold = keypoint_likelihood_threshold
        self.average_likelihood_threshold = average_likelihood_threshold
        self.keypoint_number_threshold = keypoint_number_threshold
        
        # Initialize MediaPipe components
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize pose detector
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Store previous keypoints for tracking
        self.prev_keypoints = None
        self.frame_count = 0
        
        # Initialize keypoint mapping based on model type
        self._init_keypoint_mapping()
        
        logger.info(f"Initialized PoseDetector with model={model_type}, tracking={tracking_mode}")
    
    def _init_keypoint_mapping(self):
        """Initialize keypoint mapping based on selected model."""
        # Common keypoint mapping for all models
        self.keypoint_mapping = {
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
        
        # Add model-specific mappings if needed
        if self.model_type == 'HALPE_26':
            self.keypoint_mapping.update({
                "left_big_toe": 20,
                "right_big_toe": 21,
                "left_small_toe": 22,
                "right_small_toe": 23,
                "left_heel": 24,
                "right_heel": 25,
            })
    
    def find_pose(self, frame: np.ndarray) -> List[Dict[int, Tuple[float, float, float]]]:
        """
        Process frame and find pose landmarks for all detected persons.
        
        Args:
            frame: Input video frame
            
        Returns:
            List of dictionaries mapping landmark indices to (x, y, z) coordinates
        """
        self.frame_count += 1
        h, w, _ = frame.shape
        
        # Only run detection every det_frequency frames
        run_detection = (self.frame_count % self.det_frequency == 1) or (self.prev_keypoints is None)
        
        if run_detection:
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame with MediaPipe
            results = self.pose.process(rgb_frame)
            
            if not results.pose_landmarks:
                return []
            
            # Convert landmarks to our format
            keypoints = []
            scores = []
            
            # For now, MediaPipe only detects one person
            # In a real multi-person system, we'd iterate through multiple detections
            person_keypoints = {}
            person_scores = np.ones(33)  # MediaPipe has 33 landmarks
            
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                x = landmark.x * w
                y = landmark.y * h
                z = landmark.z
                visibility = landmark.visibility if hasattr(landmark, 'visibility') else 1.0
                
                # Add to person's keypoints
                person_keypoints[idx] = (x, y, z)
                person_scores[idx] = visibility
            
            keypoints.append(person_keypoints)
            scores.append(person_scores)
            
            # Filter keypoints and persons based on confidence thresholds
            filtered_keypoints, filtered_scores = self._filter_detections(keypoints, scores)
            
            # Apply tracking
            tracked_keypoints, tracked_scores = self._apply_tracking(filtered_keypoints, filtered_scores, frame)
            
            self.prev_keypoints = tracked_keypoints
            return tracked_keypoints
        
        else:
            # Use tracking for intermediate frames
            return self.prev_keypoints
    
    def _filter_detections(self, keypoints: List[Dict[int, Tuple[float, float, float]]], 
                          scores: List[np.ndarray]) -> Tuple[List[Dict[int, Tuple[float, float, float]]], List[np.ndarray]]:
        """
        Filter detections based on confidence thresholds.
        
        Args:
            keypoints: List of keypoint dictionaries
            scores: List of score arrays
            
        Returns:
            Filtered keypoints and scores
        """
        filtered_keypoints = []
        filtered_scores = []
        
        for person_idx, (person_kpts, person_scores) in enumerate(zip(keypoints, scores)):
            # Filter keypoints by likelihood threshold
            valid_keypoints = {}
            valid_scores = []
            
            for kpt_idx, (coords, score) in enumerate(zip(person_kpts.items(), person_scores)):
                idx, (x, y, z) = coords
                if score >= self.keypoint_likelihood_threshold:
                    valid_keypoints[idx] = (x, y, z)
                    valid_scores.append(score)
            
            # Check if person has enough valid keypoints
            if len(valid_keypoints) >= len(person_kpts) * self.keypoint_number_threshold:
                # Check if average confidence is high enough
                if np.mean(valid_scores) >= self.average_likelihood_threshold:
                    filtered_keypoints.append(valid_keypoints)
                    filtered_scores.append(person_scores)
        
        return filtered_keypoints, filtered_scores
    
    def _apply_tracking(self, keypoints: List[Dict[int, Tuple[float, float, float]]], 
                       scores: List[np.ndarray],
                       frame: np.ndarray) -> Tuple[List[Dict[int, Tuple[float, float, float]]], List[np.ndarray]]:
        """
        Apply tracking to maintain consistent person IDs.
        
        Args:
            keypoints: List of keypoint dictionaries
            scores: List of score arrays
            frame: Current video frame
            
        Returns:
            Tracked keypoints and scores
        """
        if self.prev_keypoints is None or not keypoints:
            return keypoints, scores
        
        if self.tracking_mode == 'sports2d':
            return self._track_sports2d(keypoints, scores)
        else:
            # Default simple tracking
            return keypoints, scores
    
    def _track_sports2d(self, keypoints: List[Dict[int, Tuple[float, float, float]]], 
                       scores: List[np.ndarray]) -> Tuple[List[Dict[int, Tuple[float, float, float]]], List[np.ndarray]]:
        """
        Track people using Sports2D approach (distance-based association).
        
        Args:
            keypoints: List of keypoint dictionaries
            scores: List of score arrays
            
        Returns:
            Tracked keypoints and scores
        """
        # Simple implementation for now
        if not self.prev_keypoints or not keypoints:
            return keypoints, scores
        
        # For each previous person, find the closest current person
        tracked_keypoints = []
        tracked_scores = []
        used_indices = set()
        
        for prev_kpts in self.prev_keypoints:
            best_distance = float('inf')
            best_idx = -1
            
            for i, curr_kpts in enumerate(keypoints):
                if i in used_indices:
                    continue
                
                # Calculate distance between common keypoints
                total_dist = 0
                count = 0
                
                for kpt_idx in prev_kpts:
                    if kpt_idx in curr_kpts:
                        prev_pos = prev_kpts[kpt_idx]
                        curr_pos = curr_kpts[kpt_idx]
                        dist = np.sqrt((prev_pos[0] - curr_pos[0])**2 + 
                                      (prev_pos[1] - curr_pos[1])**2)
                        total_dist += dist
                        count += 1
                
                if count > 0:
                    avg_dist = total_dist / count
                    if avg_dist < best_distance:
                        best_distance = avg_dist
                        best_idx = i
            
            if best_idx != -1 and best_distance < 100:  # Threshold for association
                tracked_keypoints.append(keypoints[best_idx])
                tracked_scores.append(scores[best_idx])
                used_indices.add(best_idx)
            else:
                # If no match found, keep previous keypoints
                tracked_keypoints.append(prev_kpts)
                tracked_scores.append(np.ones(len(scores[0])) * 0.5)  # Default score
        
        # Add any new detections that weren't matched
        for i, (kpts, s) in enumerate(zip(keypoints, scores)):
            if i not in used_indices:
                tracked_keypoints.append(kpts)
                tracked_scores.append(s)
        
        return tracked_keypoints, tracked_scores
    
    def draw_pose_landmarks(self, frame: np.ndarray, landmarks: Dict[int, Tuple[float, float, float]], 
                           color: Tuple[int, int, int] = (0, 255, 0),
                           thickness: int = 2,
                           radius: int = 5,
                           draw_connections: bool = True) -> np.ndarray:
        """
        Draw pose landmarks and connections on the frame.
        
        Args:
            frame: Input video frame
            landmarks: Dictionary of landmark indices to coordinates
            color: Color to use for drawing
            thickness: Line thickness for connections
            radius: Radius for landmark points
            draw_connections: Whether to draw connections between landmarks
            
        Returns:
            Frame with landmarks drawn
        """
        for idx, (x, y, _) in landmarks.items():
            cv2.circle(frame, (int(x), int(y)), radius, color, -1)
        
        if draw_connections:
            # Define connections based on model type
            connections = self._get_connections()
            
            for connection in connections:
                if connection[0] in landmarks and connection[1] in landmarks:
                    pt1 = (int(landmarks[connection[0]][0]), int(landmarks[connection[0]][1]))
                    pt2 = (int(landmarks[connection[1]][0]), int(landmarks[connection[1]][1]))
                    cv2.line(frame, pt1, pt2, color, thickness)
        
        return frame
    
    def _get_connections(self) -> List[Tuple[int, int]]:
        """Get landmark connections based on model type."""
        # Basic connections for all models
        connections = [
            # Torso
            (11, 12), (11, 23), (12, 24), (23, 24),
            # Right arm
            (11, 13), (13, 15),
            # Left arm
            (12, 14), (14, 16),
            # Right leg
            (23, 25), (25, 27),
            # Left leg
            (24, 26), (26, 28),
        ]
        
        # Add model-specific connections
        if self.model_type == 'HALPE_26' or self.model_type == 'COCO_133':
            connections.extend([
                # Feet
                (27, 31), (28, 32), (27, 29), (28, 30),
                (29, 31), (30, 32)
            ])
        
        return connections
    
    def get_joint_angles(self, landmarks: Dict[int, Tuple[float, float, float]]) -> Dict[str, float]:
        """
        Calculate common joint angles from landmarks.
        
        Args:
            landmarks: Dictionary of landmark coordinates
            
        Returns:
            Dictionary of joint names to angles
        """
        angles = {}
        
        # Calculate elbow angles
        if all(idx in landmarks for idx in [11, 13, 15]):  # Left elbow
            angles["left_elbow"] = self._calculate_angle(
                landmarks[11], landmarks[13], landmarks[15]
            )
            
        if all(idx in landmarks for idx in [12, 14, 16]):  # Right elbow
            angles["right_elbow"] = self._calculate_angle(
                landmarks[12], landmarks[14], landmarks[16]
            )
        
        # Calculate knee angles
        if all(idx in landmarks for idx in [23, 25, 27]):  # Left knee
            angles["left_knee"] = self._calculate_angle(
                landmarks[23], landmarks[25], landmarks[27]
            )
            
        if all(idx in landmarks for idx in [24, 26, 28]):  # Right knee
            angles["right_knee"] = self._calculate_angle(
                landmarks[24], landmarks[26], landmarks[28]
            )
        
        # Calculate hip angles
        if all(idx in landmarks for idx in [11, 23, 25]):  # Left hip
            angles["left_hip"] = self._calculate_angle(
                landmarks[11], landmarks[23], landmarks[25]
            )
            
        if all(idx in landmarks for idx in [12, 24, 26]):  # Right hip
            angles["right_hip"] = self._calculate_angle(
                landmarks[12], landmarks[24], landmarks[26]
            )
        
        # Calculate trunk angle
        if all(idx in landmarks for idx in [11, 12, 23, 24, 25, 26]):
            shoulder_midpoint = self._get_midpoint(landmarks[11], landmarks[12])
            hip_midpoint = self._get_midpoint(landmarks[23], landmarks[24])
            knee_midpoint = self._get_midpoint(landmarks[25], landmarks[26])
            
            angles["trunk"] = self._calculate_angle(
                shoulder_midpoint, hip_midpoint, knee_midpoint
            )
        
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
    
    def _get_midpoint(self, p1: Tuple[float, float, float], 
                     p2: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Calculate the midpoint between two points."""
        return (
            (p1[0] + p2[0]) / 2,
            (p1[1] + p2[1]) / 2,
            (p1[2] + p2[2]) / 2
        )
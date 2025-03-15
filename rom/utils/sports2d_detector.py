# rom/utils/sports2d_detector.py
import numpy as np
import cv2
from rtmlib import PoseTracker, BodyWithFeet, Wholebody, Body
from anytree import RenderTree
from typing import Dict, List, Tuple, Any, Optional

class Sports2DPoseDetector:
    """Pose detector using RTMLib from Sports2D."""
    
    def __init__(self, 
                 model_type='HALPE_26',
                 mode='balanced',
                 det_frequency=4,
                 tracking_mode='sports2d',
                 backend='auto',
                 device='auto'):
        """
        Initialize the pose detector with Sports2D/RTMLib.
        
        Args:
            model_type: Type of pose model ('HALPE_26', 'COCO_133_WRIST', 'COCO_133', 'COCO_17')
            mode: 'lightweight', 'balanced', 'performance' or dict with custom config
            det_frequency: Detection frequency (detection runs every N frames)
            tracking_mode: 'sports2d' or 'deepsort'
            backend: 'auto', 'onnxruntime', 'openvino', 'opencv'
            device: 'auto', 'cpu', 'cuda', 'mps', 'rocm'
        """
        self.model_type = model_type.upper()
        self.mode = mode
        self.det_frequency = det_frequency
        self.tracking_mode = tracking_mode
        self.backend = backend
        self.device = device
        
        # Setup backend and device
        self._setup_backend_device()
        
        # Initialize the pose tracker with appropriate model
        self._setup_pose_tracker()
        
        # Set up keypoint mapping
        self._init_keypoint_mapping()
        
        # Initialize tracking variables
        self.prev_keypoints = None
        self.frame_count = 0
    
    def _setup_backend_device(self):
        """Set up the backend and device based on available hardware."""
        if self.device != 'auto' and self.backend != 'auto':
            return
            
        try:
            import torch
            import onnxruntime as ort
            if torch.cuda.is_available() and 'CUDAExecutionProvider' in ort.get_available_providers():
                self.device = 'cuda'
                self.backend = 'onnxruntime'
            elif torch.cuda.is_available() and 'ROCMExecutionProvider' in ort.get_available_providers():
                self.device = 'rocm'
                self.backend = 'onnxruntime'
            else:
                raise ImportError
        except ImportError:
            try:
                import onnxruntime as ort
                if 'MPSExecutionProvider' in ort.get_available_providers() or 'CoreMLExecutionProvider' in ort.get_available_providers():
                    self.device = 'mps'
                    self.backend = 'onnxruntime'
                else:
                    raise ImportError
            except ImportError:
                self.device = 'cpu'
                self.backend = 'openvino'
    
    def _setup_pose_tracker(self):
        """Set up the RTMLib pose tracker with the appropriate model."""
        # Select model class based on model_type
        if self.model_type in ('HALPE_26', 'BODY_WITH_FEET'):
            self.ModelClass = BodyWithFeet
        elif self.model_type == 'COCO_133_WRIST':
            self.ModelClass = Wholebody
        elif self.model_type in ('COCO_133', 'WHOLE_BODY'):
            self.ModelClass = Wholebody
        elif self.model_type in ('COCO_17', 'BODY'):
            self.ModelClass = Body
        else:
            raise ValueError(f"Invalid model_type: {self.model_type}")
            
        # Initialize pose tracker
        self.pose_tracker = PoseTracker(
            self.ModelClass,
            det_frequency=self.det_frequency,
            mode=self.mode,
            backend=self.backend,
            device=self.device,
            tracking=False,  # We'll handle tracking ourselves
            to_openpose=False
        )
    
    def _init_keypoint_mapping(self):
        """Initialize keypoint mapping based on selected model."""
        # This is a simplified version - you'll need to complete it based on your needs
        self.keypoint_mapping = {
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
            "right_ankle": 16,
            "left_big_toe": 20,
            "right_big_toe": 21,
            "left_small_toe": 22,
            "right_small_toe": 23,
            "left_heel": 24,
            "right_heel": 25,
        }
    
    def find_pose(self, frame: np.ndarray) -> List[Dict[int, Tuple[float, float, float]]]:
        """
        Process frame and find pose landmarks for all detected persons.
        
        Args:
            frame: Input video frame
            
        Returns:
            List of dictionaries mapping landmark indices to (x, y, z) coordinates
        """
        self.frame_count += 1
        
        # Detect poses with RTMLib
        keypoints, scores = self.pose_tracker(frame)
        
        # Track poses across frames if needed
        if self.tracking_mode == 'sports2d':
            if self.prev_keypoints is None:
                self.prev_keypoints = keypoints
            sorted_prev, keypoints, scores = self._sort_people_sports2d(self.prev_keypoints, keypoints, scores)
            self.prev_keypoints = sorted_prev
        
        # Convert to the format expected by your ROM system
        result = []
        for person_idx in range(len(keypoints)):
            person_keypoints = {}
            for kpt_idx in range(len(keypoints[person_idx])):
                # Only include keypoints with sufficient confidence
                if scores[person_idx][kpt_idx] >= 0.3:  # You can adjust this threshold
                    x, y = keypoints[person_idx][kpt_idx]
                    person_keypoints[kpt_idx] = (float(x), float(y), 0.0)  # Z is 0 for 2D
            
            if person_keypoints:  # Only include persons with valid keypoints
                result.append(person_keypoints)
        
        return result
    
    def _sort_people_sports2d(self, prev_keypoints, curr_keypoints, scores=None):
        """Sort people across frames for consistent tracking."""
        # This is a simplified implementation - you'll need to refer to 
        # the full implementation in Sports2D/Utilities/common.py
        
        # For now, a simple matching based on centroid distance
        sorted_keypoints = curr_keypoints.copy()
        sorted_scores = scores.copy() if scores is not None else None
        
        # If no previous keypoints or no current keypoints, return as is
        if len(prev_keypoints) == 0 or len(curr_keypoints) == 0:
            return prev_keypoints, sorted_keypoints, sorted_scores
        
        # Calculate centroids for previous and current keypoints
        prev_centroids = []
        for person in prev_keypoints:
            valid_pts = person[~np.isnan(person).any(axis=1)]
            centroid = np.mean(valid_pts, axis=0) if len(valid_pts) > 0 else np.array([np.nan, np.nan])
            prev_centroids.append(centroid)
            
        curr_centroids = []
        for person in curr_keypoints:
            valid_pts = person[~np.isnan(person).any(axis=1)]
            centroid = np.mean(valid_pts, axis=0) if len(valid_pts) > 0 else np.array([np.nan, np.nan])
            curr_centroids.append(centroid)
        
        # Match based on minimum distance between centroids
        matched_indices = []
        for i, prev_cent in enumerate(prev_centroids):
            if np.isnan(prev_cent).any():
                continue
            
            min_dist = float('inf')
            min_idx = -1
            for j, curr_cent in enumerate(curr_centroids):
                if np.isnan(curr_cent).any() or j in matched_indices:
                    continue
                
                dist = np.linalg.norm(prev_cent - curr_cent)
                if dist < min_dist:
                    min_dist = dist
                    min_idx = j
            
            if min_idx != -1 and min_dist < 100:  # Threshold for matching
                matched_indices.append(min_idx)
                # Swap to maintain consistent ordering
                if min_idx != i and min_idx < len(sorted_keypoints):
                    sorted_keypoints[i], sorted_keypoints[min_idx] = sorted_keypoints[min_idx], sorted_keypoints[i]
                    if sorted_scores is not None:
                        sorted_scores[i], sorted_scores[min_idx] = sorted_scores[min_idx], sorted_scores[i]
        
        return prev_keypoints, sorted_keypoints, sorted_scores
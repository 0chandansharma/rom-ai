# rom/core/exercise_handler.py
from typing import Dict, Tuple, Any, Optional, Type
import numpy as np
import logging

from rom.core.base import ROMTest, ROMData
from rom.utils.pose_detector import PoseDetector
from rom.utils.visualization import PoseVisualizer
from rom.tests.lower_back_test import (
    LowerBackFlexionTest, 
    LowerBackExtensionTest,
    LowerBackLateralFlexionTest,
    LowerBackRotationTest
)

# Setup logging
logger = logging.getLogger("rom.exercise_handler")


class ExerciseManager:
    """
    Manager for ROM exercises and assessments.
    
    This class serves as the main entry point for ROM assessments,
    managing the lifecycle of different test types and providing a 
    unified interface for processing frames.
    """
    
    def __init__(self):
        """Initialize the exercise manager."""
        self.pose_detector = PoseDetector()
        self.visualizer = PoseVisualizer()
        
        # Register test types
        self.test_registry = {
            # Lower back tests
            "lower_back_flexion": LowerBackFlexionTest,
            "lower_back_extension": LowerBackExtensionTest,
            "lower_back_lateral_flexion_left": lambda pd, vis, cfg: 
                LowerBackLateralFlexionTest(pd, vis, cfg, side="left"),
            "lower_back_lateral_flexion_right": lambda pd, vis, cfg: 
                LowerBackLateralFlexionTest(pd, vis, cfg, side="right"),
            "lower_back_rotation_left": lambda pd, vis, cfg:
                LowerBackRotationTest(pd, vis, cfg, side="left"),
            "lower_back_rotation_right": lambda pd, vis, cfg:
                LowerBackRotationTest(pd, vis, cfg, side="right"),
        }
        
        # Store active test instances
        self.active_tests = {}
    
    def create_test(self, test_type: str, config: Optional[Dict[str, Any]] = None, 
                   session_id: Optional[str] = None) -> ROMTest:
        """
        Create a test instance of the specified type.
        
        Args:
            test_type: Type of test to create
            config: Configuration for the test
            session_id: Optional session ID for tracking
            
        Returns:
            Initialized test instance
            
        Raises:
            ValueError: If test_type is not supported
        """
        if test_type not in self.test_registry:
            raise ValueError(f"Unsupported test type: {test_type}")
        
        # Generate session ID if not provided
        session_id = session_id or f"session_{id(test_type)}_{id(config)}"
        
        # Create test instance
        test_factory = self.test_registry[test_type]
        test_instance = test_factory(self.pose_detector, self.visualizer, config or {})
        
        # Store for later reference
        self.active_tests[session_id] = test_instance
        
        logger.info(f"Created test: {test_type} (Session: {session_id})")
        return test_instance
    
    def get_test(self, session_id: str) -> Optional[ROMTest]:
        """
        Get an existing test instance by session ID.
        
        Args:
            session_id: Session ID of the test
            
        Returns:
            Test instance or None if not found
        """
        return self.active_tests.get(session_id)
    
    def process_frame(self, frame: np.ndarray, test_type: str, 
                     session_id: Optional[str] = None,
                     config: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process a frame with the specified test type.
        
        Args:
            frame: Input video frame
            test_type: Type of test to use
            session_id: Optional session ID for using an existing test
            config: Configuration if creating a new test
            
        Returns:
            Tuple of (processed_frame, rom_data_dict)
        """
        # Get or create test instance
        test_instance = None
        if session_id:
            test_instance = self.get_test(session_id)
        
        if test_instance is None:
            test_instance = self.create_test(test_type, config, session_id)
        
        # Process frame
        return test_instance.process_frame(frame)
    
    def end_session(self, session_id: str) -> Dict[str, Any]:
        """
        End a test session and return final results.
        
        Args:
            session_id: Session ID to end
            
        Returns:
            Final ROM data dictionary
            
        Raises:
            ValueError: If session not found
        """
        if session_id not in self.active_tests:
            raise ValueError(f"Session not found: {session_id}")
        
        test_instance = self.active_tests[session_id]
        final_data = test_instance.data.to_dict()
        
        # Clean up
        del self.active_tests[session_id]
        logger.info(f"Ended session: {session_id}")
        
        return final_data
    
    def get_available_tests(self) -> Dict[str, Dict[str, Any]]:
        """
        Get dictionary of available test types with metadata.
        
        Returns:
            Dictionary of test types to metadata
        """
        return {
            "lower_back_flexion": {
                "name": "Lower Back Flexion",
                "description": "Assessment of lower back forward bending motion",
                "target_joint": "lower_back",
                "movement_type": "flexion"
            },
            "lower_back_extension": {
                "name": "Lower Back Extension",
                "description": "Assessment of lower back backward bending motion",
                "target_joint": "lower_back",
                "movement_type": "extension"
            },
            "lower_back_lateral_flexion_left": {
                "name": "Lower Back Lateral Flexion (Left)",
                "description": "Assessment of lower back side bending to the left",
                "target_joint": "lower_back",
                "movement_type": "lateral_flexion_left"
            },
            "lower_back_lateral_flexion_right": {
                "name": "Lower Back Lateral Flexion (Right)",
                "description": "Assessment of lower back side bending to the right",
                "target_joint": "lower_back",
                "movement_type": "lateral_flexion_right"
            },
            "lower_back_rotation_left": {
                "name": "Lower Back Rotation (Left)",
                "description": "Assessment of lower back rotation to the left",
                "target_joint": "lower_back",
                "movement_type": "rotation_left"
            },
            "lower_back_rotation_right": {
                "name": "Lower Back Rotation (Right)",
                "description": "Assessment of lower back rotation to the right",
                "target_joint": "lower_back",
                "movement_type": "rotation_right"
            }
        }


class ExerciseHandler:
    """
    Legacy compatibility class for existing code.
    
    This class provides a simplified interface compatible with the
    original codebase while using the new ExerciseManager internally.
    """
    
    def __init__(self, exercise_name: str):
        """
        Initialize the exercise handler.
        
        Args:
            exercise_name: Name of the exercise
        """
        self.exercise_name = exercise_name
        self.manager = ExerciseManager()
        
        # Map old exercise names to new test types
        self.exercise_map = {
            "hawkins": "shoulder_hawkins_test",
            "lowerback": "lower_back_flexion",
            "lower_back": "lower_back_flexion",
            "lower_back_flexion": "lower_back_flexion",
            "lower_back_extension": "lower_back_extension"
        }
        
        # Get the appropriate test type
        self.test_type = self.exercise_map.get(exercise_name.lower(), "lower_back_flexion")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process a single frame.
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (processed_frame, rom_data)
        """
        return self.manager.process_frame(frame, self.test_type)
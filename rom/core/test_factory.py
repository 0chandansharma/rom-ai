# rom/core/test_factory.py
from typing import Dict, List, Any, Optional, Type
import logging
import time

from rom.core.base import EnhancedROMTest
from rom.utils.pose_detector import PoseDetector
from rom.utils.visualization import EnhancedVisualizer
from rom.core.configurable_test import ConfigurableROMTest
from rom.config.config_manager import ConfigManager
from rom.tests.lower_back_test import (
    EnhancedLowerBackFlexionTest, 
    EnhancedLowerBackExtensionTest,
    EnhancedLowerBackLateralFlexionTest,
    EnhancedLowerBackRotationTest
)

logger = logging.getLogger("rom.test_factory")

class ConfigurableTestFactory:
    """Factory for creating ROM tests based on user configuration."""
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the test factory.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager or ConfigManager()
        
        # Register standard test types
        self.test_registry = {
            # Lower back tests
            "lower_back_flexion": EnhancedLowerBackFlexionTest,
            "lower_back_extension": EnhancedLowerBackExtensionTest,
            "lower_back_lateral_flexion_left": lambda pd, vis, cfg: 
                EnhancedLowerBackLateralFlexionTest(pd, vis, cfg, side="left"),
            "lower_back_lateral_flexion_right": lambda pd, vis, cfg: 
                EnhancedLowerBackLateralFlexionTest(pd, vis, cfg, side="right"),
            "lower_back_rotation_left": lambda pd, vis, cfg:
                EnhancedLowerBackRotationTest(pd, vis, cfg, side="left"),
            "lower_back_rotation_right": lambda pd, vis, cfg:
                EnhancedLowerBackRotationTest(pd, vis, cfg, side="right"),
        }
        
        # Store active test instances
        self.active_tests = {}
        
        logger.info(f"Initialized ConfigurableTestFactory with {len(self.test_registry)} test types")
    
    def create_test(self, 
                   test_type: str, 
                   test_id: Optional[str] = None,
                   config: Optional[Dict[str, Any]] = None) -> EnhancedROMTest:
        """
        Create a test instance based on configuration.
        
        Args:
            test_type: Type of test to create
            test_id: Optional unique ID for the test
            config: Optional configuration overrides
            
        Returns:
            Configured test instance
        """
        # Generate test ID if not provided
        test_id = test_id or f"{test_type}_{int(time.time())}"
        
        # Get base configuration for this test type
        base_config = self.config_manager.get_complete_config(test_type)
        
        # Merge with provided config overrides
        if config:
            for key, value in config.items():
                if isinstance(value, dict) and key in base_config and isinstance(base_config[key], dict):
                    # Deep merge for dictionaries
                    base_config[key].update(value)
                else:
                    # Direct replacement for other types
                    base_config[key] = value
        
        # Create PoseDetector with configuration
        pose_detector = PoseDetector(
            model_type=base_config.get("model_type", "HALPE_26"),
            det_frequency=base_config.get("det_frequency", 4),
            tracking_mode=base_config.get("tracking_mode", "sports2d"),
            keypoint_likelihood_threshold=base_config.get("keypoint_likelihood_threshold", 0.3),
            average_likelihood_threshold=base_config.get("average_likelihood_threshold", 0.5),
            keypoint_number_threshold=base_config.get("keypoint_number_threshold", 0.3)
        )
        
        # Create Visualizer with configuration
        vis_config = base_config.get("visualization", {})
        visualizer = EnhancedVisualizer(
            theme=vis_config.get("theme", "dark")
        )
        
        # Check if this is a standard test type
        if test_type in self.test_registry:
            # Create standard test
            test_factory = self.test_registry[test_type]
            test_instance = test_factory(pose_detector, visualizer, base_config)
        else:
            # Check if this is a custom test type
            custom_tests = self.config_manager.config.get("custom_tests", {})
            if test_type in custom_tests:
                # Create configurable test with custom configuration
                test_instance = ConfigurableROMTest(
                    pose_detector, 
                    visualizer, 
                    base_config,
                    test_type,
                    self.config_manager
                )
            else:
                # Create generic configurable test
                logger.warning(f"Unknown test type: {test_type}, creating generic configurable test")
                test_instance = ConfigurableROMTest(
                    pose_detector, 
                    visualizer, 
                    base_config,
                    "custom",
                    self.config_manager
                )
        
        # Store test instance
        self.active_tests[test_id] = test_instance
        
        logger.info(f"Created test instance: {test_type} ({test_id})")
        return test_instance
    
    def get_test(self, test_id: str) -> Optional[EnhancedROMTest]:
        """
        Get an existing test instance by ID.
        
        Args:
            test_id: ID of the test
            
        Returns:
            Test instance or None if not found
        """
        return self.active_tests.get(test_id)
    
    def end_test(self, test_id: str) -> bool:
        """
        End a test session and clean up resources.
        
        Args:
            test_id: ID of the test
            
        Returns:
            True if successful, False otherwise
        """
        if test_id in self.active_tests:
            del self.active_tests[test_id]
            logger.info(f"Ended test session: {test_id}")
            return True
        
        return False
    
    def get_available_tests(self) -> Dict[str, Dict[str, Any]]:
        """
        Get dictionary of available test types with metadata.
        
        Returns:
            Dictionary of test types to metadata
        """
        standard_tests = {
            "lower_back_flexion": {
                "name": "Lower Back Flexion",
                "description": "Assessment of lower back forward bending motion",
                "target_joint": "lower_back",
                "movement_type": "flexion",
                "target_rom": 60.0
            },
            "lower_back_extension": {
                "name": "Lower Back Extension",
                "description": "Assessment of lower back backward bending motion",
                "target_joint": "lower_back",
                "movement_type": "extension",
                "target_rom": 25.0
            },
            "lower_back_lateral_flexion_left": {
                "name": "Lower Back Lateral Flexion (Left)",
                "description": "Assessment of lower back side bending to the left",
                "target_joint": "lower_back",
                "movement_type": "lateral_flexion_left",
                "target_rom": 25.0
            },
            "lower_back_lateral_flexion_right": {
                "name": "Lower Back Lateral Flexion (Right)",
                "description": "Assessment of lower back side bending to the right",
                "target_joint": "lower_back",
                "movement_type": "lateral_flexion_right",
                "target_rom": 25.0
            },
            "lower_back_rotation_left": {
                "name": "Lower Back Rotation (Left)",
                "description": "Assessment of lower back rotation to the left",
                "target_joint": "lower_back",
                "movement_type": "rotation_left",
                "target_rom": 45.0
            },
            "lower_back_rotation_right": {
                "name": "Lower Back Rotation (Right)",
                "description": "Assessment of lower back rotation to the right",
                "target_joint": "lower_back",
                "movement_type": "rotation_right",
                "target_rom": 45.0
            }
        }
        
        # Add custom tests
        custom_tests = {}
        for test_id, test_config in self.config_manager.config.get("custom_tests", {}).items():
            custom_tests[test_id] = {
                "name": test_config.get("name", test_id),
                "description": test_config.get("description", "Custom test configuration"),
                "target_joint": test_config.get("joint_type", "custom"),
                "movement_type": "custom",
                "target_rom": test_config.get("target_rom", 0),
                "is_custom": True
            }
        
        return {**standard_tests, **custom_tests}
    
    def create_custom_test(self, 
                          name: str,
                          body_parts: List[str],
                          joint_angles: List[Dict[str, Any]],
                          segment_angles: Optional[List[Dict[str, Any]]] = None,
                          primary_angle: Optional[str] = None,
                          target_rom: float = 60.0,
                          description: Optional[str] = None) -> str:
        """
        Create a custom test configuration.
        
        Args:
            name: Name for the custom test
            body_parts: List of body part names to track
            joint_angles: List of joint angle definitions
            segment_angles: Optional list of segment angle definitions
            primary_angle: Name of the primary angle to track
            target_rom: Target ROM value
            description: Optional description
            
        Returns:
            ID of the created test
        """
        # Create test configuration
        test_config = {
            "name": name,
            "description": description or f"Custom test: {name}",
            "body_parts": body_parts,
            "joint_angles": joint_angles,
            "segment_angles": segment_angles or [],
            "primary_angle": primary_angle or (joint_angles[0]["name"] if joint_angles else None),
            "target_rom": target_rom
        }
        
        # Save test configuration
        test_id = self.config_manager.create_custom_test(name, test_config)
        
        return test_id
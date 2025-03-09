# rom/config/config_manager.py
import json
import os
from typing import Dict, List, Any, Optional, Union
import logging
from pathlib import Path
import copy

logger = logging.getLogger("rom.config_manager")

class ConfigManager:
    """
    Manages configuration settings for ROM assessment system.
    
    This class provides methods for loading, saving, and managing
    configuration settings for the ROM assessment system.
    """
    
    DEFAULT_CONFIG = {
        # Pose detection settings
        "pose": {
            "model_type": "HALPE_26",
            "det_frequency": 4,
            "tracking_mode": "sports2d",
            "keypoint_likelihood_threshold": 0.3,
            "average_likelihood_threshold": 0.5,
            "keypoint_number_threshold": 0.3,
            "model_complexity": 1
        },
        
        # Assessment settings
        "assessment": {
            "ready_time_required": 20,
            "angle_buffer_size": 100,
            "position_tolerance": 10
        },
        
        # Visualization settings
        "visualization": {
            "theme": "dark",
            "show_landmarks": True,
            "show_connections": True,
            "show_angles": True,
            "show_info_panel": True,
            "show_trajectory": True,
            "trajectory_length": 100,
            "highlight_primary_angle": True,
            "display_mode": "body",  # "body", "list", "both", or "none"
            "font_size": 0.8,
            "line_thickness": 2
        },
        
        # Processing settings
        "processing": {
            "filter_type": "butterworth",  # "butterworth", "moving_average", "gaussian", or "none"
            "interpolate": True,
            "max_gap_size": 10,
            "smoothing_window": 5,
            "butterworth_cutoff": 6.0,
            "butterworth_order": 4,
            "fps": 30.0
        },
        
        # Reporting settings
        "reporting": {
            "include_plots": True,
            "include_recommendations": True,
            "include_normative_data": True,
            "output_format": "pdf"  # "pdf", "png", or "html"
        },
        
        # Test-specific settings
        "test_defaults": {
            "lower_back_flexion": {
                "joint_angles": ["trunk_angle", "left_hip_angle", "right_hip_angle"],
                "segment_angles": ["trunk_segment", "left_thigh_segment", "right_thigh_segment"],
                "primary_angle": "trunk_angle",
                "target_rom": 60.0
            },
            "lower_back_extension": {
                "joint_angles": ["trunk_angle", "left_hip_angle", "right_hip_angle"],
                "segment_angles": ["trunk_segment", "left_thigh_segment", "right_thigh_segment"],
                "primary_angle": "trunk_angle",
                "target_rom": 25.0
            },
            "lower_back_lateral_flexion_left": {
                "joint_angles": ["lateral_trunk_angle", "left_shoulder_hip_angle"],
                "segment_angles": ["trunk_segment_lateral", "left_side_segment"],
                "primary_angle": "lateral_trunk_angle",
                "target_rom": 25.0
            },
            "lower_back_lateral_flexion_right": {
                "joint_angles": ["lateral_trunk_angle", "right_shoulder_hip_angle"],
                "segment_angles": ["trunk_segment_lateral", "right_side_segment"],
                "primary_angle": "lateral_trunk_angle",
                "target_rom": 25.0
            },
            "lower_back_rotation_left": {
                "joint_angles": ["rotation_angle", "shoulder_alignment", "hip_alignment"],
                "segment_angles": ["shoulder_segment", "hip_segment"],
                "primary_angle": "rotation_angle",
                "target_rom": 45.0
            },
            "lower_back_rotation_right": {
                "joint_angles": ["rotation_angle", "shoulder_alignment", "hip_alignment"],
                "segment_angles": ["shoulder_segment", "hip_segment"],
                "primary_angle": "rotation_angle",
                "target_rom": 45.0
            }
        },
        
        # Custom test settings
        "custom_tests": {}
    }
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory for storing configuration files
        """
        self.config_dir = Path(config_dir) if config_dir else Path.home() / ".rom"
        self.config_path = self.config_dir / "config.json"
        self.custom_configs_dir = self.config_dir / "custom_configs"
        
        # Create directories if they don't exist
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.custom_configs_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or create default configuration
        self.config = self._load_config()
        
        logger.info(f"Configuration manager initialized with config at: {self.config_path}")
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file or create default.
        
        Returns:
            Configuration dictionary
        """
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults in case new options were added
                    return self._merge_with_defaults(config)
            except Exception as e:
                logger.error(f"Error loading configuration: {str(e)}")
                logger.info("Using default configuration")
                return copy.deepcopy(self.DEFAULT_CONFIG)
        else:
            # Create default configuration
            logger.info("Creating default configuration")
            default_config = copy.deepcopy(self.DEFAULT_CONFIG)
            self._save_config(default_config)
            return default_config
    
    def _save_config(self, config: Dict[str, Any]) -> bool:
        """
        Save configuration to file.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=4)
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            return False
    
    def _merge_with_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge provided configuration with defaults to ensure all options are present.
        
        Args:
            config: User configuration
            
        Returns:
            Merged configuration
        """
        merged_config = copy.deepcopy(self.DEFAULT_CONFIG)
        
        def merge_dicts(source, destination):
            for key, value in source.items():
                if isinstance(value, dict) and key in destination:
                    merge_dicts(value, destination[key])
                else:
                    destination[key] = value
        
        merge_dicts(config, merged_config)
        return merged_config
    
    def save(self) -> bool:
        """
        Save current configuration.
        
        Returns:
            True if successful, False otherwise
        """
        return self._save_config(self.config)
    
    def get_pose_config(self) -> Dict[str, Any]:
        """
        Get pose detection configuration.
        
        Returns:
            Pose detection configuration
        """
        return self.config.get("pose", {})
    
    def get_assessment_config(self) -> Dict[str, Any]:
        """
        Get assessment configuration.
        
        Returns:
            Assessment configuration
        """
        return self.config.get("assessment", {})
    
    def get_visualization_config(self) -> Dict[str, Any]:
        """
        Get visualization configuration.
        
        Returns:
            Visualization configuration
        """
        return self.config.get("visualization", {})
    
    def get_processing_config(self) -> Dict[str, Any]:
        """
        Get processing configuration.
        
        Returns:
            Processing configuration
        """
        return self.config.get("processing", {})
    
    def get_test_config(self, test_type: str) -> Dict[str, Any]:
        """
        Get configuration for a specific test type.
        
        Args:
            test_type: Type of test
            
        Returns:
            Test configuration
        """
        if test_type in self.config.get("custom_tests", {}):
            return self.config["custom_tests"][test_type]
        else:
            return self.config.get("test_defaults", {}).get(test_type, {})
    
    def get_complete_config(self, test_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get complete configuration for a test.
        
        Args:
            test_type: Optional test type to include specific settings
            
        Returns:
            Complete configuration dictionary
        """
        complete_config = {
            **self.get_pose_config(),
            **self.get_assessment_config(),
            **self.get_visualization_config(),
            **self.get_processing_config()
        }
        
        if test_type:
            complete_config.update(self.get_test_config(test_type))
        
        return complete_config
    
    def update_section(self, section: str, updates: Dict[str, Any]) -> bool:
        """
        Update a section of the configuration.
        
        Args:
            section: Section name (pose, assessment, visualization, processing)
            updates: Dictionary of updates
            
        Returns:
            True if successful, False otherwise
        """
        if section not in self.config:
            logger.error(f"Invalid configuration section: {section}")
            return False
        
        self.config[section].update(updates)
        return self.save()
    
    def update_test_config(self, test_type: str, updates: Dict[str, Any], is_custom: bool = False) -> bool:
        """
        Update configuration for a specific test type.
        
        Args:
            test_type: Type of test
            updates: Dictionary of updates
            is_custom: Whether this is a custom test
            
        Returns:
            True if successful, False otherwise
        """
        if is_custom:
            if "custom_tests" not in self.config:
                self.config["custom_tests"] = {}
            
            if test_type not in self.config["custom_tests"]:
                self.config["custom_tests"][test_type] = {}
            
            self.config["custom_tests"][test_type].update(updates)
        else:
            if "test_defaults" not in self.config:
                self.config["test_defaults"] = {}
            
            if test_type not in self.config["test_defaults"]:
                self.config["test_defaults"][test_type] = {}
            
            self.config["test_defaults"][test_type].update(updates)
        
        return self.save()
    
    def create_custom_test(self, test_name: str, test_config: Dict[str, Any]) -> str:
        """
        Create a new custom test configuration.
        
        Args:
            test_name: Name for the custom test
            test_config: Test configuration
            
        Returns:
            Unique ID for the custom test
        """
        # Generate unique ID
        test_id = f"custom_{test_name.lower().replace(' ', '_')}_{int(time.time())}"
        
        # Add to custom tests
        if "custom_tests" not in self.config:
            self.config["custom_tests"] = {}
        
        self.config["custom_tests"][test_id] = test_config
        
        # Save configuration
        self.save()
        
        # Also save as separate file for easier sharing
        custom_config_path = self.custom_configs_dir / f"{test_id}.json"
        with open(custom_config_path, 'w') as f:
            json.dump(test_config, f, indent=4)
        
        return test_id
    
    def load_custom_test(self, config_path: str) -> Optional[str]:
        """
        Load a custom test configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Test ID if successful, None otherwise
        """
        try:
            with open(config_path, 'r') as f:
                test_config = json.load(f)
            
            # Extract test name from config or filename
            if "name" in test_config:
                test_name = test_config["name"]
            else:
                test_name = Path(config_path).stem
            
            # Create custom test
            return self.create_custom_test(test_name, test_config)
            
        except Exception as e:
            logger.error(f"Error loading custom test: {str(e)}")
            return None
    
    def delete_custom_test(self, test_id: str) -> bool:
        """
        Delete a custom test configuration.
        
        Args:
            test_id: ID of custom test
            
        Returns:
            True if successful, False otherwise
        """
        if "custom_tests" in self.config and test_id in self.config["custom_tests"]:
            del self.config["custom_tests"][test_id]
            
            # Remove separate file if it exists
            custom_config_path = self.custom_configs_dir / f"{test_id}.json"
            if custom_config_path.exists():
                custom_config_path.unlink()
            
            return self.save()
        
        return False
    
    def reset_to_defaults(self) -> bool:
        """
        Reset configuration to defaults.
        
        Returns:
            True if successful, False otherwise
        """
        self.config = copy.deepcopy(self.DEFAULT_CONFIG)
        return self.save()
    
    def export_config(self, output_path: str) -> bool:
        """
        Export current configuration to file.
        
        Args:
            output_path: Path to save configuration
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            return True
        except Exception as e:
            logger.error(f"Error exporting configuration: {str(e)}")
            return False
    
    def import_config(self, config_path: str) -> bool:
        """
        Import configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(config_path, 'r') as f:
                imported_config = json.load(f)
            
            # Validate and merge with defaults
            self.config = self._merge_with_defaults(imported_config)
            return self.save()
            
        except Exception as e:
            logger.error(f"Error importing configuration: {str(e)}")
            return False
# rom/analysis/assessment_analyzer.py
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple, Any, Optional
import json
import os
from datetime import datetime

class AssessmentAnalyzer:
    """
    Analyze ROM assessment data and generate reports.
    
    This class provides methods for analyzing ROM assessment results
    and generating detailed reports with insights and recommendations.
    """
    
    def __init__(self, normative_data_path: Optional[str] = None):
        """
        Initialize the assessment analyzer.
        
        Args:
            normative_data_path: Path to normative data file (JSON)
        """
        self.normative_data = self._load_normative_data(normative_data_path)
    
    def _load_normative_data(self, data_path: Optional[str]) -> Dict[str, Any]:
        """
        Load normative data from JSON file.
        
        Args:
            data_path: Path to normative data file
            
        Returns:
            Dictionary of normative data
        """
        default_data = {
            "lower_back_flexion": {"mean": 60.0, "std": 10.0, "min": 40.0, "max": 80.0},
            "lower_back_extension": {"mean": 25.0, "std": 5.0, "min": 15.0, "max": 35.0},
            "lower_back_lateral_flexion_left": {"mean": 25.0, "std": 5.0, "min": 15.0, "max": 35.0},
            "lower_back_lateral_flexion_right": {"mean": 25.0, "std": 5.0, "min": 15.0, "max": 35.0},
            "lower_back_rotation_left": {"mean": 45.0, "std": 7.0, "min": 30.0, "max": 60.0},
            "lower_back_rotation_right": {"mean": 45.0, "std": 7.0, "min": 30.0, "max": 60.0}
        }
        
        if data_path and os.path.exists(data_path):
            try:
                with open(data_path, 'r') as f:
                    loaded_data = json.load(f)
                return {**default_data, **loaded_data}
            except:
                pass
        
        return default_data
    
    def analyze_assessment(self, 
                          assessment_data: Dict[str, Any], 
                          angle_history: Optional[Dict[str, List[float]]] = None) -> Dict[str, Any]:
        """
        Analyze assessment data and generate insights.
        
        Args:
            assessment_data: Assessment result data
            angle_history: Optional history of angle measurements
            
        Returns:
            Dictionary with analysis results
        """
        test_type = assessment_data.get("test_type")
        joint_type = assessment_data.get("joint_type")
        rom = assessment_data.get("rom")
        
        if not test_type or rom is None:
            return {
                "status": "error",
                "message": "Invalid assessment data"
            }
        
        # Get normative data for this test
        norm_data = self.normative_data.get(test_type, {})
        norm_mean = norm_data.get("mean", 0)
        norm_std = norm_data.get("std", 1)
        norm_min = norm_data.get("min", 0)
        norm_max = norm_data.get("max", 0)
        
        # Calculate percentile
        if norm_std > 0:
            z_score = (rom - norm_mean) / norm_std
            percentile = min(100, max(0, 50 + 50 * z_score / 3))  # Approximate percentile
        else:
            percentile = 50
        
        # Calculate percentage of normal
        percent_of_normal = (rom / norm_mean) * 100 if norm_mean > 0 else 0
        
        # Determine range category
        if rom >= norm_mean + norm_std:
            range_category = "excellent"
            severity = "none"
        elif rom >= norm_mean:
            range_category = "good"
            severity = "none"
        elif rom >= norm_mean - norm_std:
            range_category = "fair"
            severity = "mild"
        elif rom >= norm_mean - 2 * norm_std:
            range_category = "limited"
            severity = "moderate"
        else:
            range_category = "severely_limited"
            severity = "severe"
        
        # Analyze movement patterns if history available
        movement_analysis = {}
        if angle_history and len(angle_history) > 0:
            # Find primary angle (typically the one with the name matching test_type)
            primary_angle_name = next((name for name in angle_history.keys() if test_type in name), None)
            if not primary_angle_name and angle_history:
                # Just use the first one if no match
                primary_angle_name = list(angle_history.keys())[0]
            
            if primary_angle_name and len(angle_history[primary_angle_name]) > 10:
                angle_data = angle_history[primary_angle_name]
                
                # Analyze smoothness (jerk)
                diffs = np.diff(angle_data)
                jerk = np.diff(diffs)
                mean_jerk = np.mean(np.abs(jerk))
                
                if mean_jerk < 0.5:
                    smoothness = "very_smooth"
                elif mean_jerk < 1.0:
                    smoothness = "smooth"
                elif mean_jerk < 2.0:
                    smoothness = "somewhat_smooth"
                else:
                    smoothness = "jerky"
                
                # Analyze speed
                movement_time = len(angle_data) / 30  # Assuming 30 fps
                angle_range = max(angle_data) - min(angle_data)
                avg_speed = angle_range / movement_time if movement_time > 0 else 0
                
                if avg_speed < 5:
                    speed_category = "very_slow"
                elif avg_speed < 15:
                    speed_category = "slow"
                elif avg_speed < 30:
                    speed_category = "moderate"
                else:
                    speed_category = "fast"
                
                movement_analysis = {
                    "smoothness": smoothness,
                    "mean_jerk": float(mean_jerk),
                    "speed_category": speed_category,
                    "avg_speed_deg_per_sec": float(avg_speed)
                }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(test_type, range_category, severity, movement_analysis)
        
        # Comprehensive analysis result
        analysis_result = {
            "test_type": test_type,
            "joint_type": joint_type,
            "rom": rom,
            "norm_data": {
                "mean": norm_mean,
                "std": norm_std,
                "min": norm_min,
                "max": norm_max
            },
            "percentile": percentile,
            "percent_of_normal": percent_of_normal,
            "range_category": range_category,
            "severity": severity,
            "movement_analysis": movement_analysis,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }
        
        return analysis_result
    
    def _generate_recommendations(self, 
                                test_type: str, 
                                range_category: str, 
                                severity: str,
                                movement_analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Generate personalized recommendations based on assessment results.
        
        Args:
            test_type: Type of test performed
            range_category: Category of range of motion
            severity: Severity of limitation
            movement_analysis: Analysis of movement patterns
            
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        # Base recommendations by test type and severity
        if "flexion" in test_type:
            if severity == "none":
                recommendations.append({
                    "type": "exercise",
                    "title": "Maintain Flexibility",
                    "description": "Continue with regular forward bending exercises to maintain your excellent flexibility."
                })
            elif severity == "mild":
                recommendations.append({
                    "type": "exercise",
                    "title": "Cat-Cow Stretch",
                    "description": "Perform the cat-cow stretch daily to improve spine flexibility."
                })
                recommendations.append({
                    "type": "exercise",
                    "title": "Standing Forward Bend",
                    "description": "Practice standing forward bends, holding for 20-30 seconds."
                })
            else:  # moderate or severe
                recommendations.append({
                    "type": "consult",
                    "title": "Physical Therapy Assessment",
                    "description": "Consider consulting a physical therapist for a comprehensive assessment."
                })
                recommendations.append({
                    "type": "exercise",
                    "title": "Gentle Seated Forward Bends",
                    "description": "Start with gentle seated forward bends, focusing on proper form."
                })
        
        elif "extension" in test_type:
            if severity == "none":
                recommendations.append({
                    "type": "exercise",
                    "title": "Maintain Back Extension",
                    "description": "Continue with regular back extension exercises to maintain flexibility."
                })
            elif severity == "mild":
                recommendations.append({
                    "type": "exercise",
                    "title": "Prone Press-Ups",
                    "description": "Perform prone press-ups to improve back extension mobility."
                })
            else:  # moderate or severe
                recommendations.append({
                    "type": "consult",
                    "title": "Professional Assessment",
                    "description": "Consult with a healthcare professional to rule out underlying conditions."
                })
                recommendations.append({
                    "type": "exercise",
                    "title": "Gentle Extensions",
                    "description": "Start with gentle standing back extensions, 10 repetitions several times daily."
                })
        
        elif "lateral_flexion" in test_type:
            if severity == "none":
                recommendations.append({
                    "type": "exercise",
                    "title": "Maintain Side Bending",
                    "description": "Continue with regular side bending exercises to maintain flexibility."
                })
            else:
                side = "left" if "left" in test_type else "right"
                recommendations.append({
                    "type": "exercise",
                    "title": f"Side Bending Stretch ({side.title()})",
                    "description": f"Practice standing side bends to the {side}, holding for 15-20 seconds."
                })
        
        elif "rotation" in test_type:
            if severity == "none":
                recommendations.append({
                    "type": "exercise",
                    "title": "Maintain Rotation",
                    "description": "Continue with regular rotation exercises to maintain trunk mobility."
                })
            else:
                side = "left" if "left" in test_type else "right"
                recommendations.append({
                    "type": "exercise",
                    "title": f"Seated Rotations ({side.title()})",
                    "description": f"Perform seated rotations to the {side} side, 10-15 repetitions, 3 times daily."
                })
        
        # Add recommendations based on movement analysis
        if movement_analysis:
            smoothness = movement_analysis.get("smoothness")
            if smoothness == "jerky":
                recommendations.append({
                    "type": "technique",
                    "title": "Movement Control",
                    "description": "Practice slower, controlled movements to improve coordination and reduce jerkiness."
                })
            
            speed = movement_analysis.get("speed_category")
            if speed == "very_slow":
                recommendations.append({
                    "type": "technique",
                    "title": "Movement Confidence",
                    "description": "Work on building confidence in movement by gradually increasing speed while maintaining control."
                })
        
        # Add general recommendations
        recommendations.append({
            "type": "lifestyle",
            "title": "Stay Active",
            "description": "Maintain an active lifestyle with regular exercise including flexibility, strength, and cardiovascular activities."
        })
        
        if range_category in ["limited", "severely_limited"]:
            recommendations.append({
                "type": "follow_up",
                "title": "Follow-Up Assessment",
                "description": "Schedule a follow-up assessment in 4-6 weeks to track your progress."
            })
        
        return recommendations
    
    def generate_comparison(self, 
                           previous_data: Dict[str, Any], 
                           current_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comparison between previous and current assessments.
        
        Args:
            previous_data: Previous assessment data
            current_data: Current assessment data
            
        Returns:
            Dictionary with comparison results
        """
        if not previous_data or not current_data:
            return {
                "status": "error",
                "message": "Insufficient data for comparison"
            }
        
        prev_rom = previous_data.get("rom", 0)
        curr_rom = current_data.get("rom", 0)
        
        # Calculate changes
        absolute_change = curr_rom - prev_rom
        percent_change = (absolute_change / prev_rom * 100) if prev_rom > 0 else 0
        
        # Determine improvement status
        if absolute_change >= 5:
            status = "significant_improvement"
        elif absolute_change >= 2:
            status = "moderate_improvement"
        elif absolute_change > 0:
            status = "slight_improvement"
        elif absolute_change == 0:
            status = "no_change"
        elif absolute_change > -2:
            status = "slight_decrease"
        elif absolute_change > -5:
            status = "moderate_decrease"
        else:
            status = "significant_decrease"
        
        # Generate insight message
        if status in ["significant_improvement", "moderate_improvement"]:
            insight = "Your ROM has improved since your last assessment. Continue with your current exercises."
        elif status == "slight_improvement":
            insight = "You've shown some improvement. Consider increasing the intensity of your exercises."
        elif status == "no_change":
            insight = "Your ROM remains stable. Consider adjusting your approach for better results."
        else:  # decrease
            insight = "Your ROM has decreased. This may be due to reduced activity or other factors. Consider consulting a professional."
        
        comparison_result = {
            "test_type": current_data.get("test_type"),
            "previous_date": previous_data.get("timestamp", "unknown"),
            "current_date": current_data.get("timestamp"),
            "previous_rom": prev_rom,
            "current_rom": curr_rom,
            "absolute_change": absolute_change,
            "percent_change": percent_change,
            "status": status,
            "insight": insight
        }
        
        return comparison_result
    
    def create_report_data(self, 
                         assessment_data: Dict[str, Any], 
                         analysis_result: Dict[str, Any],
                         comparison_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create comprehensive report data.
        
        Args:
            assessment_data: Original assessment data
            analysis_result: Analysis results
            comparison_result: Optional comparison results
            
        Returns:
            Dictionary with complete report data
        """
        report_data = {
            "report_id": f"ROM-{int(time.time())}",
            "timestamp": datetime.now().isoformat(),
            "assessment": assessment_data,
            "analysis": analysis_result
        }
        
        if comparison_result:
            report_data["comparison"] = comparison_result
        
        return report_data
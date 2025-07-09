import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


class PoseVisualizer:
    """Utility class for visualizing pose keypoints and analysis results."""
    
    def __init__(self):
        # Define colors for different keypoints
        self.colors = {
            'head': (255, 0, 0),      # Red
            'shoulders': (0, 255, 0),  # Green
            'arms': (0, 0, 255),       # Blue
            'hips': (255, 255, 0),     # Yellow
            'legs': (255, 0, 255),     # Magenta
            'good': (0, 255, 0),       # Green for good posture
            'needs_improvement': (255, 165, 0),  # Orange for needs improvement
            'insufficient_data': (128, 128, 128)  # Gray for insufficient data
        }
        
        # Define connections between keypoints for skeleton drawing
        self.skeleton_connections = [
            # Head connections
            ('nose', 'left_eye'), ('nose', 'right_eye'),
            ('left_eye', 'left_ear'), ('right_eye', 'right_ear'),
            
            # Torso connections
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            
            # Arm connections
            ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
            
            # Leg connections
            ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'), ('right_knee', 'right_ankle')
        ]
    
    def draw_pose_on_image(self, image_path: str, keypoints: Dict, 
                          save_path: Optional[str] = None) -> np.ndarray:
        """
        Draw pose keypoints and skeleton on an image.
        
        Args:
            image_path: Path to the input image
            keypoints: Dictionary of keypoints with x, y, confidence
            save_path: Optional path to save the annotated image
            
        Returns:
            Annotated image as numpy array
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Draw skeleton connections
        for connection in self.skeleton_connections:
            start_kp, end_kp = connection
            if start_kp in keypoints and end_kp in keypoints:
                start_pos = (int(keypoints[start_kp]['x']), int(keypoints[start_kp]['y']))
                end_pos = (int(keypoints[end_kp]['x']), int(keypoints[end_kp]['y']))
                
                # Only draw if confidence is high enough
                if keypoints[start_kp]['confidence'] > 0.5 and keypoints[end_kp]['confidence'] > 0.5:
                    cv2.line(image, start_pos, end_pos, (255, 255, 255), 2)
        
        # Draw keypoints
        for kp_name, kp_data in keypoints.items():
            if kp_data['confidence'] > 0.3:  # Only draw if confident enough
                x, y = int(kp_data['x']), int(kp_data['y'])
                color = self._get_keypoint_color(kp_name)
                cv2.circle(image, (x, y), 5, color, -1)
                cv2.putText(image, kp_name, (x + 10, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Save if requested
        if save_path:
            cv2.imwrite(save_path, image)
            print(f"✅ Annotated image saved to {save_path}")
        
        return image
    
    def draw_posture_analysis(self, image_path: str, analysis: Dict, 
                            save_path: Optional[str] = None) -> np.ndarray:
        """
        Draw posture analysis results on an image.
        
        Args:
            image_path: Path to the input image
            analysis: Posture analysis results
            save_path: Optional path to save the annotated image
            
        Returns:
            Annotated image as numpy array
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Draw keypoints and skeleton
        if 'keypoints' in analysis:
            image = self.draw_pose_on_image(image_path, analysis['keypoints'])
        
        # Add analysis text
        self._add_analysis_text(image, analysis)
        
        # Save if requested
        if save_path:
            cv2.imwrite(save_path, image)
            print(f"✅ Analysis image saved to {save_path}")
        
        return image
    
    def _get_keypoint_color(self, keypoint_name: str) -> Tuple[int, int, int]:
        """Get color for a specific keypoint."""
        if 'nose' in keypoint_name or 'eye' in keypoint_name or 'ear' in keypoint_name:
            return self.colors['head']
        elif 'shoulder' in keypoint_name:
            return self.colors['shoulders']
        elif 'elbow' in keypoint_name or 'wrist' in keypoint_name:
            return self.colors['arms']
        elif 'hip' in keypoint_name:
            return self.colors['hips']
        elif 'knee' in keypoint_name or 'ankle' in keypoint_name:
            return self.colors['legs']
        else:
            return (255, 255, 255)  # White for unknown keypoints
    
    def _add_analysis_text(self, image: np.ndarray, analysis: Dict):
        """Add analysis text to the image."""
        height, width = image.shape[:2]
        
        # Create text overlay
        text_lines = [
            f"Overall Score: {analysis.get('overall_score', 0):.2f}",
            f"Head Position: {analysis.get('head_position', {}).get('status', 'unknown')}",
            f"Shoulder Alignment: {analysis.get('shoulder_alignment', {}).get('status', 'unknown')}",
            f"Back Straightness: {analysis.get('back_straightness', {}).get('status', 'unknown')}"
        ]
        
        # Add recommendations
        recommendations = analysis.get('recommendations', [])
        if recommendations:
            text_lines.append("")
            text_lines.append("Recommendations:")
            for rec in recommendations[:3]:  # Show first 3 recommendations
                text_lines.append(f"• {rec}")
        
        # Draw text background
        text_height = len(text_lines) * 25 + 20
        cv2.rectangle(image, (10, 10), (400, text_height), (0, 0, 0), -1)
        cv2.rectangle(image, (10, 10), (400, text_height), (255, 255, 255), 2)
        
        # Draw text
        for i, line in enumerate(text_lines):
            y_pos = 30 + i * 25
            color = (255, 255, 255)  # White text
            
            # Color code the status
            if 'good' in line.lower():
                color = self.colors['good']
            elif 'needs_improvement' in line.lower():
                color = self.colors['needs_improvement']
            elif 'insufficient_data' in line.lower():
                color = self.colors['insufficient_data']
            
            cv2.putText(image, line, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)


def calculate_angle(point1: Tuple[float, float], 
                   point2: Tuple[float, float], 
                   point3: Tuple[float, float]) -> float:
    """
    Calculate the angle between three points.
    
    Args:
        point1: First point (x, y)
        point2: Middle point (x, y)
        point3: Third point (x, y)
        
    Returns:
        Angle in degrees
    """
    a = np.array(point1)
    b = np.array(point2)
    c = np.array(point3)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)


def calculate_distance(point1: Tuple[float, float], 
                      point2: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        point1: First point (x, y)
        point2: Second point (x, y)
        
    Returns:
        Distance between points
    """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def get_keypoint_coordinates(keypoints: Dict, keypoint_name: str) -> Tuple[float, float]:
    """
    Get coordinates for a specific keypoint.
    
    Args:
        keypoints: Dictionary of keypoints
        keypoint_name: Name of the keypoint
        
    Returns:
        Tuple of (x, y) coordinates
    """
    if keypoint_name in keypoints:
        return (keypoints[keypoint_name]['x'], keypoints[keypoint_name]['y'])
    else:
        raise ValueError(f"Keypoint {keypoint_name} not found in keypoints dictionary") 
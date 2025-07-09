import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
import os


class PostureAnalyzer:
    def __init__(self, model_path: str = "models/yolopose_v1.pt"):
        """
        Initialize the posture analyzer with your trained YOLO model.
        
        Args:
            model_path: Path to your trained YOLO pose model
        """
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the trained YOLO pose model."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found at {self.model_path}")
            
            self.model = YOLO(self.model_path)
            print(f"✅ YOLO model loaded successfully from {self.model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def detect_pose(self, image_path: str) -> Dict:
        """
        Detect pose in an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing pose detection results
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        try:
            # Run inference
            results = self.model(image_path)
            
            # Extract keypoints
            keypoints = self._extract_keypoints(results)
            
            return {
                "success": True,
                "keypoints": keypoints,
                "image_path": image_path,
                "num_people": len(keypoints)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "image_path": image_path
            }
    
    def analyze_posture(self, image_path: str) -> Dict:
        """
        Get pose keypoints from detected pose.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing pose keypoints
        """
        pose_result = self.detect_pose(image_path)
        
        if not pose_result["success"]:
            return pose_result
        
        # Get keypoints for each detected person
        pose_data = []
        for i, keypoints in enumerate(pose_result["keypoints"]):
            keypoint_data = self._get_keypoints_only(keypoints)
            keypoint_data["person_id"] = i
            pose_data.append(keypoint_data)
        
        return {
            "success": True,
            "pose_data": pose_data,
            "image_path": image_path,
            "num_people": len(pose_data)
        }
    
    def _extract_keypoints(self, results) -> List[np.ndarray]:
        """Extract keypoints from YOLO results."""
        keypoints_list = []
        
        for result in results:
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                keypoints = result.keypoints.data.cpu().numpy()
                for kp in keypoints:
                    keypoints_list.append(kp)
        
        return keypoints_list
    
    def _get_keypoints_only(self, keypoints: np.ndarray) -> Dict:
        """
        Extract keypoints for a single person.
        
        Args:
            keypoints: Array of keypoints [x, y, confidence]
            
        Returns:
            Dictionary with keypoint data
        """
        # COCO2017 keypoint format (17 keypoints)
        keypoint_names = [
            "nose",           # 0
            "left_eye",       # 1
            "right_eye",      # 2
            "left_ear",       # 3
            "right_ear",      # 4
            "left_shoulder",  # 5
            "right_shoulder", # 6
            "left_elbow",     # 7
            "right_elbow",    # 8
            "left_wrist",     # 9
            "right_wrist",    # 10
            "left_hip",       # 11
            "right_hip",      # 12
            "left_knee",      # 13
            "right_knee",     # 14
            "left_ankle",     # 15
            "right_ankle"     # 16
        ]
        
        # Create keypoint dictionary
        kp_dict = {}
        for i, name in enumerate(keypoint_names):
            if i < len(keypoints):
                kp_dict[name] = {
                    "x": float(keypoints[i][0]),
                    "y": float(keypoints[i][1]),
                    "confidence": float(keypoints[i][2])
                }
        
        return {
            "keypoints": kp_dict,
            "num_keypoints": len([kp for kp in kp_dict.values() if kp["confidence"] > 0.1])
        }
    
    def _analyze_single_posture(self, keypoints: np.ndarray) -> Dict:
        """
        Analyze posture for a single person.
        
        Args:
            keypoints: Array of keypoints [x, y, confidence]
            
        Returns:
            Dictionary with posture analysis results
        """
        # COCO2017 keypoint format (17 keypoints)
        keypoint_names = [
            "nose",           # 0
            "left_eye",       # 1
            "right_eye",      # 2
            "left_ear",       # 3
            "right_ear",      # 4
            "left_shoulder",  # 5
            "right_shoulder", # 6
            "left_elbow",     # 7
            "right_elbow",    # 8
            "left_wrist",     # 9
            "right_wrist",    # 10
            "left_hip",       # 11
            "right_hip",      # 12
            "left_knee",      # 13
            "right_knee",     # 14
            "left_ankle",     # 15
            "right_ankle"     # 16
        ]
        
        # Create keypoint dictionary
        kp_dict = {}
        for i, name in enumerate(keypoint_names):
            if i < len(keypoints):
                kp_dict[name] = {
                    "x": keypoints[i][0],
                    "y": keypoints[i][1],
                    "confidence": keypoints[i][2]
                }
        
        # Analyze posture components
        analysis = {
            "keypoints": kp_dict,
            "head_position": self._analyze_head_position(kp_dict),
            "shoulder_alignment": self._analyze_shoulder_alignment(kp_dict),
            "back_straightness": self._analyze_back_straightness(kp_dict),
            "overall_score": 0.0,
            "recommendations": []
        }
        
        # Calculate overall score
        scores = []
        if analysis["head_position"]["score"] is not None:
            scores.append(analysis["head_position"]["score"])
        if analysis["shoulder_alignment"]["score"] is not None:
            scores.append(analysis["shoulder_alignment"]["score"])
        if analysis["back_straightness"]["score"] is not None:
            scores.append(analysis["back_straightness"]["score"])
        
        if scores:
            analysis["overall_score"] = np.mean(scores)
        
        # Generate recommendations
        analysis["recommendations"] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _analyze_head_position(self, kp_dict: Dict) -> Dict:
        """Analyze head position relative to shoulders."""
        if "left_shoulder" not in kp_dict or "right_shoulder" not in kp_dict:
            return {"score": None, "status": "insufficient_data"}
        
        # Simple analysis: check if head is centered over shoulders
        left_shoulder = kp_dict["left_shoulder"]
        right_shoulder = kp_dict["right_shoulder"]
        
        if "nose" in kp_dict:
            nose = kp_dict["nose"]
            shoulder_center_x = (left_shoulder["x"] + right_shoulder["x"]) / 2
            head_offset = abs(nose["x"] - shoulder_center_x)
            
            # Score based on head alignment (lower offset = better)
            max_offset = abs(left_shoulder["x"] - right_shoulder["x"]) * 0.3
            score = max(0, 1 - (head_offset / max_offset)) if max_offset > 0 else 0
            
            return {
                "score": score,
                "status": "good" if score > 0.7 else "needs_improvement",
                "head_offset": head_offset
            }
        
        return {"score": None, "status": "insufficient_data"}
    
    def _analyze_shoulder_alignment(self, kp_dict: Dict) -> Dict:
        """Analyze shoulder alignment."""
        if "left_shoulder" not in kp_dict or "right_shoulder" not in kp_dict:
            return {"score": None, "status": "insufficient_data"}
        
        left_shoulder = kp_dict["left_shoulder"]
        right_shoulder = kp_dict["right_shoulder"]
        
        # Check if shoulders are level
        height_diff = abs(left_shoulder["y"] - right_shoulder["y"])
        shoulder_width = abs(left_shoulder["x"] - right_shoulder["x"])
        
        # Score based on shoulder levelness (lower height diff = better)
        max_height_diff = shoulder_width * 0.1
        score = max(0, 1 - (height_diff / max_height_diff)) if max_height_diff > 0 else 0
        
        return {
            "score": score,
            "status": "good" if score > 0.7 else "needs_improvement",
            "height_difference": height_diff
        }
    
    def _analyze_back_straightness(self, kp_dict: Dict) -> Dict:
        """Analyze back straightness using shoulder and hip alignment."""
        if not all(kp in kp_dict for kp in ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]):
            return {"score": None, "status": "insufficient_data"}
        
        left_shoulder = kp_dict["left_shoulder"]
        right_shoulder = kp_dict["right_shoulder"]
        left_hip = kp_dict["left_hip"]
        right_hip = kp_dict["right_hip"]
        
        # Calculate shoulder and hip angles
        shoulder_angle = np.arctan2(right_shoulder["y"] - left_shoulder["y"], 
                                   right_shoulder["x"] - left_shoulder["x"])
        hip_angle = np.arctan2(right_hip["y"] - left_hip["y"], 
                              right_hip["x"] - left_hip["x"])
        
        # Check alignment (shoulders and hips should be roughly parallel)
        angle_diff = abs(shoulder_angle - hip_angle)
        score = max(0, 1 - (angle_diff / (np.pi / 6)))  # Allow 30 degrees difference
        
        return {
            "score": score,
            "status": "good" if score > 0.7 else "needs_improvement",
            "angle_difference": angle_diff
        }
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate posture improvement recommendations."""
        recommendations = []
        
        if analysis["head_position"]["status"] == "needs_improvement":
            recommendations.append("Keep your head centered over your shoulders")
        
        if analysis["shoulder_alignment"]["status"] == "needs_improvement":
            recommendations.append("Level your shoulders - one shoulder appears higher than the other")
        
        if analysis["back_straightness"]["status"] == "needs_improvement":
            recommendations.append("Straighten your back - maintain natural spine alignment")
        
        if not recommendations:
            recommendations.append("Great posture! Keep it up!")
        
        return recommendations 
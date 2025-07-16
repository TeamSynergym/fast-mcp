import cv2 
import numpy as np
from typing import Dict, List, Tuple, Optional 
from PIL import Image, ImageDraw, ImageFont 
import os


class PoseVisualizer:
    """자세 키포인트 및 분석 결과를 시각화하기 위한 유틸리티 클래스."""
    
    def __init__(self):
        # 키포인트에 대한 색상 정의 (BGR 형식)
        self.colors = {
            'head': (0, 0, 255),       # 빨간색
            'shoulders': (0, 255, 0),  # 초록색
            'arms': (255, 0, 0),       # 파란색
            'hips': (0, 255, 255),     # 노란색
            'legs': (255, 0, 255),     # 마젠타
            'skeleton': (255, 255, 255) # 스켈레톤 (흰색)
        }
        
        # 스켈레톤을 그리기 위한 키포인트 연결 정의
        self.skeleton_connections = [
            # 머리 연결
            ('nose', 'left_eye'), ('nose', 'right_eye'),
            ('left_eye', 'left_ear'), ('right_eye', 'right_ear'),
            
            # 몸통 연결
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            
            # 팔 연결
            ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
            
            # 다리 연결
            ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'), ('right_knee', 'right_ankle')
        ]
        
        # 신뢰도 임계값
        self.confidence_threshold = 0.3
    
    @staticmethod
    def load_image(image_path: str):
        """이미지를 로드하고 오류 처리를 개선"""
        try:
            # Check if file exists first
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Check if file is empty
            if os.path.getsize(image_path) == 0:
                raise ValueError(f"Image file is empty: {image_path}")
            
            # Try to read the image
            image = cv2.imread(image_path)
            if image is None:
                # Try different approaches to identify the issue
                import imghdr
                image_type = imghdr.what(image_path)
                if image_type is None:
                    raise ValueError(f"File is not a valid image format: {image_path}")
                else:
                    raise ValueError(f"OpenCV cannot read {image_type} image: {image_path}")
            
            return image
        except Exception as e:
            print(f"❌ Error loading image: {e}")
            return None
    
    def _is_valid_keypoint(self, keypoint_data: Dict) -> bool:
        """키포인트가 유효한지 확인 (좌표와 신뢰도 체크)"""
        if not keypoint_data:
            return False
        
        x, y = keypoint_data.get('x', 0), keypoint_data.get('y', 0)
        confidence = keypoint_data.get('confidence', 0)
        
        # 좌표가 0이 아니고 신뢰도가 임계값 이상인 경우만 유효
        return x > 0 and y > 0 and confidence >= self.confidence_threshold
    
    def _draw_skeleton_and_keypoints(self, image: np.ndarray, keypoints: Dict) -> np.ndarray:
        """스켈레톤과 키포인트를 이미지에 그립니다."""
        height, width = image.shape[:2]
        
        # 스켈레톤 연결선 그리기
        for start_kp, end_kp in self.skeleton_connections:
            if (start_kp in keypoints and end_kp in keypoints and 
                self._is_valid_keypoint(keypoints[start_kp]) and 
                self._is_valid_keypoint(keypoints[end_kp])):
                
                # get_keypoint_coordinates 함수 대신 직접 좌표 추출
                start_point = (keypoints[start_kp]['x'], keypoints[start_kp]['y'])
                end_point = (keypoints[end_kp]['x'], keypoints[end_kp]['y'])
                
                # 이미지 경계 내에 있는지 확인
                if (0 <= start_point[0] < width and 0 <= start_point[1] < height and
                    0 <= end_point[0] < width and 0 <= end_point[1] < height):
                    
                    cv2.line(image, 
                            (int(start_point[0]), int(start_point[1])), 
                            (int(end_point[0]), int(end_point[1])), 
                            self.colors['skeleton'], 2)

        # 키포인트 그리기
        for kp_name, kp_data in keypoints.items():
            if self._is_valid_keypoint(kp_data):
                x, y = int(kp_data['x']), int(kp_data['y'])
                confidence = kp_data.get('confidence', 1.0)
                
                # 이미지 경계 내에 있는지 확인
                if 0 <= x < width and 0 <= y < height:
                    color = self._get_keypoint_color(kp_name)
                    
                    # 신뢰도에 따라 원의 크기 조정
                    radius = int(5 * confidence) if confidence < 1.0 else 5
                    cv2.circle(image, (x, y), radius, color, -1)
                    
                    # 신뢰도가 낮은 경우 테두리 추가
                    if confidence < 0.7:
                        cv2.circle(image, (x, y), radius + 1, (0, 0, 0), 1)

        return image

    def draw_pose_on_image(self, image_np: np.ndarray, analysis: Dict, 
                          save_path: Optional[str] = None) -> np.ndarray:
        """
        이미지에 자세 분석 결과를 그립니다.
        """
        # 입력 유효성 검사
        if image_np is None or image_np.size == 0:
            raise ValueError("입력 이미지가 비어있거나 None입니다.")
        
        # RGB 이미지를 BGR로 변환 (OpenCV 처리를 위해)
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image = image_np.copy()
        
        if 'keypoints' in analysis and analysis['keypoints']:
            image = self._draw_skeleton_and_keypoints(image, analysis['keypoints'])
        else:
            print("⚠️ 경고: 시각화를 위한 키포인트가 없습니다.")
        
        # 분석 텍스트 추가
        try:
            self._add_analysis_text(image, analysis)
        except Exception as e:
            print(f"❌ 분석 텍스트 추가 중 오류 발생: {e}")
        
        # 요청된 경우 저장 (BGR 형식으로 저장)
        if save_path:
            try:
                # 디렉토리가 존재하지 않으면 생성
                save_dir = os.path.dirname(save_path)
                if save_dir and not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                
                success = cv2.imwrite(save_path, image)
                if success:
                    print(f"✅ 분석 이미지가 {save_path}에 저장되었습니다.")
                else:
                    print(f"❌ 이미지 저장 실패: {save_path}")
            except Exception as e:
                print(f"❌ 이미지 저장 중 오류 발생: {e}")
        
        # 반환 전에 BGR을 RGB로 변환 (PIL/Streamlit 표시를 위해)
        if len(image.shape) == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            return image
    
    def _get_keypoint_color(self, keypoint_name: str) -> Tuple[int, int, int]:
        """특정 키포인트에 대한 색상을 가져옵니다."""
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
            return (255, 255, 255)  # 알 수 없는 키포인트는 흰색
    
    def _add_analysis_text(self, image: np.ndarray, analysis: Dict) -> None:
        """이미지에 분석 텍스트를 추가합니다."""
        height, width = image.shape[:2]
        
        # 기본 텍스트 설정
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)  # 흰색 텍스트
        thickness = 2
        
        y_offset = 30
        
        # 점수 정보 표시
        if 'scores' in analysis and analysis['scores']:
            scores = analysis['scores']
            for score_name, score_value in scores.items():
                if score_value is not None:
                    text = f"{score_name}: {score_value}"
                    cv2.putText(image, text, (10, y_offset), font, font_scale, color, thickness)
                    y_offset += 25
    
        # 키포인트 개수 표시
        if 'num_keypoints' in analysis:
            text = f"Keypoints: {analysis['num_keypoints']}/17"
            cv2.putText(image, text, (10, y_offset), font, font_scale, color, thickness)
            y_offset += 25
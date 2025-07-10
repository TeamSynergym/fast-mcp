import cv2 
import numpy as np
from typing import Dict, List, Tuple, Optional 
from PIL import Image, ImageDraw, ImageFont 


class PoseVisualizer:
    """자세 키포인트 및 분석 결과를 시각화하기 위한 유틸리티 클래스."""
    
    def __init__(self):
        # 키포인트에 대한 색상 정의
        self.colors = {
            'head': (255, 0, 0),       # 빨간색
            'shoulders': (0, 255, 0),  # 초록색
            'arms': (0, 0, 255),       # 파란색
            'hips': (255, 255, 0),     # 노란색
            'legs': (255, 0, 255),     # 마젠타
            
            'good': (0, 255, 0),       # 좋은 자세 (초록색)
            'needs_improvement': (255, 165, 0), # 개선 필요 (주황색)
            'insufficient_data': (128, 128, 128) # 데이터 부족 (회색)
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
    

    def draw_pose_on_image(self, image_np: np.ndarray, keypoints: Dict, 
                             save_path: Optional[str] = None) -> np.ndarray:
        """
        이미지에 자세 키포인트와 스켈레톤을 그립니다.
        
        Args:
            image_np: NumPy 배열 형식의 입력 이미지 (H, W, 3).
                      PIL에서 온 경우 RGB 형식이어야 하며, OpenCV를 위해 BGR로 변환됩니다.
            keypoints: x, y, confidence를 포함하는 키포인트 딕셔너리
            save_path: 주석이 추가된 이미지를 저장할 선택적 경로
            
        Returns:
            NumPy 배열 형식의 주석이 추가된 이미지 (BGR 형식)
        """
        # 이미지가 RGB인 경우 BGR로 변환 (OpenCV 기본 형식)
        if image_np.ndim == 3 and image_np.shape[2] == 3 and image_np.dtype == np.uint8:
            image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image = image_np.copy() # 쓰기 가능한 복사본


        # 스켈레톤 연결선 그리기
        for connection in self.skeleton_connections:
            start_kp, end_kp = connection
            if start_kp in keypoints and end_kp in keypoints:
                start_pos = (int(keypoints[start_kp]['x']), int(keypoints[start_kp]['y']))
                end_pos = (int(keypoints[end_kp]['x']), int(keypoints[end_kp]['y']))
                
                # 신뢰도가 충분히 높을 때만 그리기
                if keypoints[start_kp]['confidence'] > 0.5 and keypoints[end_kp]['confidence'] > 0.5:
                    cv2.line(image, start_pos, end_pos, (255, 255, 255), 2) # 흰색 선
        
        # 키포인트 그리기
        for kp_name, kp_data in keypoints.items():
            if kp_data['confidence'] > 0.3:  # 신뢰도가 충분히 높을 때만 그리기
                x, y = int(kp_data['x']), int(kp_data['y'])
                color = self._get_keypoint_color(kp_name)
                cv2.circle(image, (x, y), 5, color, -1) # 키포인트 원 그리기
                cv2.putText(image, kp_name, (x + 10, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA) # 키포인트 이름 텍스트
        
        # 요청된 경우 저장
        if save_path:
            cv2.imwrite(save_path, image)
            print(f"✅ 주석이 추가된 이미지가 {save_path}에 저장되었습니다.")
        
        return image
    

    def draw_posture_analysis(self, image_np: np.ndarray, analysis: Dict, 
                                 save_path: Optional[str] = None) -> np.ndarray:
        """
        이미지에 자세 분석 결과를 그립니다.
        
        Args:
            image_np: NumPy 배열 형식의 입력 이미지 (H, W, 3).
                      PIL에서 온 경우 RGB 형식이어야 합니다.
            analysis: 자세 분석 결과 딕셔너리.
                      'keypoints' 딕셔너리를 포함해야 합니다.
            save_path: 주석이 추가된 이미지를 저장할 선택적 경로.
            
        Returns:
            NumPy 배열 형식의 주석이 추가된 이미지 (BGR 형식).
        """
        # 키포인트와 스켈레톤을 그립니다.
        if 'keypoints' in analysis:
            image = self.draw_pose_on_image(image_np, analysis['keypoints'])
        else:
            # 분석 딕셔너리에 키포인트가 없는 경우, 입력 이미지_np의 복사본으로 작업합니다.
            if image_np.ndim == 3 and image_np.shape[2] == 3:
                image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            else:
                image = image_np.copy()
            print("경고: 시각화를 위한 분석 딕셔너리에서 'keypoints'를 찾을 수 없습니다.")
        
        # 분석 텍스트 추가
        self._add_analysis_text(image, analysis)
        
        # 요청된 경우 저장
        if save_path:
            cv2.imwrite(save_path, image)
            print(f"✅ 분석 이미지가 {save_path}에 저장되었습니다.")
        
        return image
    
    def _get_keypoint_color(self, keypoint_name: str) -> Tuple[int, int, int]:
        """특정 키포인트에 대한 색상 (BGR 형식)을 가져옵니다."""
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
    
    def _add_analysis_text(self, image: np.ndarray, analysis: Dict):
        """이미지에 분석 텍스트를 추가합니다. 이미지는 BGR 형식이어야 합니다."""
        height, width = image.shape[:2]
        
        # 'analysis' 딕셔너리에서 점수, 피드백, 측정값 가져오기
        # 여기서 'analysis' 딕셔너리는 단일 사람의 분석 결과여야 하며,
        # 'scores', 'feedback', 'measurements'를 포함합니다.
        scores = analysis.get('scores', {})
        feedback = analysis.get('feedback', {})
        measurements = analysis.get('measurements', {})

        # 표시 이름 및 단위 매핑
        measurement_map = {
            '어깨score': ('shoulder_tilt_angle', '°', '어깨 기울기'),
            '골반틀어짐score': ('hip_tilt_angle', '°', '골반 틀어짐'),
            '척추휨score': ('torso_tilt_angle', '°', '척추 휨'),
            '척추굽음score': ('back_angle', '°', '척추 굽음'),
            '거북목score': ('neck_forward_angle', '°', '거북목 각도')
        }

        # 점수에 따라 등급과 이모지를 반환하는 헬퍼 함수
        def get_grade_and_emoji(score):
            if score is None: return "분석불가", "❓"
            if score >= 90: return "매우 좋음", "✅"
            if score >= 70: return "양호", "✔️"
            if score >= 40: return "주의", "🟡"
            return "나쁨", "🔴"

        text_lines = []
        text_lines.append("--- 자세 분석 ---")

        # 점수 추가
        for score_name, score_value in scores.items():
            if score_value is not None:
                measurement_key, unit, display_name = measurement_map.get(score_name, (None, '', score_name))
                raw_value = measurements.get(measurement_key)
                grade, emoji = get_grade_and_emoji(score_value)

                display_text = ""
                if raw_value is not None:
                    if score_name == '척추굽음score':
                        # 척추 굽음 각도는 측정값이 수직선과의 편차이므로, 직관적인 굽은 각도를 계산하여 표시
                        deviation = raw_value 
                        display_text = f"(측정된 편차: {deviation:.1f}{unit})"
                    else:
                        display_text = f"(측정값: {raw_value:.1f}{unit})"
                
                text_lines.append(f"{emoji} {display_name}: {score_value}점 {display_text} - {grade}")
        
        # 피드백 추가
        text_lines.append("") # 빈 줄 추가
        text_lines.append("--- 피드백 ---")
        if feedback:
            for issue, message in feedback.items():
                if issue == 'overall': text_lines.append(f"👍 종합: {message}")
                elif issue == 'error': text_lines.append(f"❌ 오류: {message}")
                else: text_lines.append(f"⚠️ {issue.replace('_', ' ').title()}: {message}") # 이슈 이름을 보기 좋게 변환
        else:
            text_lines.append("ℹ️ 특별한 피드백이 없습니다.")
            
        # 텍스트 배경 치수 결정
        max_text_width = 0
        line_height = 25 # 한 줄의 높이
        for line in text_lines:
            (w, h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1) # 텍스트 크기 가져오기
            max_text_width = max(max_text_width, w) # 최대 텍스트 너비 업데이트
        
        padding = 20 # 패딩
        start_x = 10 # 텍스트 시작 x 좌표
        start_y = 10 # 텍스트 시작 y 좌표
        box_width = min(max_text_width + 2 * padding, width - 20) # 화면을 벗어나지 않도록 너비 조정
        box_height = len(text_lines) * line_height + 2 * padding # 배경 상자 높이
        
        # 텍스트 배경 그리기
        cv2.rectangle(image, (start_x, start_y), (start_x + box_width, start_y + box_height), (0, 0, 0), -1) # 검은색 배경
        cv2.rectangle(image, (start_x, start_y), (start_x + box_width, start_y + box_height), (255, 255, 255), 2) # 흰색 테두리
        
        # 텍스트 그리기
        for i, line in enumerate(text_lines):
            y_pos = start_y + padding + i * line_height # 각 줄의 y 좌표
            color = (255, 255, 255)  # 기본 흰색 텍스트
            
            # 피드백/상태 키워드에 따라 색상 코드 지정
            if '매우 좋음' in line or '좋은 자세' in line or '✅' in line:
                color = self.colors['good']
            elif '양호' in line or '✔️' in line:
                color = (0, 255, 255) # '양호'는 시안색
            elif '주의' in line or '⚠️' in line:
                color = self.colors['needs_improvement']
            elif '나쁨' in line or '🔴' in line or '오류' in line:
                color = (0, 0, 255) # '나쁨' 또는 오류는 빨간색
            elif '분석불가' in line or '❓' in line:
                color = self.colors['insufficient_data']
            
            cv2.putText(image, line, (start_x + padding, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA) # 텍스트 추가


# 헬퍼 함수
def calculate_angle(point1: Tuple[float, float], 
                    point2: Tuple[float, float], 
                    point3: Tuple[float, float]) -> float:
    """세 점 사이의 각도를 계산합니다."""
    a = np.array(point1)
    b = np.array(point2)
    c = np.array(point3)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0)) # 코사인 값 클리핑
    
    return np.degrees(angle) # 각도를 도로 변환하여 반환


def calculate_distance(point1: Tuple[float, float], 
                       point2: Tuple[float, float]) -> float:
    """두 점 사이의 유클리드 거리를 계산합니다."""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def get_keypoint_coordinates(keypoints: Dict, keypoint_name: str) -> Tuple[float, float]:
    """키포인트 딕셔너리에서 특정 키포인트의 좌표를 가져옵니다."""
    if keypoint_name in keypoints:
        return (keypoints[keypoint_name]['x'], keypoints[keypoint_name]['y'])
    else:
        # 원하는 동작에 따라 (0,0)을 반환하거나 오류를 발생시킬 수 있습니다.
        # 중요한 키포인트의 경우 오류를 발생시키는 것이 일반적으로 더 안전합니다.
        # 시각화의 경우, 신뢰도가 낮으면 (0,0)은 아무것도 그리지 않을 수 있습니다.
        return (0.0, 0.0) # 또는 ValueError(f"키포인트 {keypoint_name}를 찾을 수 없습니다...") 발생
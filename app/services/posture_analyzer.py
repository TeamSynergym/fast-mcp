import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, Optional
import os


class PostureAnalyzer:
    def __init__(self, model_path: str = "models/yolopose_v1.pt"):
        """
        자세 분석기를 학습된 YOLO 모델로 초기화합니다.
        
        Args:
            model_path: 학습된 YOLO 포즈 모델의 경로
        """
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """학습된 YOLO 포즈 모델을 로드합니다."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"모델을 {self.model_path}에서 찾을 수 없습니다.")
            
            self.model = YOLO(self.model_path)
            print(f"✅ YOLO 모델이 {self.model_path}에서 성공적으로 로드되었습니다.")
        except Exception as e:
            print(f"❌ 모델 로드 중 오류 발생: {e}")
            raise
    
    def detect_pose(self, image_path: str) -> Dict:
        """
        이미지에서 자세를 감지합니다.
        
        변경 사항: 여러 사람이 감지될 경우, 가장 높은 전체 키포인트 신뢰도를 가진
                  한 사람의 키포인트만 추출하여 반환합니다.
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            자세 감지 결과를 포함하는 딕셔너리.
            'keypoints'는 가장 신뢰도 높은 한 사람의 키포인트 배열을 포함하는 리스트입니다.
        """
        if self.model is None:
            raise ValueError("모델이 로드되지 않았습니다.")
        
        try:
            # 추론 실행
            results = self.model(image_path)
            
            highest_confidence_kp = None
            max_avg_confidence = -1

            for result in results:
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    for kp_data_for_person in result.keypoints.data.cpu().numpy():
                        # 각 사람의 키포인트 데이터에서 신뢰도만 추출
                        confidences = kp_data_for_person[:, 2]
                        
                        # 0.1보다 큰 신뢰도 값을 가진 키포인트만 평균에 포함
                        confident_kps = confidences[confidences > 0.1]
                        
                        if len(confident_kps) > 0:
                            avg_confidence = np.mean(confident_kps)
                        else:
                            avg_confidence = 0 # 신뢰도 있는 키포인트가 없으면 평균 0

                        # 현재 사람의 평균 신뢰도가 지금까지의 최대값보다 높으면 업데이트
                        if avg_confidence > max_avg_confidence:
                            max_avg_confidence = avg_confidence
                            highest_confidence_kp = kp_data_for_person
            
            # keypoints가 비어있으면 (사람이 감지되지 않으면) num_people = 0
            num_people_detected = 1 if highest_confidence_kp is not None else 0 

            return {
                "success": True,
                "keypoints": [highest_confidence_kp] if highest_confidence_kp is not None else [], # 단일 사람의 키포인트 배열을 리스트에 담아 반환
                "image_path": image_path,
                "num_people": num_people_detected
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "image_path": image_path
            }
    
    def analyze_posture(self, image_path: str, mode: str = 'front') -> Dict:
        """
        이미지 내 사람의 자세를 감지하고 분석합니다.
        
        Args:
            image_path: 이미지 파일 경로
            mode: 'front'는 정면 분석, 'side'는 측면 분석
            
        Returns:
            감지된 한 사람에 대한 자세 분석 결과를 포함하는 딕셔너리.
            분석에는 'keypoints', 'scores', 'feedback', 'measurements'가 포함됩니다.
        """
        pose_detection_result = self.detect_pose(image_path)
        
        if not pose_detection_result["success"]:
            return pose_detection_result # 감지 실패 시 오류 반환

        # 사람이 감지되지 않았으면 빈 결과 반환
        if not pose_detection_result["keypoints"]:
            return {
                "success": True,
                "pose_data": [],
                "image_path": image_path,
                "num_people": 0
            }
            
        all_pose_data = []
        keypoints_array_for_person = pose_detection_result["keypoints"][0] 
        
        # 상세 키포인트 딕셔너리 가져오기
        keypoints_dict = self._get_keypoints_only(keypoints_array_for_person)
        
        # 상세 자세 분석 수행
        analysis_results = self._analyze_single_posture(keypoints_array_for_person, mode=mode)
        
        # 이 사람에 대한 모든 정보 결합
        person_analysis = {
            "person_id": 0, # 한 명만 분석하므로 ID는 0으로 고정
            "keypoints": keypoints_dict["keypoints"], # 키포인트 딕셔너리
            "num_keypoints": keypoints_dict["num_keypoints"],
            "scores": analysis_results["scores"],
            "feedback": analysis_results["feedback"],
            "measurements": analysis_results["measurements"]
        }
        all_pose_data.append(person_analysis)
        
        return {
            "success": True,
            "pose_data": all_pose_data, # 단일 사람에 대한 분석 결과 리스트
            "image_path": image_path,
            "num_people": len(all_pose_data)
        }

    def _get_keypoints_only(self, keypoints: np.ndarray) -> Dict:
        """
        단일 사람의 키포인트를 추출하여 딕셔너리 형식으로 변환합니다.
        """
        keypoint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear", 
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", 
            "left_wrist", "right_wrist", "left_hip", "right_hip", 
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
        
        kp_dict = {}
        for i, name in enumerate(keypoint_names):
            if i < len(keypoints):
                kp_dict[name] = {
                    "x": float(keypoints[i][0]),
                    "y": float(keypoints[i][1]),
                    "confidence": float(keypoints[i][2])
                }
        
        # 신뢰도 0.1 이상인 키포인트의 개수를 계산합니다.
        num_confident_keypoints = sum(1 for kp in kp_dict.values() if kp["confidence"] > 0.1)
        
        return {
            "keypoints": kp_dict,
            "num_keypoints": num_confident_keypoints
        }

    def _calculate_angle(self, p1, p2, p3):
        """
        세 점 사이의 각도를 계산합니다.
        """
        # Ensure coordinates are extracted as 2-element arrays (x, y)
        p1_coords = np.array(p1, dtype=np.float32)[:2]
        p2_coords = np.array(p2, dtype=np.float32)[:2]
        p3_coords = np.array(p3, dtype=np.float32)[:2]
        
        # Check if any coordinate set is effectively zero (indicating undetected or invalid keypoint)
        if np.all(p1_coords == 0) or np.all(p2_coords == 0) or np.all(p3_coords == 0):
            return 180.0

        v1 = p1_coords - p2_coords
        v2 = p3_coords - p2_coords
        dot_product = np.dot(v1, v2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        
        if norm_product == 0: 
            return 180.0 # 영 벡터로 인한 나눗셈 방지
        
        # 아크코사인 계산 및 결과 클리핑 (-1.0, 1.0) 범위 유지
        angle_rad = np.arccos(np.clip(dot_product / norm_product, -1.0, 1.0))
        return np.degrees(angle_rad)

    def _calculate_balanced_score(self, value, grade_info):
        """
        이상적인 값으로부터의 편차를 기반으로 균형 잡힌 점수를 계산합니다.
        """
        ideal = grade_info['ideal']
        grades = grade_info['grades'] # (최대 편차, 현재 점수 최소값) 튜플 리스트
        deviation = abs(value - ideal) # 이상적인 값과의 절대 편차

        prev_dev, prev_score_max = 0, 100 # 이전 편차와 이전 점수 상한 초기화
        for max_dev, current_score_min in grades:
            if deviation <= max_dev:
                score_max, score_min = prev_score_max, current_score_min
                dev_range = max_dev - prev_dev # 현재 등급의 편차 범위
                if dev_range == 0: return score_min # 범위가 0이면 최소 점수 반환 (예: 정확히 이상적인 값)
                
                # 현재 등급 내에서 편차의 비율 계산
                ratio_in_grade = (deviation - prev_dev) / dev_range
                # 점수 계산: 이전 점수 상한에서 편차 비율에 따라 점수 감소
                score = score_max - (ratio_in_grade * (score_max - score_min))
                return max(0, round(score)) # 점수가 0 미만이 되지 않도록 하고 반올림
            prev_dev = max_dev
            prev_score_max = current_score_min
        return 0 # 모든 등급을 초과하면 0점

    def _analyze_single_posture(self, keypoints: np.ndarray, mode: str = 'front', config: Optional[Dict] = None) -> Dict:
        """
        단일 사람의 자세를 분석합니다.
        """
        keypoints_xy = keypoints[:, :2] # x, y 좌표만 추출

        if config is None:
            # 자세 분석을 위한 기본 설정값
            config = {
                'shoulder_tilt_feedback_threshold': 3.5, # 어깨 기울기 피드백 임계값
                'hip_tilt_feedback_threshold': 3.5,      # 골반 기울기 피드백 임계값
                'torso_tilt_feedback_threshold': 4.0,    # 몸통 기울기 피드백 임계값
                'neck_forward_angle_feedback_threshold': 15.0, # 거북목 각도 피드백 임계값
                'back_bend_feedback_threshold': 10.0,    # 척추 굽힘 피드백 임계값
                # 점수 계산을 위한 등급 정보: 'ideal'은 이상적인 값, 'grades'는 (최대 편차, 해당 점수 최소값) 리스트
                'shoulder_tilt_grades': {'ideal': 0, 'grades': [(1.5, 90), (3.5, 70), (8.5, 40), (20, 0)]},
                'hip_tilt_grades': {'ideal': 0, 'grades': [(1.5, 90), (3.5, 70), (8.5, 40), (20, 0)]},
                'torso_tilt_grades': {'ideal': 0, 'grades': [(2, 90), (4, 70), (9.5, 40), (22, 0)]},
                'neck_forward_angle_grades': {'ideal': 0, 'grades': [(8, 90), (15, 70), (25, 40), (35, 0)]},
                'back_bend_grades': {'ideal': 0, 'grades': [(4, 90), (10, 70), (20, 40), (30, 0)]}
            }

        # 키포인트 인덱스 정의 (OpenPose 17개 키포인트 순서)
        NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SHOULDER, R_SHOULDER, \
        L_ELBOW, R_ELBOW, L_WRIST, R_WRIST, L_HIP, R_HIP, \
        L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = range(17)

        feedback, scores, measurements = {}, {}, {}
        # 측정값 및 점수 딕셔너리 초기화
        for key in ['torso_tilt_angle', 'shoulder_tilt_angle', 'hip_tilt_angle', 'neck_forward_angle', 'back_angle']:
            measurements[key] = None
        for key in ['척추휨score', '어깨score', '골반틀어짐score', '거북목score', '척추굽음score']:
            scores[key] = None

        if mode == 'front':
            # 정면 분석에 필요한 키포인트
            l_shoulder = keypoints_xy[L_SHOULDER]
            r_shoulder = keypoints_xy[R_SHOULDER]
            l_hip = keypoints_xy[L_HIP]
            r_hip = keypoints_xy[R_HIP] 

            min_conf_front = 0.3 # 정면 분석 최소 신뢰도
            # 필요한 키포인트의 신뢰도 확인
            if not (keypoints[L_SHOULDER][2] > min_conf_front and
                    keypoints[R_SHOULDER][2] > min_conf_front and
                    keypoints[L_HIP][2] > min_conf_front and
                    keypoints[R_HIP][2] > min_conf_front):
                feedback['error'] = "정면 분석에 필요한 어깨 또는 골반 키포인트의 신뢰도가 낮습니다."
                return {"scores": scores, "feedback": feedback, "measurements": measurements}

            # 몸통 기울기 분석
            shoulder_center = (l_shoulder + r_shoulder) / 2
            hip_center = (l_hip + r_hip) / 2
            
            dx_torso = shoulder_center[0] - hip_center[0]
            dy_torso = shoulder_center[1] - hip_center[1]

            # 수직선에 대한 몸통의 기울기 각도 계산 (dx/dy 사용)
            if abs(dy_torso) < 1e-6: # 분모가 0에 가까울 때
                torso_tilt_angle = 90.0 if abs(dx_torso) > 0 else 0.0
            else:
                torso_tilt_angle = np.degrees(np.arctan(abs(dx_torso) / abs(dy_torso)))

            measurements['torso_tilt_angle'] = torso_tilt_angle
            scores['척추휨score'] = self._calculate_balanced_score(torso_tilt_angle, config['torso_tilt_grades'])
            if torso_tilt_angle > config['torso_tilt_feedback_threshold']:
                feedback['torso_tilt'] = f"몸통이 {torso_tilt_angle:.1f}° 옆으로 기울어 '주의'가 필요합니다."

            def get_tilt_angle(p1_coords, p2_coords):
                """두 점을 잇는 선의 수평에 대한 기울기 각도를 계산합니다."""
                if np.array_equal(p1_coords, p2_coords): return 0.0

                dx = p2_coords[0] - p1_coords[0]
                dy = p2_coords[1] - p1_coords[1] 

                angle_rad = np.arctan2(dy, dx) # 라디안 각도
                angle_deg = np.degrees(angle_rad) # 도 단위 각도

                tilt = abs(angle_deg % 180) # 0-180 범위로 조정
                if tilt > 90:
                    tilt = 180 - tilt # 0-90 범위로 조정
                    
                return abs(tilt) # 절대값 반환

            # 어깨 기울기 분석
            shoulder_tilt_angle = get_tilt_angle(l_shoulder, r_shoulder)
            measurements['shoulder_tilt_angle'] = shoulder_tilt_angle
            scores['어깨score'] = self._calculate_balanced_score(shoulder_tilt_angle, config['shoulder_tilt_grades'])
            if shoulder_tilt_angle > config['shoulder_tilt_feedback_threshold']:
                feedback['shoulder_tilt'] = f"어깨가 {shoulder_tilt_angle:.1f}° 기울어져 '주의'가 필요합니다."

            # 골반 기울기 분석
            hip_tilt_angle = get_tilt_angle(l_hip, r_hip)
            measurements['hip_tilt_angle'] = hip_tilt_angle
            scores['골반틀어짐score'] = self._calculate_balanced_score(hip_tilt_angle, config['hip_tilt_grades'])
            if hip_tilt_angle > config['hip_tilt_feedback_threshold']:
                feedback['hip_tilt'] = f"골반이 {hip_tilt_angle:.1f}° 기울어져 '주의'가 필요합니다."

        elif mode == 'side':
            min_conf_side_required = 0.2 # 측면 분석 최소 신뢰도

            def get_confident_kp(kp_idx1, kp_idx2):
                """두 키포인트 중 신뢰도가 더 높은 키포인트를 반환합니다. 둘 다 낮으면 None 반환."""
                kp1_conf = keypoints[kp_idx1][2]
                kp2_conf = keypoints[kp_idx2][2]

                if kp1_conf > min_conf_side_required and kp2_conf > min_conf_side_required:
                    return keypoints_xy[kp_idx1] if kp1_conf > kp2_conf else keypoints_xy[kp_idx2]
                elif kp1_conf > min_conf_side_required:
                    return keypoints_xy[kp_idx1]
                elif kp2_conf > min_conf_side_required:
                    return keypoints_xy[kp_idx2]
                else:
                    return None

            # 필요한 키포인트 추출 (좌우 중 더 신뢰도 높은 것 선택)
            ear_kp = get_confident_kp(L_EAR, R_EAR)
            shoulder_kp = get_confident_kp(L_SHOULDER, R_SHOULDER)
            hip_kp = get_confident_kp(L_HIP, R_HIP)
            
            # 무릎과 발목은 척추 굽힘 계산에는 보조적으로 사용하며, 없어도 전체 분석이 멈추지는 않음.
            knee_kp = get_confident_kp(L_KNEE, R_KNEE)
            ankle_kp = get_confident_kp(L_ANKLE, R_ANKLE)

            # 핵심 키포인트 누락 여부 확인 및 피드백
            missing_core_kp_feedback = []
            if ear_kp is None: missing_core_kp_feedback.append("귀")
            if shoulder_kp is None: missing_core_kp_feedback.append("어깨")
            if hip_kp is None: missing_core_kp_feedback.append("골반")

            if missing_core_kp_feedback:
                feedback['error'] = f"측면 분석에 필요한 다음 핵심 키포인트의 신뢰도가 낮습니다: {', '.join(missing_core_kp_feedback)}."
            
            # 척추 곧음은 어깨-골반 선이 수직선과 이루는 각도로 평가됩니다.
            # 앉거나 서 있는 자세 모두에 견고하게 적용되며 몸통 정렬에 중점을 둡니다.
            if shoulder_kp is not None and hip_kp is not None:
                dx_sh = shoulder_kp[0] - hip_kp[0] # 어깨와 골반의 x좌표 차이
                dy_sh = shoulder_kp[1] - hip_kp[1] # 어깨와 골반의 y좌표 차이
                
                # 수직선과의 각도: arctan(abs(dx)/abs(dy)). 0으로 나누는 경우 처리.
                # 완벽하게 수직인 선(곧은 등)은 dx_sh = 0 이므로 angle_sh_deg = 0.
                if abs(dy_sh) < 1e-6: # 선이 거의 수평에 가까워 분모가 0에 가까워지는 경우 방지
                    back_deviation = 90.0 if abs(dx_sh) > 0 else 0.0
                else:
                    back_deviation = np.degrees(np.arctan(abs(dx_sh) / abs(dy_sh)))
                
                measurements['back_angle'] = back_deviation # 수직선으로부터의 편차로 저장
                scores['척추굽음score'] = self._calculate_balanced_score(back_deviation, config['back_bend_grades'])
                
                # 수직선으로부터의 편차를 기반으로 피드백
                if back_deviation > config['back_bend_feedback_threshold']:
                    feedback['back_bend'] = f"등/허리가 {back_deviation:.1f}°(어깨-골반 선 정렬) 기울어져 '주의'가 필요합니다."
            else:
                scores['척추굽음score'] = 0
                feedback['back_bend_error'] = "척추 굽음 분석에 필요한 (어깨, 골반) 키포인트가 부족하여 계산할 수 없습니다."

            # 거북목 각도 분석
            if ear_kp is not None and shoulder_kp is not None:
                # 어깨 키포인트에서 수직으로 가상의 점을 만듭니다.
                shoulder_vertical = (shoulder_kp[0], shoulder_kp[1] + 10) 
                # 귀-어깨-가상 수직점 사이의 내부 각도를 계산합니다.
                internal_neck_angle = self._calculate_angle(ear_kp, shoulder_kp, shoulder_vertical)
                # 이 내부 각도를 이용하여 거북목 각도를 계산합니다 (180도에서 내부 각도 빼기).
                neck_forward_angle = 180 - internal_neck_angle

                measurements['neck_forward_angle'] = neck_forward_angle
                scores['거북목score'] = self._calculate_balanced_score(neck_forward_angle, config['neck_forward_angle_grades'])
                if neck_forward_angle > config['neck_forward_angle_feedback_threshold']:
                    feedback['head_forward'] = f"목이 앞으로 {neck_forward_angle:.1f}° 기울어져 '주의'가 필요합니다."
            else:
                scores['거북목score'] = 0
                feedback['neck_error'] = "거북목 분석에 필요한 (귀, 어깨) 키포인트가 부족하여 계산할 수 없습니다."
            

        else:
            feedback['error'] = f"'{mode}'는 유효한 모드가 아닙니다. 'front' 또는 'side'를 사용하세요."

        # 모든 피드백이 없고 오류도 없는 경우 긍정적인 전체 피드백 제공
        if not feedback and 'error' not in feedback and 'back_bend_error' not in feedback and 'neck_error' not in feedback:
            feedback['overall'] = "전체적으로 매우 좋은 자세를 유지하고 있습니다! 👍"

        return {"scores": scores, "feedback": feedback, "measurements": measurements}
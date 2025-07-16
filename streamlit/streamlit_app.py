import streamlit as st
import os
import sys 
import tempfile 
from PIL import Image 
import numpy as np 
import cv2 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 자세 분석 모듈 임포트
from app.services.posture_analyzer import PostureAnalyzer
from app.utils.pose_utils import PoseVisualizer

def main():
    # Streamlit 페이지 설정
    st.set_page_config(
        page_title="YOLO를 이용한 자세 분석",
        page_icon="🧍",
        layout="wide"
    )
    
    st.title("🧍 YOLO를 이용한 자세 분석")
    st.markdown("학습된 YOLO 모델을 사용하여 이미지에서 자세를 분석합니다.")
    
    # 사이드바에 설정 섹션 추가
    st.sidebar.header("⚙️ 설정")
    model_path = st.sidebar.text_input(
        "모델 경로",
        value="models/yolopose_v1.pt", 
        help="학습된 YOLO 포즈 모델의 경로" 
    )
    
    # 분석 모드 선택을 위한 라디오 버튼 추가 (정면/측면)
    analysis_mode = st.sidebar.radio(
        "분석 모드 선택", 
        ('front', 'side'), 
        help="'front'는 정면 보기 분석, 'side'는 시상면(측면) 보기 분석을 선택하세요." 
    )
    
    # 모델 파일 존재 여부 확인
    if not os.path.exists(model_path):
        st.error(f"❌ 모델을 찾을 수 없습니다: {model_path}") 
        st.info("models/ 디렉토리에 yolopose_v1.pt 모델을 넣어주세요.") 
        return 
    
    # 자세 분석기 및 시각화 도구 초기화
    try:
        analyzer = PostureAnalyzer(model_path=model_path)
        visualizer = PoseVisualizer()
        st.success("✅ YOLO 모델이 성공적으로 로드되었습니다!") 
    except Exception as e:
        st.error(f"❌ 모델 로드 중 오류 발생: {e}")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📸 이미지 업로드")
        
        # 파일 업로더 위젯
        uploaded_file = st.file_uploader(
            "이미지 파일 선택",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="자세를 분석할 사람의 이미지를 업로드하세요."
        )
        
        if uploaded_file is not None:
            # 새로운 파일이 업로드된 경우, 저장하고 세션 상태 업데이트
            if 'uploaded_file_id' not in st.session_state or st.session_state.uploaded_file_id != uploaded_file.file_id:
                st.subheader("원본 이미지")
                image = Image.open(uploaded_file) # 이미지 열기
                st.image(image, caption="업로드된 이미지", use_container_width=True) # 이미지 표시
                
                # 임시 파일로 저장 (분석에 사용)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    if image.mode in ('RGBA', 'LA', 'P'): # RGBA, LA, P 모드의 이미지를 RGB로 변환
                        image = image.convert('RGB')
                    image.save(tmp_file.name)
                    st.session_state.temp_image_path = tmp_file.name 
                
                # 새로운 이미지 업로드 시 이전 분석 결과 초기화
                st.session_state.pose_result = None
                st.session_state.uploaded_file_id = uploaded_file.file_id
                
            else: # 이미 처리/표시된 파일인 경우, 다시 표시
                st.subheader("원본 이미지")
                image = Image.open(uploaded_file)
                st.image(image, caption="업로드된 이미지", use_container_width=True)
                
            # '자세 분석' 버튼
            if st.button("🔍 자세 분석", type="primary"):
                if 'temp_image_path' in st.session_state and os.path.exists(st.session_state.temp_image_path):
                    with st.spinner("자세 분석 중..."):
                        try:
                            # 선택된 analysis_mode를 사용하여 자세 분석
                            analysis_result = analyzer.analyze_posture(st.session_state.temp_image_path, mode=analysis_mode)
                            
                            if analysis_result["success"]:
                                st.success(f"✅ {analysis_result['num_people']} 명 감지됨")
                                
                                 # --- Ollama 진단 메시지 생성 ---
                                for i, person_data in enumerate(analysis_result["pose_data"]):
                                    st.session_state.ollama_diagnosis = analyzer.generate_ollama_diagnosis(person_data, analysis_mode)
                                    person_data['ollama_diagnosis'] = st.session_state.ollama_diagnosis # 결과에 진단 메시지 추가
                                # --- Ollama 진단 메시지 생성 끝 ---
                                
                                st.session_state.pose_result = analysis_result # 분석 결과 세션 상태에 저장
                            else:
                                st.error(f"❌ 분석 실패: {analysis_result.get('error', '알 수 없는 오류')}")
                        except Exception as e:
                            st.error(f"❌ 분석 중 오류 발생: {e}") 
                else:
                    st.warning("먼저 이미지를 업로드해주세요.") 
    
    with col2:
        st.header("📊 분석 결과")
        
        if hasattr(st.session_state, 'pose_result') and st.session_state.pose_result:
            pose_result = st.session_state.pose_result
            
            if pose_result["success"] and "pose_data" in pose_result and pose_result["pose_data"]:
                for i, pose_data in enumerate(pose_result["pose_data"]): # 각 사람에 대한 데이터 반복
                    st.subheader(f"👤 사람 {i + 1}")
                    
                    if 'keypoints' not in pose_data or not pose_data['keypoints']:
                        st.warning("⚠️ 이 사람에 대한 키포인트 데이터가 없거나 비어 있습니다.")
                        continue

                    keypoints_dict = pose_data['keypoints']
                    num_keypoints = pose_data.get('num_keypoints', 0)

                    st.metric("감지된 키포인트", f"{num_keypoints}/17") # 감지된 키포인트 개수 표시
                    st.progress(num_keypoints / 17.0) # 키포인트 감지율 진행 바
                    
                    # 점수 표시
                    st.subheader("💯 자세 점수")
                    if 'scores' in pose_data and pose_data['scores']:
                        scores = pose_data['scores']
                        measurements = pose_data.get('measurements', {})

                        # 점수 이름과 표시 이름, 측정 키 매핑 정의
                        # 활성 모드에 따라 이 맵을 필터링
                        measurement_map = {
                            '어깨score': ('shoulder_tilt_angle', '°', '어깨 기울기'),
                            '골반틀어짐score': ('hip_tilt_angle', '°', '골반 틀어짐'),
                            '척추휨score': ('torso_tilt_angle', '°', '척추 휨'),
                            '척추굽음score': ('back_angle', '°', '척추 굽음'),
                            '거북목score': ('neck_forward_angle', '°', '거북목 각도')
                        }

                        # 점수에 따라 등급과 이모지 반환 함수
                        def get_grade_and_emoji(score):
                            if score is None: return "분석불가", "❓"
                            if score >= 90: return "매우 좋음", "✅"
                            if score >= 70: return "양호", "✔️"
                            if score >= 40: return "주의", "🟡"
                            return "나쁨", "🔴"

                        active_scores_found = False
                        # 선택된 모드와 관련된 점수만 표시
                        if analysis_mode == 'front':
                            display_score_names = ['척추휨score', '어깨score', '골반틀어짐score']
                        elif analysis_mode == 'side':
                            display_score_names = ['척추굽음score', '거북목score']
                        else:
                            display_score_names = list(scores.keys()) # 폴백: 모드를 알 수 없으면 모두 표시


                        for score_name in display_score_names:
                            score_value = scores.get(score_name)
                            if score_value is not None:
                                active_scores_found = True
                                measurement_key, unit, display_name = measurement_map.get(score_name, (None, '', score_name))
                                raw_value = measurements.get(measurement_key)
                                grade, emoji = get_grade_and_emoji(score_value)

                                display_text = ""
                                if raw_value is not None:
                                    if score_name == '척추굽음score':
                                        deviation = raw_value 
                                        display_text = f"(측정된 편차: {deviation:.1f}{unit})"
                                    else:
                                        display_text = f"(측정값: {raw_value:.1f}{unit})"
                                    
                                st.markdown(f"{emoji} **{display_name}**: {score_value:>3}점 {display_text} - {grade}")
                        
                        if not active_scores_found:
                            st.info(f"ℹ️ {analysis_mode.capitalize()} 모드에 해당하는 점수가 없습니다. 올바른 사진과 모드를 선택해주세요.")

                    else:
                        st.info("ℹ️ 점수 데이터가 없습니다.")

                    # 일반 피드백 표시
                    st.subheader("💬 일반 피드백")
                    if 'feedback' in pose_data and pose_data['feedback']:
                        feedback = pose_data['feedback']
                        # 모드에 따라 관련 피드백만 표시하도록 필터링
                        if analysis_mode == 'front':
                            feedback_keys_to_display = ['overall', 'error', 'torso_tilt', 'shoulder_tilt', 'hip_tilt']
                        elif analysis_mode == 'side':
                            feedback_keys_to_display = ['overall', 'error', 'head_forward', 'back_bend', 'back_bend_error', 'neck_error']
                        else:
                            feedback_keys_to_display = list(feedback.keys()) # 폴백: 모두 표시

                        feedback_shown = False
                        for issue_key in feedback_keys_to_display:
                            if issue_key in feedback:
                                message = feedback[issue_key]
                                if issue_key == 'overall': st.markdown(f"👍 **종합**: {message}")
                                elif issue_key.endswith('_error'): st.error(f"❌ **오류**: {message}") # 에러 메시지
                                else: st.warning(f"⚠️ **{issue_key.replace('_', ' ').title()}**: {message}") # 경고 메시지
                                feedback_shown = True
                        
                        if not feedback_shown:
                            st.info("ℹ️ 현재 분석 모드에 해당하는 피드백이 없습니다.")

                    else:
                        st.info("ℹ️ 피드백이 없습니다.")

                    # --- Ollama 진단 메시지 출력 ---
                    st.subheader("🤖 Ollama AI 진단")
                    if 'ollama_diagnosis' in pose_data and pose_data['ollama_diagnosis']:
                        st.markdown(pose_data['ollama_diagnosis'])
                    else:
                        st.info("Ollama AI 진단 메시지가 생성되지 않았습니다.")
                    # --- Ollama 진단 메시지 출력 끝 ---


                    # 키포인트 표시
                    st.subheader("📍 키포인트")
                    
                    keypoint_data = []
                    # 신뢰도 0.1 이상인 키포인트만 기본적으로 표시
                    for name, kp in keypoints_dict.items():
                        if kp.get('confidence', 0) > 0.1:
                            keypoint_data.append({
                                "Keypoint": name,
                                "X": f"{kp['x']:.1f}",
                                "Y": f"{kp['y']:.1f}",
                                "Confidence": f"{kp['confidence']:.3f}"
                            })
                    
                    if keypoint_data:
                        st.dataframe(keypoint_data, use_container_width=True) # 데이터프레임으로 표시
                    else:
                        st.warning("⚠️ 신뢰도 있는 키포인트가 감지되지 않았습니다.")
                    
                    # 모든 키포인트 (낮은 신뢰도 포함) 확장 가능 섹션
                    with st.expander("🔍 모든 키포인트 (낮은 신뢰도 포함)"):
                        all_keypoints = []
                        for name, kp in keypoints_dict.items():
                            all_keypoints.append({
                                "Keypoint": name,
                                "X": f"{kp['x']:.1f}",
                                "Y": f"{kp['y']:.1f}",
                                "Confidence": f"{kp['confidence']:.3f}",
                                "Status": "✅" if kp.get('confidence', 0) > 0.1 else "❌" # 신뢰도에 따른 상태
                            })
                        st.dataframe(all_keypoints, use_container_width=True)
                    
                    # 시각화 섹션
                    st.subheader("🎨 시각화")
                    if st.session_state.get('temp_image_path') and os.path.exists(st.session_state.temp_image_path):
                        try:
                            original_image_for_viz = Image.open(st.session_state.temp_image_path)
                            if original_image_for_viz.mode in ('RGBA', 'LA', 'P'):
                                original_image_for_viz = original_image_for_viz.convert('RGB')
                            
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_viz_output:
                                image_array = np.array(original_image_for_viz, dtype=np.uint8)

                                visualizer.draw_pose_on_image(
                                    image_array,
                                    pose_data,
                                    tmp_viz_output.name # 시각화 결과 저장 경로
                                )
                                
                                viz_image = Image.open(tmp_viz_output.name) # 시각화된 이미지 열기
                                st.image(viz_image, caption="자세 감지 시각화", use_container_width=True) # 시각화 이미지 표시
                                
                                os.unlink(tmp_viz_output.name) # 임시 파일 삭제
                            
                        except Exception as e:
                            print(f"시각화 중 오류 발생: {e}")
                    else:
                        st.info("시각화를 보려면 이미지를 업로드하고 분석하세요.")
            else:
                st.info("아직 자세 분석 데이터가 없습니다. 이미지를 업로드하고 분석하세요!")
        else:
            st.info("👆 이미지를 업로드하고 '자세 분석'을 클릭하여 결과를 확인하세요.")
    
    # 푸터 (꼬리말)
    st.markdown("---")
    st.markdown(
        "**참고:** 이 시스템은 COCO2017으로 학습된 YOLO11x-pose 모델을 사용하여 자세 키포인트를 감지합니다. "
        "최상의 결과를 위해 이미지 속 인물이 명확하게 보이고 조명이 잘 되어 있는지 확인하세요."
    )


if __name__ == "__main__":
    main() # 메인 함수 실행
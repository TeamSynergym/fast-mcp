import matplotlib
matplotlib.rc('font', family='Malgun Gothic')
import matplotlib.pyplot as plt
import numpy as np
import cloudinary
import cloudinary.uploader
import uuid
import os

# 1. 점수 dict에서 radar chart용 점수 dict(3각/5각)를 추출하는 함수
# - 정면+측면(5개): 목, 어깨, 골반, 척추(정면), 척추(측면)
# - 정면만(3개): 어깨, 골반, 척추(정면)
def extract_radar_scores(scores: dict) -> dict:
    """
    점수 dict에서 radar chart용 점수 dict를 추출한다.
    - 정면+측면(5개): 목, 어깨, 골반, 척추(정면), 척추(측면)
    - 정면만(3개): 어깨, 골반, 척추(정면)
    """
    if all(k in scores and scores[k] is not None for k in ["목score", "어깨score", "골반틀어짐score", "척추휨score", "척추굽음score"]):
        # 오각형(정면+측면)
        return {
            "목": scores["목score"],
            "어깨": scores["어깨score"],
            "골반": scores["골반틀어짐score"],
            "척추(정면)": scores["척추휨score"],
            "척추(측면)": scores["척추굽음score"]
        }
    elif all(k in scores and scores[k] is not None for k in ["어깨score", "골반틀어짐score", "척추휨score"]):
        # 삼각형(정면만)
        return {
            "어깨": scores["어깨score"],
            "골반": scores["골반틀어짐score"],
            "척추(정면)": scores["척추휨score"]
        }
    else:
        # 차트 생성 불가
        return {}

# 2. radar_scores dict를 받아 차트 생성, 업로드, 삭제까지 처리하고 업로드된 url을 반환하는 함수
#    (내부적으로 plot_radar_chart를 helper로 사용)
def create_and_upload_radar_chart(radar_scores, folder="radar_charts/"):
    """
    radar_scores dict를 받아 차트 생성, 업로드, 삭제까지 처리하고 업로드된 url을 반환한다.
    사용 흐름:
    1. extract_radar_scores로 radar_scores 추출
    2. create_and_upload_radar_chart(radar_scores) 호출
       (내부적으로 plot_radar_chart로 파일 생성 → cloudinary 업로드 → 임시파일 삭제)
    """
    def _plot_radar_chart(scores: dict, output_path: str = "radar_chart.png") -> str:
        # 내부 helper: radar_scores dict(3개 또는 5개 부위 점수)를 받아 차트 이미지를 파일로 저장
        if "목" in scores and "척추(측면)" in scores:
            labels = ["목", "어깨", "골반", "척추(정면)", "척추(측면)"]
            rotation_offset = np.deg2rad(10)
        else:
            labels = ["어깨", "골반", "척추(정면)"]
            rotation_offset = np.deg2rad(10)
        values = [scores.get(label, 0) for label in labels]
        values += values[:1]
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles = [(a + rotation_offset) for a in angles]
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.plot(angles, values, linewidth=2, linestyle='solid')
        ax.fill(angles, values, alpha=0.25)
        ax.set_ylim(0, 100)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        plt.title("부위별 점수")
        if rotation_offset != 0:
            ax.set_theta_offset(rotation_offset)
        plt.savefig(output_path)
        plt.close()
        return output_path

    radar_chart_path = f"radar_chart_{uuid.uuid4().hex}.png"
    # 1. 차트 이미지 파일 생성 (내부 helper 사용)
    _plot_radar_chart(radar_scores, output_path=radar_chart_path)
    radar_chart_url = None
    try:
        # 2. Cloudinary 업로드
        upload_result = cloudinary.uploader.upload(radar_chart_path, folder=folder)
        radar_chart_url = upload_result["secure_url"]
        print(f"[create_and_upload_radar_chart] Cloudinary 업로드 성공: {radar_chart_url}")
    except Exception as e:
        print(f"[create_and_upload_radar_chart] Cloudinary 업로드 에러: {e}")
        radar_chart_url = None
    finally:
        # 3. 임시 파일 삭제
        if os.path.exists(radar_chart_path):
            os.remove(radar_chart_path)
    return radar_chart_url 

 
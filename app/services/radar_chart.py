import matplotlib
matplotlib.rc('font', family='Malgun Gothic')
import matplotlib.pyplot as plt
import numpy as np

def plot_radar_chart(scores: dict, output_path: str = "radar_chart.png") -> str:
    """
    scores: {
        "목": float (optional),
        "어깨": float,
        "골반": float,
        "척추(정면)": float,
        "척추(측면)": float (optional)
    }
    - 정면만 있으면 어깨, 골반, 척추(정면)로 삼각형
    - 정면+측면 있으면 목, 어깨, 골반, 척추(정면), 척추(측면)로 오각형
    """
    if "목" in scores and "척추(측면)" in scores:
        labels = ["목", "어깨", "골반", "척추(정면)", "척추(측면)"]
        rotation_offset = np.deg2rad(10)  # 오각형은 회전 없음
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
    # --- 추가: 눈금(0~100)도 같이 회전 ---
    if rotation_offset != 0:
        ax.set_theta_offset(rotation_offset)
    plt.savefig(output_path)
    plt.close()
    return output_path 
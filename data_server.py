import matplotlib
matplotlib.rc('font', family='Malgun Gothic')  # 윈도우라면 'Malgun Gothic' 추천
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False     # 마이너스(-) 깨짐 방지
import numpy as np
import pandas as pd
import seaborn as sns

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

mcp = FastMCP("DataAnalysis")

@mcp.tool()
def describe_column(csv_path: str, column: str) -> dict:
  """
  Get summary statistics (count, mean, std, min, max, etc.) for a specific column in a CSV file.

  Args:
    csv_path(str): The file path to the CSV file.
    column(str): The name of the column to compute statistic for.

  Returns:
    dict: A dictionary containing summary statistics for the specified column.
  """
  df = pd.read_csv(csv_path)
  if column not in df.columns:
    raise ValueError(f"Column '{column}' not found in the CSV file.")
  return df[column].describe().to_dict()


@mcp.tool()
def plot_histogram(csv_path: str, column: str, bins: int = 10) -> str:
  """
  Generate and save a density histogram for a specific column in a CSV file.

  Args:
    csv_path(str): The file path to the CSV file.
    column(str): The name of the column to plot.
    bins(int): The number of bins for the histogram. Defaults to 10.

  Returns:
    str: The file path where the histogram image is saved.
  """
  df = pd.read_csv(csv_path)
  if column not in df.columns:
    raise ValueError(f"Column '{column}' not found in the CSV file.")
  
  plt.figure(figsize=(8, 6))
  sns.histplot(
    df[column].dropna(),
    bins=bins,
    kde=True,
    stat="density",
    edgecolor="black",
    alpha=0.6,
  )
  plt.xlabel(column)
  plt.ylabel("Density")
  plt.title(f"Density Histogram of {column}")

  output_path = f"{column}_histogram.png"
  plt.savefig(output_path)
  plt.close()

  return output_path


@mcp.tool()
def model(csv_path: str, x_columns: list, y_column: str) -> dict:
  """
  Automatically train a model (classification or regression) based on the target column type.

  Args:
    csv_path(str): The file path to the CSV file.
    x_columns(list): List of feature column names.
    y_column(str): The target column name.

  Returns:
    Dictionary with model type, performance metric and score.
  """
  df = pd.read_csv(csv_path)

  for col in x_columns + [y_column]:
    if col not in df.columns:
      raise ValueError(f"Column '{col}' not found in the CSV file.")
    
  X = df[x_columns]
  y= df[y_column]

  for col in X.select_dtypes(include=['object']).columns:
    X[col] = LabelEncoder().fit_transform(X[col])

  is_classification = y.dtype == "object" or len(y.unique()) <= 10

  if is_classification:
    y = LabelEncoder().fit_transform(y)
    model = RandomForestClassifier()
    metric_name = "accuracy"
  else:
    model = RandomForestRegressor()
    metric_name = "rmse"

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)

  if is_classification:
    score = accuracy_score(y_test, y_pred)
    model_type = "classification"
  else:
    score = root_mean_squared_error(y_test, y_pred, squared=False)
    model_type = "regression"

  return {"model_type": model_type, "metric": metric_name, "score": score}


@mcp.tool()
def plot_radar_chart(values: dict, output_path: str = "radar_chart.png") -> str:
    """
    오각형(레이더 차트) 그래프를 생성하고 이미지를 저장합니다.
    - 차트의 범위는 0부터 100까지이며, 20단위로 눈금을 표시합니다.
    - 차트를 회전시켜 첫 항목이 상단을 향하도록 조정합니다.

    Args:
        values (dict): {"목": float, "어깨": float, "골반": float, "척추(정면)": float, "척추(측면)": float}
        output_path (str): 저장할 이미지 파일 경로

    Returns:
        str: 저장된 이미지 파일 경로
    """

    matplotlib.rc('font', family='Malgun Gothic')
    plt.rcParams['axes.unicode_minus'] = False

    labels = ["목", "어깨", "골반", "척추(정면)", "척추(측면)"]
    stats = [values.get(label, 0) for label in labels]

    #회전 각도 설정 (단위: 도). 90으로 설정하면 첫 항목('목')이 위쪽(12시 방향)
    rotation_angle_deg = 20
    #도(degree)를 라디안(radian)으로 변환합니다.
    rotation_offset = np.deg2rad(rotation_angle_deg)
    # np.linspace를 사용하여 5개의 꼭짓점을 360도에 고르게 분포
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False) + rotation_offset
    angles = angles.tolist()
    # 그래프를 닫기 위해 시작점 데이터를 끝에 추가
    stats += stats[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, stats, linewidth=2, linestyle='solid')
    ax.fill(angles, stats, alpha=0.25)
    # Y축(반지름)의 최소/최대값 설정 (0부터 100까지)
    ax.set_ylim(0, 100)
    # Y축 눈금 설정 (20 단위로)
    ax.set_yticks(np.arange(0, 101, 20))
    #숫자 눈금(0~100)이 표시될 각도를 설정
    ax.set_rlabel_position(126)
    
    # X축(각도)의 눈금 및 라벨 설정
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    plt.title("부위별 균형 변화")
    plt.savefig(output_path)
    plt.close()
    return output_path


@mcp.prompt()
def default_prompt(message: str) -> list[base.Message]:
  return [
    base.AssistantMessage(
      "You are a helpful data analysis assistant. \n"
      "Please clearly organize and return the results of the tool calling and the data analysis."
    ),
    base.UserMessage(message),
  ]


if __name__ == "__main__":
  mcp.run(transport="stdio")
  # 테스트용 예시 데이터
  # test_values = {
  #       "목": 80,
  #       "어깨": 65,
  #       "골반": 75,
  #       "척추(정면)": 90,
  #       "척추(측면)": 50}
  # output = plot_radar_chart(test_values, "test_radar_chart.png")
  # print(f"Radar chart saved to: {output}")
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
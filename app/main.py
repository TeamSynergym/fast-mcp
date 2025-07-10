import os
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.services.posture_analyzer import PostureAnalyzer

# --- [수정된 함수] ---
# NumPy 2.0+ 버전에 호환되도록 수정한 헬퍼 함수
def convert_numpy_to_python(data):
    if isinstance(data, dict):
        return {key: convert_numpy_to_python(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_to_python(item) for item in data]
    # np.floating은 모든 NumPy float 타입을 포함합니다 (np.float32, np.float64 등)
    elif isinstance(data, np.floating):
        return float(data)
    # np.integer는 모든 NumPy int 타입을 포함합니다 (np.int32, np.int64 등)
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    return data
# -------------------------


# 1. FastAPI 앱과 PostureAnalyzer를 초기화합니다.
app = FastAPI()
analyzer = PostureAnalyzer(model_path="models/yolopose_v1.pt")

# 2. curl로 받을 요청의 형식을 정의합니다.
class AnalysisRequest(BaseModel):
    image_path: str
    mode: str = 'front'

# 3. /analyze 라는 주소로 POST 요청을 처리할 API 엔드포인트를 만듭니다.
@app.post("/analyze")
async def analyze_posture_endpoint(request: AnalysisRequest):
    """
    cURL 요청을 받아 자세 분석을 실행하고 결과를 JSON으로 반환합니다.
    """
    if not os.path.exists(request.image_path):
        raise HTTPException(status_code=404, detail="Image not found on the server.")

    analysis_result = analyzer.analyze_posture(
        image_path=request.image_path,
        mode=request.mode
    )
    
    if analysis_result["success"] and analysis_result["pose_data"]:
        person_data = analysis_result["pose_data"][0]
        diagnosis = analyzer.generate_ollama_diagnosis(person_data, request.mode)
        analysis_result["ollama_diagnosis"] = diagnosis
    
    # 반환하기 직전에 데이터 타입을 변환합니다.
    final_result = convert_numpy_to_python(analysis_result)

    return final_result
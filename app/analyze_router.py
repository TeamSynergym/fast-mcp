import os
import json
import requests
from fastapi import FastAPI, HTTPException, Request, APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
import redis
from app.graph_workflow import app_graph, app_merge_graph
from langchain_openai import ChatOpenAI
import cloudinary
import cloudinary.uploader
import uuid
import tempfile
import shutil
from fastapi.encoders import jsonable_encoder

load_dotenv()

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)
redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "192.168.2.6"), port=6379, db=0, decode_responses=True)
router = APIRouter()
llm = ChatOpenAI(model="gpt-4o-mini")

# --- FastAPI 요청 바디 데이터 모델 정의 ---
class AnalysisRequest(BaseModel):
    """
    단일 이미지 분석 요청에 사용되는 데이터 모델.
    - image_url: Cloudinary 이미지 URL
    - analysis_mode: 분석 모드 ('front' 또는 'side'), 기본값 'front'
    """
    image_url: str
    analysis_mode: str = "front"

class MultiAnalysisRequest(BaseModel):
    """
    정면+측면 MERGE 분석 요청에 사용되는 데이터 모델.
    - front_image_url: 정면 Cloudinary 이미지 URL
    - side_image_url: 측면 Cloudinary 이미지 URL
    """
    front_image_url: str
    side_image_url: str

# --- 분석/레이더차트만 실행하는 엔드포인트 (변경) ---
@router.post("/analyze-graph")
async def analyze_graph_endpoint(request: AnalysisRequest):
    print("=== [분석 요청] (Graph 기반) ===")
    print("request:", request)
    image_url = request.image_url
    if not image_url.startswith("https://res.cloudinary.com/"):
        raise HTTPException(status_code=400, detail="Cloudinary 이미지 URL만 허용됩니다.")
    input_data = {
        "image_url": image_url,
        "mode": request.analysis_mode
    }
    result = app_graph.invoke(input_data)
    # 반환 직전 실제 result를 예쁘게 출력
    print("[Python 반환 result] (단일)")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return jsonable_encoder(result)

@router.post("/analyze-graph/merge")
async def analyze_graph_merge_endpoint(request: MultiAnalysisRequest):
    input_data = {
        "front_image_url": request.front_image_url,
        "side_image_url": request.side_image_url
    }
    result = app_merge_graph.invoke(input_data)
    # 반환 직전 실제 result를 예쁘게 출력
    print("[Python 반환 result] (merge)")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return jsonable_encoder(result)

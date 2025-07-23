import os
import json
import requests
from fastapi import FastAPI, HTTPException, Request, APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
import redis
from app.services.posture_analyzer import PostureAnalyzer
from app.services.exercise_vector_db import ExerciseVectorDB
from app.services.radar_chart import plot_radar_chart
from app.services.exercise_vector_db import ExerciseVectorDB
from app.services.recommend_exercise_service import recommend_exercise_node
from langchain_openai import ChatOpenAI
import cloudinary
import cloudinary.uploader
import uuid
import tempfile
import shutil


load_dotenv()

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)
redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=6379, db=0, decode_responses=True)
posture_analyzer = PostureAnalyzer(model_path="models/yolopose_v1.pt")
vector_db = ExerciseVectorDB()
router = APIRouter()
llm = ChatOpenAI(model="gpt-4o-mini")
vector_db = ExerciseVectorDB()

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
    print("=== [분석 요청] ===")
    print("request:", request)
    # Cloudinary URL만 허용
    image_url = request.image_url
    if not image_url.startswith("https://res.cloudinary.com/"):
        raise HTTPException(status_code=400, detail="Cloudinary 이미지 URL만 허용됩니다.")
    analysis_result = posture_analyzer.analyze_posture(image_url, mode=request.analysis_mode)
    print("analysis_result:", analysis_result)
    if not analysis_result.get("success"):
        print("자세 분석 실패!")
        raise HTTPException(status_code=400, detail="자세 분석 실패")
    pose_data = analysis_result.get("pose_data", [{}])[0] if analysis_result.get("pose_data") else {}
    print("pose_data:", pose_data)
    diagnosis = posture_analyzer.generate_ollama_diagnosis(pose_data, request.analysis_mode)
    print("diagnosis:", diagnosis)
    # 추천운동 생성
    rec_result = recommend_exercise_node(diagnosis.get("korean", ""), vector_db, llm)
    if "error" in rec_result:
        raise HTTPException(status_code=500, detail=rec_result["error"])
    recommended_exercise = rec_result.get("recommended_exercise", {})
    search_query = rec_result.get("search_query", "")

    # 점수 dict 동적 생성 (정면/측면 여부에 따라)
    scores = pose_data.get('scores', {}) if pose_data else {}
    print("scores:", scores)
    radar_scores = {}
    # 오각형: 정면+측면 모두 있을 때
    if all(k in scores and scores[k] is not None for k in ["목score", "어깨score", "골반틀어짐score", "척추휨score", "척추굽음score"]):
        radar_scores = {
            "목": scores["목score"],
            "어깨": scores["어깨score"],
            "골반": scores["골반틀어짐score"],
            "척추(정면)": scores["척추휨score"],
            "척추(측면)": scores["척추굽음score"]
        }
    # 삼각형: 정면만 있을 때
    elif all(k in scores and scores[k] is not None for k in ["어깨score", "골반틀어짐score", "척추휨score"]):
        radar_scores = {
            "어깨": scores["어깨score"],
            "골반": scores["골반틀어짐score"],
            "척추(정면)": scores["척추휨score"]
        }
    # 그 외(측면만 등): 차트 생성 X
    print("radar_scores:", radar_scores)

    # 차트 생성 조건: radar_scores가 3개(삼각형) 또는 5개(오각형)일 때만
    radar_chart_url = None
    if len(radar_scores) in (3, 5) and any(v > 0 for v in radar_scores.values()):
        radar_chart_path = f"radar_chart_{uuid.uuid4().hex}.png"
        plot_radar_chart(radar_scores, output_path=radar_chart_path)
        print("radar_chart_path:", radar_chart_path)
        print("파일 존재 여부:", os.path.exists(radar_chart_path))
        try:
            upload_result = cloudinary.uploader.upload(radar_chart_path, folder="radar_charts/")
            radar_chart_url = upload_result["secure_url"]
            print("Cloudinary 업로드 성공:", radar_chart_url)
        except Exception as e:
            print("Cloudinary 업로드 에러:", e)
            radar_chart_url = None
        finally:
            if os.path.exists(radar_chart_path):
                os.remove(radar_chart_path)
    else:
        print("radar_scores가 3개/5개가 아니거나 모두 0입니다. 차트 생성/업로드 생략.")

    print("최종 응답 radar_chart_url:", radar_chart_url)
    return {
        "diagnosis": diagnosis,
        "recommended_exercise": recommended_exercise,
        "search_query": search_query,
        "pose_data": pose_data,
        "radar_chart_url": radar_chart_url,
        "spineCurvScore": scores.get("척추굽음score", 0),
        "spineScolScore": scores.get("척추휨score", 0),
        "pelvicScore": scores.get("골반틀어짐score", 0),
        "neckScore": scores.get("거북목score", 0),
        "shoulderScore": scores.get("어깨score", 0)
    }

@router.post("/analyze-graph/merge")
async def analyze_graph_merge_endpoint(request: MultiAnalysisRequest):
    front_url = request.front_image_url
    side_url = request.side_image_url
    if not front_url.startswith("https://res.cloudinary.com/") or not side_url.startswith("https://res.cloudinary.com/"):
        raise HTTPException(status_code=400, detail="Cloudinary 이미지 URL만 허용됩니다.")
    # 분석
    front_result = posture_analyzer.analyze_posture(front_url, mode='front')
    front_pose = front_result.get('pose_data', [{}])[0] if front_result.get('pose_data') else {}
    side_result = posture_analyzer.analyze_posture(side_url, mode='side')
    side_pose = side_result.get('pose_data', [{}])[0] if side_result.get('pose_data') else {}
    print("[merge] front_pose measurements:", front_pose.get('measurements'))
    print("[merge] side_pose measurements:", side_pose.get('measurements'))

    # 5개 부위 점수 합치기
    front_scores = front_pose.get('scores', {}) if front_pose else {}
    side_scores = side_pose.get('scores', {}) if side_pose else {}
    all_scores = {
        "어깨score": front_scores.get("어깨score"),
        "골반틀어짐score": front_scores.get("골반틀어짐score"),
        "척추휨score": front_scores.get("척추휨score"),
        "척추굽음score": side_scores.get("척추굽음score"),
        "거북목score": side_scores.get("거북목score"),
    }
    # 가장 낮은 점수와 해당 부위 찾기
    min_part = None
    min_score = 101
    for key, val in all_scores.items():
        if val is not None and val < min_score:
            min_score = val
            min_part = key
    # 해당 부위의 pose_data 선택 (front/side)
    if min_part in ["어깨score", "골반틀어짐score", "척추휨score"]:
        pose_data_for_diag = front_pose
    else:
        pose_data_for_diag = side_pose
    # 진단 생성 (5개 부위 중 가장 낮은 점수만 기반)
    diagnosis = posture_analyzer.generate_ollama_diagnosis(pose_data_for_diag, "merge")

    # 추천운동 생성
    rec_result = recommend_exercise_node(diagnosis.get("korean", ""), vector_db, llm)
    if "error" in rec_result:
        raise HTTPException(status_code=500, detail=rec_result["error"])
    recommended_exercise = rec_result.get("recommended_exercise", {})
    search_query = rec_result.get("search_query", "")

    front_scores = front_pose.get('scores', {}) if front_pose else {}
    side_scores = side_pose.get('scores', {}) if side_pose else {}
    radar_scores = {
        "목": side_scores.get("거북목score", 0),
        "어깨": front_scores.get("어깨score", 0),
        "골반": front_scores.get("골반틀어짐score", 0),
        "척추(정면)": front_scores.get("척추휨score", 0),
        "척추(측면)": side_scores.get("척추굽음score", 0)
    }
    radar_chart_url = None
    if all(v is not None for v in radar_scores.values()) and any(v > 0 for v in radar_scores.values()):
        radar_chart_path = f"radar_chart_{uuid.uuid4().hex}.png"
        plot_radar_chart(radar_scores, output_path=radar_chart_path)
        try:
            upload_result = cloudinary.uploader.upload(radar_chart_path, folder="radar_charts/")
            radar_chart_url = upload_result["secure_url"]
        except Exception as e:
            radar_chart_url = None
        finally:
            if os.path.exists(radar_chart_path):
                os.remove(radar_chart_path)
    return {
        "diagnosis": diagnosis,
        "recommended_exercise": recommended_exercise,
        "search_query": search_query,
        "radar_scores": radar_scores,
        "radar_chart_url": radar_chart_url,
        "front_pose_data": front_pose,
        "side_pose_data": side_pose,
        "spineCurvScore": side_scores.get("척추굽음score", 0),
        "spineScolScore": front_scores.get("척추휨score", 0),
        "pelvicScore": front_scores.get("골반틀어짐score", 0),
        "neckScore": side_scores.get("거북목score", 0),
        "shoulderScore": front_scores.get("어깨score", 0)
    }

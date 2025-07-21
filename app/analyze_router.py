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

def download_image_to_tempfile(image_url: str) -> str:
    response = requests.get(image_url, stream=True)
    if response.status_code == 200:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        with open(tmp.name, 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
        return tmp.name
    else:
        raise Exception(f"이미지 다운로드 실패: {image_url}")

class AnalysisRequest(BaseModel):
    image_url: str
    analysis_mode: str = "front"

class MultiAnalysisRequest(BaseModel):
    front_image_url: str
    side_image_url: str

# --- 분석/레이더차트만 실행하는 엔드포인트 (변경) ---
@router.post("/analyze-graph")
async def analyze_graph_endpoint(request: AnalysisRequest):
    print("=== [분석 요청] ===")
    print("request:", request)
    # Cloudinary가 아닌 경우 로컬로 다운로드 후 업로드
    image_url = request.image_url
    local_path = None
    if not image_url.startswith("https://res.cloudinary.com/"):
        local_path = download_image_to_tempfile(image_url)
        # Cloudinary 업로드
        try:
            upload_result = cloudinary.uploader.upload(local_path)
            image_url = upload_result["secure_url"]
        finally:
            if os.path.exists(local_path):
                os.remove(local_path)
    analysis_result = posture_analyzer.analyze_posture(image_url, mode=request.analysis_mode)
    print("analysis_result:", analysis_result)
    if not analysis_result.get("success"):
        print("자세 분석 실패!")
        raise HTTPException(status_code=400, detail="자세 분석 실패")
    pose_data = analysis_result.get("pose_data", [{}])[0] if analysis_result.get("pose_data") else {}
    print("pose_data:", pose_data)
    diagnosis = posture_analyzer.generate_ollama_diagnosis(pose_data, request.analysis_mode)
    print("diagnosis:", diagnosis)
    search_query = f"{request.analysis_mode} 자세 교정 운동"
    recommended_list = vector_db.search(search_query, top_k=1)
    recommended_exercise = recommended_list[0] if recommended_list else {}
    print("recommended_exercise:", recommended_exercise)

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
        "pose_data": pose_data,
        "radar_chart_url": radar_chart_url,
        "recommended_exercise": recommended_exercise,
        "spineCurvScore": scores.get("척추굽음score", 0),
        "spineScolScore": scores.get("척추휨score", 0),
        "pelvicScore": scores.get("골반틀어짐score", 0),
        "neckScore": scores.get("거북목score", 0),
        "shoulderScore": scores.get("어깨score", 0)
    }

@router.post("/analyze-graph/merge")
async def analyze_graph_merge_endpoint(request: MultiAnalysisRequest):
    # 정면
    front_url = request.front_image_url
    front_local = None
    if not front_url.startswith("https://res.cloudinary.com/"):
        front_local = download_image_to_tempfile(front_url)
        try:
            upload_result = cloudinary.uploader.upload(front_local)
            front_url = upload_result["secure_url"]
        finally:
            if os.path.exists(front_local):
                os.remove(front_local)
    # 측면
    side_url = request.side_image_url
    side_local = None
    if not side_url.startswith("https://res.cloudinary.com/"):
        side_local = download_image_to_tempfile(side_url)
        try:
            upload_result = cloudinary.uploader.upload(side_local)
            side_url = upload_result["secure_url"]
        finally:
            if os.path.exists(side_local):
                os.remove(side_local)
    # 분석
    front_result = posture_analyzer.analyze_posture(front_url, mode='front')
    front_pose = front_result.get('pose_data', [{}])[0] if front_result.get('pose_data') else {}
    side_result = posture_analyzer.analyze_posture(side_url, mode='side')
    side_pose = side_result.get('pose_data', [{}])[0] if side_result.get('pose_data') else {}
    # measurements 로그 추가
    print("[merge] front_pose measurements:", front_pose.get('measurements'))
    print("[merge] side_pose measurements:", side_pose.get('measurements'))
    # diagnosis 생성 (front, side 모두)
    diagnosis = {
        "front": posture_analyzer.generate_ollama_diagnosis(front_pose, "front"),
        "side": posture_analyzer.generate_ollama_diagnosis(side_pose, "side")
    }
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

    # (recommend_exercise_node 함수 추가)
def recommend_exercise_node(state):
    print("[Node 2] 맞춤 운동 추천 중 (from VectorDB)...")
    if state.get("error"): return {}
    try:
        diagnosis_text = state["diagnosis"]["korean"]
        from app.services.exercise_vector_db import ExerciseVectorDB
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o-mini")
        vector_db = ExerciseVectorDB()
        prompt = f"""아래의 자세 진단 내용에 가장 적합한 '단 한 가지'의 검색어을 추천해줘. 
        ~난이도, ~효과를 가진, ~부위의, ~운동의 순서로 검색어를 작성해야해.
        VectorDB 검색에 사용할 키워드 문장 오직 한개만 간결하게 한 줄로 답해줘.
        
        [진단 내용]
        {diagnosis_text}
        [출력 예시]
        - 중급 난이도의 유연성을 높이는 효과를 가진 골반 부위의 스트레칭 운동
        [생성된 검색어]
        """
        llm_query = llm.invoke(prompt).content.strip()
        print(f"  > LLM 생성 검색어: '{llm_query}'")
        recommended_list = vector_db.search(llm_query, top_k=1)
        if not recommended_list:
            raise ValueError("VectorDB에서 추천 운동을 찾지 못했습니다.")
        retrieved_exercise = recommended_list[0]
        print(f"  > VectorDB 검색 결과 운동명: '{retrieved_exercise['name']}'")
        return {"recommended_exercise": retrieved_exercise, "search_query": llm_query}
    except Exception as e:
        return {"error": f"운동 추천 노드 오류: {e}"}

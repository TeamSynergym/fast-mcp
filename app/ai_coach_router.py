from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import redis
import os
from dotenv import load_dotenv
from app.services.ai_coach_service import ai_coach_interaction_service
from app.analyze_router import llm  # 글로벌 llm 인스턴스 재사용
import json

load_dotenv()
redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=6379, db=0, decode_responses=True)

router = APIRouter(prefix="/ai-coach", tags=["AI Coach"])

class AiCoachRequest(BaseModel):
    userId: int
    diagnosis: dict  # {"korean": ...}
    recommended_exercise: dict  # {"name": ...}
    message: str = None  # 사용자의 추가 메시지(선택)

class AiCoachResponse(BaseModel):
    type: str
    response: str
    history: list = None

@router.post("", response_model=AiCoachResponse)
async def ai_coach_endpoint(req: AiCoachRequest):
    session_key = f"ai_coach_session:{req.userId}"
    # Redis에서 기존 대화 내역 불러오기 (없으면 빈 리스트)
    history = []
    if redis_client.exists(session_key):
        try:
            history = json.loads(redis_client.get(session_key))
        except Exception:
            history = []
    # 사용자의 메시지가 있으면 대화 내역에 추가
    if req.message:
        history.append({"role": "user", "content": req.message})
    # AI 코치 응답 생성
    diagnosis_text = req.diagnosis.get("korean", "")
    ai_coach_result = ai_coach_interaction_service(diagnosis_text, req.recommended_exercise, llm)
    if "error" in ai_coach_result:
        raise HTTPException(status_code=500, detail=ai_coach_result["error"])
    ai_coach_message = ai_coach_result["ai_coach_response"]
    # 대화 내역에 AI 코치 응답 추가
    history.append({"role": "ai_coach", "content": ai_coach_message})
    # Redis에 24시간 TTL로 저장 (세션 1개만 유지)
    redis_client.set(session_key, json.dumps(history), ex=60*60*24)
    return AiCoachResponse(type="ai_coach", response=ai_coach_message, history=history) 
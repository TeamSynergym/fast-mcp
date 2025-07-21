from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import requests
import json
import redis

redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

router = APIRouter(prefix="/ai-coach", tags=["AI Coach"])

class CoachRequest(BaseModel):
    userId: int
    historyId: int
    message: str = None
    diagnosis: dict = None  # Spring에서 진단 결과를 직접 전달받음
    recommended_exercise: dict = None  # Spring에서 추천운동을 직접 전달받음

class CoachResponse(BaseModel):
    type: str
    response: str

@router.post("", response_model=CoachResponse)
async def ai_coach_endpoint(req: CoachRequest):
    session_key = f"chat_session:{req.userId}"
    # SPRING_API_URL 삭제
    # 진단/추천운동 등은 req에서 직접 받음
    diagnosis = req.diagnosis or {"korean": "진단 정보가 없습니다."}
    recommended_exercise = req.recommended_exercise or {"name": "목 스트레칭"}
    state = {
        "diagnosis": diagnosis,
        "recommended_exercise": recommended_exercise,
        "user_message": req.message
    }
    from app.analyze_router import recommend_exercise_node
    from app.main_graph import ai_coach_interaction_node
    # recommend_exercise_node는 analyze_router에서 import
    # 이미 추천운동이 있으면 recommend_exercise_node는 생략 가능
    result = ai_coach_interaction_node(state)
    if "error" in result:
        return CoachResponse(type="error", response=f"AI 코치 오류: {result['error']}")
    response_text = result["messages"][-1].content if "messages" in result and result["messages"] else "AI 코치 응답이 없습니다."
    redis_client.rpush(session_key, json.dumps({"type": "ai_coach", "content": response_text}))
    return CoachResponse(type="ai_coach", response=response_text) 
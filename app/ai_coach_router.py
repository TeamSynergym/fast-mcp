from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import redis
import os
from dotenv import load_dotenv
from app.analyze_router import llm  # 글로벌 llm 인스턴스 재사용
import json
from app.graph_workflow import app_ai_coach_graph

load_dotenv()
redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "192.168.2.6"), port=6379, db=0, decode_responses=True)

router = APIRouter()

class AiCoachRequest(BaseModel):
    user_id: int = Field(..., alias='userId')
    diagnosis: dict
    recommended_exercise: dict
    message: str = None

    class Config:
        allow_population_by_field_name = True

class AiCoachResponse(BaseModel):
    type: str
    response: str
    history: list = None

@router.post("/ai-coach")
async def ai_coach_endpoint(request: AiCoachRequest):
    input_data = {
        "user_id": request.user_id,
        "diagnosis": request.diagnosis,
        "recommended_exercise": request.recommended_exercise,
        "message": request.message
    }
    result = app_ai_coach_graph.invoke(input_data)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result 
from app.graph.recommendation_builder import create_recommendation_graph
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
from dotenv import load_dotenv

load_dotenv()

class RecommendationRequest(BaseModel):
    """운동 추천 요청을 위한 데이터 모델"""
    user_id: str
    user_profile: Dict[str, Any] = Field(..., description="키, 몸무게, 나이, 운동 목표 등")
    posture_analysis: Dict[str, Any] = Field(..., description="사용자 체형 분석 결과")
    exercise_history: List[Dict[str, Any]]
    liked_exercises: List[Dict[str, Any]]
    user_routines: List[Dict[str, Any]]
    
router = APIRouter(
    prefix="/workflow",
    tags=["Exercise Recommendation"]
)

recommendation_graph = create_recommendation_graph()

@router.post("/recommend-exercises", summary="AI 운동 추천")
async def recommend_exercises(request_data: RecommendationRequest) -> Dict[str, Any]:
    """
    사용자 데이터를 기반으로 개인화된 운동을 추천합니다.
    """
    inputs = request_data.dict()
    final_state = recommendation_graph.invoke(inputs)

    return {
        "message": "AI 추천 운동 목록입니다.",
        "recommendations": final_state.get("recommendations"),
        "reason": final_state.get("recommendation_reason")
    }
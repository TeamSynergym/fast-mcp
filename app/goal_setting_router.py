from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# LangGraph 관련 모듈 import
from app.graph2.builder import create_graph

class GoalRequest(BaseModel):
    """
    Java 백엔드로부터 AI 목표 생성을 요청받을 때 사용하는 모델.
    이전 에러 해결 과정에서 정의한 모델을 그대로 사용합니다.
    """
    exercise_history: List[Dict[str, Any]]
    coach_persona: str

router = APIRouter(
    prefix="/workflow",    
    tags=["Goal Setting"] 
)

graph = create_graph()
print("✅ AI 목표 생성 그래프가 성공적으로 컴파일되었습니다.")

@router.post("/generate-goal", summary="AI 목표 생성 워크플로우 실행")
def run_goal_generation_workflow(request: GoalRequest) -> Dict[str, Any]:
    """
    Java 백엔드로부터 사용자 데이터를 받아 전체 목표 제안 워크플로우를 실행합니다.
    이 엔드포인트는 중간에 멈추지 않고, 최종 확정된 목표까지 생성한 후 결과를 반환합니다.
    """
    print("--- [/workflow/generate-goal] 요청 수신 ---")
    
    # 1. LangGraph의 입력(ExerciseState) 형식에 맞게 데이터를 구성합니다.
    #    Java에서 보내주지 않는 필드들은 여기서 기본값을 채워줍니다.
    inputs = {
        "exercise_history": request.exercise_history,
        "coach_persona": request.coach_persona,
        "user_id": request.exercise_history[0].get("userId", "unknown_user"),
        "user_email": None,
        "comparison_stats": {},
        "fatigue_analysis": {},
        "slump_prediction": {},
        "analysis_result": "",
        "suggested_goals": "",
        "feedback": {},
        "final_goals": "",
        "generated_badge": {},
        "is_goal_achieved": False
    }

    # 2. LangGraph 워크플로우를 실행합니다.
    #    .invoke()는 그래프가 끝날 때까지 모든 노드를 실행합니다.
    final_state = graph.invoke(inputs, {"recursion_limit": 15})
    
    if not final_state or not final_state.get('final_goals'):
        raise HTTPException(status_code=500, detail="워크플로우가 최종 목표를 생성하지 못했습니다.")

    print("--- ✅ 워크플로우 전체 완료. 최종 결과 반환 ---")
    
    # 3. Java 백엔드가 필요한 'analysis_result'와 'final_goals'만 추출하여 반환합니다.
    response_data = {
        "user_id": inputs["user_id"],  # 초기 입력에서 user_id를 가져옵니다.
        "analysis_result": final_state.get("analysis_result", "분석 결과가 없습니다."),
        "final_goals": final_state.get("final_goals", {}),
        "generated_badge": final_state.get("generated_badge", {}),
    }
    
    print("🔍 최종 결과:", response_data)
    
    return response_data
# app/graph/state.py
from typing import TypedDict, List, Dict, Any, Optional

class ExerciseState(TypedDict):
    """
    LangGraph 워크플로우의 상태를 정의합니다.
    """
    # --- 입력 및 DB 데이터 ---
    user_id: str
    jwt_token: str
    user_email: Optional[str]
    exercise_history: List[Dict[str, Any]]

    # --- 사용자 선택 및 AI 분석 데이터 ---
    coach_persona: str
    fatigue_analysis: Dict[str, Any]
    slump_prediction: Dict[str, Any]
    comparison_stats: Dict[str, Any]
    analysis_result: str
    suggested_goals: str
    feedback: Dict[str, str]

    # --- 최종 결과 ---
    final_goals: str
    generated_badge: Dict[str, str]
    is_goal_achieved: bool

from typing import TypedDict, List, Dict, Any

class RecommendationState(TypedDict):
    """
    AI 운동 추천 워크플로우의 상태를 정의합니다.
    """
    # --- 입력 데이터 ---
    user_id: str
    exercise_history: List[Dict[str, Any]]
    liked_exercises: List[Dict[str, Any]] # 좋아요 한 운동 정보
    user_routines: List[Dict[str, Any]]   # 사용자가 저장한 루틴 정보
    user_profile: Dict[str, Any]          # 키, 몸무게, 목표 등
    posture_analysis: Dict[str, Any]

    # --- AI 분석 데이터 ---
    user_summary: str           # 사용자의 특성을 요약한 텍스트
    search_query: str           # Vector DB 검색을 위한 생성된 쿼리
    recommendations: List[Dict[str, Any]] # 최종 추천 운동 목록
    recommendation_reason: str  # 추천 이유
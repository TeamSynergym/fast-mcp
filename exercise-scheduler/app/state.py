from typing import TypedDict, List, Dict, Any

class ExerciseState(TypedDict):
    """
    LangGraph 워크플로우의 상태를 정의하는 클래스입니다.
    그래프의 각 노드는 이 상태 객체를 공유하며 데이터를 주고받습니다.
    """
    # --- 입력 및 DB 데이터 ---
    user_id: str
    exercise_history: List[Dict[str, Any]] # 백엔드 API로부터 받은 운동 기록 리스트

    # --- 중간 처리 결과 ---
    analysis_result: str          # 운동 기록 분석 결과 텍스트
    suggested_goals: str          # LLM이 제안한 목표 (JSON 형식의 문자열)
    feedback: str                 # 제안된 목표에 대한 사용자의 피드백

    # --- 최종 결과 ---
    final_goals: str              # 사용자가 확정한 최종 목표
    is_goal_achieved: bool        # 목표 달성 여부를 나타내는 플래그

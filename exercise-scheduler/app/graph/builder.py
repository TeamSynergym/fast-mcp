# app/graph/builder.py
from langgraph.graph import StateGraph, END
from .state import ExerciseState 
from .nodes import (
    fetch_data_node,
    fetch_email_node,
    detect_fatigue_boredom_node,
    recommend_new_routine_node,
    predict_slump_node,
    fetch_comparison_data_node,
    analyze_records_node,
    persona_selection_node,
    suggest_goals_node,
    wait_for_feedback_node,
    refine_goals_node,
    finalize_goal_node
)
from .edges import (
    check_fatigue_condition_edge
)

def create_graph():
    """
    SynergyM 운동 목표 '설정' 워크플로우 그래프를 생성하고 컴파일합니다.
    """
    builder = StateGraph(ExerciseState)

    # 1. 노드 추가
    builder.add_node("fetch_data", fetch_data_node)
    builder.add_node("fetch_email", fetch_email_node)
    builder.add_node("fatigue_detector", detect_fatigue_boredom_node)
    builder.add_node("persona_selector", persona_selection_node)  # 위치 변경
    builder.add_node("routine_recommender", recommend_new_routine_node)
    builder.add_node("slump_predictor", predict_slump_node)
    builder.add_node("comparison_fetcher", fetch_comparison_data_node)
    builder.add_node("analyzer", analyze_records_node)
    builder.add_node("suggester", suggest_goals_node)
    builder.add_node("interrupter", wait_for_feedback_node)
    builder.add_node("refiner", refine_goals_node)
    builder.add_node("finalizer", finalize_goal_node)

    # 2. 엣지 연결 (새로운 유저플로우)
    builder.set_entry_point("fetch_data")
    builder.add_edge("fetch_data", "fetch_email")
    builder.add_edge("fetch_email", "fatigue_detector")
    
    # 2-1. 피로/지루함 상태에 따라 분기합니다.
    builder.add_edge("fatigue_detector", "persona_selector")  # persona_selector로 연결
    builder.add_conditional_edges(
        "persona_selector",  # 분기 전에 persona_selector 실행
        check_fatigue_condition_edge,
        {
            "needs_intervention": "routine_recommender",
            "continue_normal_flow": "slump_predictor" 
        }
    )
    builder.add_edge("routine_recommender", "slump_predictor")

    # 2-2. 목표 설정
    builder.add_edge("slump_predictor", "comparison_fetcher")
    builder.add_edge("comparison_fetcher", "analyzer")
    builder.add_edge("analyzer", "suggester")
    builder.add_edge("suggester", "interrupter")
    builder.add_edge("interrupter", "refiner")
    builder.add_edge("refiner", "finalizer")
    builder.add_edge("finalizer", END)

    return builder.compile()
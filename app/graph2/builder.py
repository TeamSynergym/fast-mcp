from langgraph.graph import StateGraph, END
from .state import ExerciseState 
from .nodes import (
    detect_fatigue_boredom_node,
    recommend_new_routine_node,
    predict_slump_node,
    analyze_records_node,
    suggest_goals_node,
    finalize_goal_node,
    generate_badge_node,
)
from .edges import (
    check_fatigue_condition_edge,
    check_goal_completion_edge
)

def create_graph():
    """
    단일 API 호출로 목표 분석부터 최종 확정까지 모두 처리하는
    단순화된 워크플로우 그래프를 생성합니다.
    """
    builder = StateGraph(ExerciseState)
    
    # 1. 그래프에 포함될 모든 노드를 정의합니다.
    builder.add_node("fatigue_detector", detect_fatigue_boredom_node)
    builder.add_node("routine_recommender", recommend_new_routine_node)
    builder.add_node("slump_predictor", predict_slump_node)
    builder.add_node("analyzer", analyze_records_node)
    builder.add_node("suggester", suggest_goals_node)
    builder.add_node("finalizer", finalize_goal_node)
    builder.add_node("badge_generator", generate_badge_node) 

    # 2. 그래프의 시작점(진입점)을 'fatigue_detector'로 고정합니다.
    builder.set_entry_point("fatigue_detector")
    
    # 3. 노드 간의 데이터 흐름(엣지)을 순서대로 연결합니다.
    builder.add_conditional_edges(
        "fatigue_detector",
        check_fatigue_condition_edge,
        { "needs_intervention": "routine_recommender", "continue_normal_flow": "slump_predictor" }
    )
    builder.add_edge("routine_recommender", "slump_predictor")
    builder.add_edge("slump_predictor", "analyzer")
    builder.add_edge("analyzer", "suggester")
    builder.add_edge("suggester", "finalizer")

    # 최종 목표 확정 후, 목표 달성 여부에 따라 분기합니다.
    builder.add_conditional_edges(
        "finalizer",
        check_goal_completion_edge,
        { "goal_achieved": "badge_generator", "goal_not_achieved": END }
    )
    
    # 목표 달성 시 뱃지 생성 후 보상 노드를 거쳐 종료됩니다.
    builder.add_edge("badge_generator", END)
    
    # 4. 중간에 멈추는 설정 없이, 끝까지 실행되는 그래프를 컴파일합니다.
    return builder.compile()
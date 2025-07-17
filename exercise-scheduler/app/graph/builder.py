# app/graph/builder.py
from langgraph.graph import StateGraph, END
from .state import ExerciseState 
from .nodes import (
    detect_fatigue_boredom_node,
    recommend_new_routine_node,
    predict_slump_node,
    analyze_records_node,
    persona_selection_node,
    suggest_goals_node,
    wait_for_feedback_node,
    refine_goals_node,
    finalize_goal_node,
    generate_badge_node,
    provide_reward_node,
)
from .edges import (
    check_fatigue_condition_edge,
    check_goal_completion_edge
)

def route_to_start(state: ExerciseState) -> str:
    """
    입력 상태를 확인하여 그래프의 시작 지점을 결정합니다.
    'feedback' 키가 존재하면, 사용자의 피드백을 처리하는 'refiner' 노드부터 시작합니다.
    그렇지 않으면, 워크플로우의 맨 처음인 'fatigue_detector' 부터 시작합니다.
    """
    print(f"--- [Routing] Checking for 'feedback' in state keys: {list(state.keys())}")
    if "feedback" in state and state["feedback"]:
        print("--- [Routing] Feedback found. Starting from 'refiner'. ---")
        return "refiner"
    else:
        print("--- [Routing] No feedback. Starting from 'fatigue_detector'. ---")
        return "fatigue_detector"

def create_graph():
    builder = StateGraph(ExerciseState)
    builder.add_node("fatigue_detector", detect_fatigue_boredom_node)
    builder.add_node("persona_selector", persona_selection_node)
    builder.add_node("routine_recommender", recommend_new_routine_node)
    builder.add_node("slump_predictor", predict_slump_node)
    builder.add_node("analyzer", analyze_records_node)
    builder.add_node("suggester", suggest_goals_node)
    builder.add_node("interrupter", wait_for_feedback_node)
    builder.add_node("refiner", refine_goals_node)
    builder.add_node("finalizer", finalize_goal_node)
    builder.add_node("badge_generator", generate_badge_node) 
    builder.add_node("rewarder", provide_reward_node)

    builder.set_conditional_entry_point(route_to_start)
    
    builder.add_edge("fatigue_detector", "persona_selector")
    
    builder.add_conditional_edges(
        "persona_selector",
        check_fatigue_condition_edge,
        { "needs_intervention": "routine_recommender", "continue_normal_flow": "slump_predictor" }
    )
    builder.add_edge("routine_recommender", "slump_predictor")

    builder.add_edge("slump_predictor", "analyzer")
    builder.add_edge("analyzer", "suggester")
    builder.add_edge("suggester", "interrupter")
    builder.add_edge("interrupter", "refiner")
    builder.add_edge("refiner", "finalizer")

    builder.add_conditional_edges(
        "finalizer",
        check_goal_completion_edge,
        { "goal_achieved": "badge_generator", "goal_not_achieved": END }
    )
    builder.add_edge("badge_generator", "rewarder")
    builder.add_edge("rewarder", END)
    
    return builder.compile()
# app/graph/builder.py
from langgraph.graph import StateGraph, END
from .state import ExerciseState 
from .nodes import (
    fetch_data_node,
    fetch_email_node,
    analyze_records_node,
    suggest_goals_node,
    wait_for_feedback_node,
    refine_goals_node,
    provide_reward_node
)
from .edges import check_goal_completion_edge

def create_graph():
    """
    SynergyM 운동 목표 제안 워크플로우 그래프를 생성하고 컴파일합니다.
    """
    builder = StateGraph(ExerciseState)

    # 1. 노드 추가
    builder.add_node("fetch_data", fetch_data_node)
    builder.add_node("fetch_email", fetch_email_node)
    builder.add_node("analyzer", analyze_records_node)
    builder.add_node("suggester", suggest_goals_node)
    builder.add_node("interrupter", wait_for_feedback_node)
    builder.add_node("refiner", refine_goals_node)
    builder.add_node("rewarder", provide_reward_node)

    # 2. 엣지 연결
    builder.set_entry_point("fetch_data")
    builder.add_edge("fetch_data", "fetch_email")
    builder.add_edge("fetch_email", "analyzer")
    builder.add_edge("analyzer", "suggester")
    builder.add_edge("suggester", "interrupter")
    builder.add_edge("interrupter", "refiner")
    builder.add_edge("rewarder", END)

    # 3. 조건부 엣지 연결
    builder.add_conditional_edges(
        "refiner",
        check_goal_completion_edge,
        {
            "goal_achieved": "rewarder",
            "goal_not_achieved": END
        }
    )

    # 4. 그래프 컴파일
    return builder.compile()
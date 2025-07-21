from langgraph.graph import StateGraph, END
from .state import ExerciseState
from .nodes import (
    predict_slump_node,
    analyze_records_node,
    suggest_goals_node,
    finalize_goal_node,
    generate_badge_node,
)
from .edges import (
    check_goal_completion_edge
)

def create_graph():
    """
    제공된 노드 파일을 기반으로 워크플로우 그래프를 생성합니다.
    """
    builder = StateGraph(ExerciseState)

    # 1. 그래프에 포함될 모든 노드를 정의합니다.
    builder.add_node("slump_predictor", predict_slump_node)
    builder.add_node("analyzer", analyze_records_node)
    builder.add_node("suggester", suggest_goals_node)
    builder.add_node("finalizer", finalize_goal_node)
    builder.add_node("badge_generator", generate_badge_node)

    # 2. 그래프의 시작점(진입점)을 설정합니다.
    # --- 수정: 존재하는 노드인 'slump_predictor'를 시작점으로 변경합니다 ---
    builder.set_entry_point("slump_predictor")

    # 3. 노드 간의 데이터 흐름(엣지)을 순서대로 연결합니다.
    builder.add_edge("slump_predictor", "analyzer")
    builder.add_edge("analyzer", "suggester")
    builder.add_edge("suggester", "finalizer")

    # 4. 최종 목표 확정 후, 목표 달성 여부에 따라 분기합니다.
    builder.add_conditional_edges(
        "finalizer",
        check_goal_completion_edge,
        { "goal_achieved": "badge_generator", "goal_not_achieved": END }
    )

    # 5. 뱃지 생성 후 워크플로우를 종료합니다.
    builder.add_edge("badge_generator", END)

    # 6. 완성된 그래프를 컴파일합니다.
    return builder.compile()
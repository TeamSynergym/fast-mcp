from langgraph.graph import StateGraph, END
from .state import ExerciseState 
from .nodes import (
    fetch_email_node, # 테스트에 필요
    generate_badge_node, # 테스트에 필요
    provide_reward_node # 테스트에 필요
)

def create_achievement_test_graph():
    """
    업적 저장 기능만 테스트하기 위한 미니 그래프를 생성합니다.
    """
    builder = StateGraph(ExerciseState)

    # 테스트에 필요한 노드만 추가
    builder.add_node("fetch_email", fetch_email_node)
    builder.add_node("badge_generator", generate_badge_node)
    builder.add_node("rewarder", provide_reward_node)

    # 실행 흐름 연결
    builder.set_entry_point("fetch_email")
    builder.add_edge("fetch_email", "badge_generator")
    builder.add_edge("badge_generator", "rewarder")
    builder.add_edge("rewarder", END)

    return builder.compile()
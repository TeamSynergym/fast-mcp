import os
import sys
from dotenv import load_dotenv

# Add the parent directory of 'app' to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(dotenv_path="c:\\Users\\PC\\Synergym\\fast-mcp\\.env")

from langgraph.graph import StateGraph, END

# app 폴더 내의 다른 모듈에서 필요한 클래스와 함수를 가져옵니다.
from app.state import ExerciseState
from app.nodes import (
    fetch_data_node,
    analyze_records,
    suggest_goals,
    wait_for_feedback,
    refine_goals,
    check_goal_completion,
    provide_reward
)

# .env 파일에서 환경 변수(OPENAI_API_KEY)를 로드합니다.
load_dotenv()

# ===================================================================
# ===== 테스트용 더미 데이터 및 함수 (나중에 이 부분을 삭제 또는 주석 처리) =====
# ===================================================================
# 백엔드 API 대신 아래의 더미 데이터를 사용하려면 True로 설정하세요.
USE_DUMMY_DATA = True 

def get_dummy_exercise_history():
    """테스트를 위한 더미 운동 기록을 반환합니다."""
    print("⚠️ 경고: 실제 API가 아닌 테스트용 더미 데이터를 사용합니다.")
    return [
        {'user_id': 'user123', 'exercise_date': '2025-07-01', 'completion_rate': 85.0, 'memo': '첫날, 가볍게 시작'},
        {'user_id': 'user123', 'exercise_date': '2025-07-03', 'completion_rate': 95.0, 'memo': '컨디션 최상!'},
        {'user_id': 'user123', 'exercise_date': '2025-07-04', 'completion_rate': 90.0, 'memo': '어제보다 조금 힘들었음'},
        {'user_id': 'user123', 'exercise_date': '2025-07-06', 'completion_rate': 100.0, 'memo': '완벽하게 소화! 기분 좋다.'},
        {'user_id': 'user123', 'exercise_date': '2025-07-08', 'completion_rate': 75.0, 'memo': '비가 와서 실내에서 진행'}
    ]

def dummy_fetch_data_node(state: ExerciseState) -> dict:
    """더미 데이터를 반환하는 fetcher 노드의 대체 함수입니다."""
    print("--- [Node] 1. 데이터 가져오기 실행 (더미 모드) ---")
    history = get_dummy_exercise_history()
    return {"exercise_history": history, "user_id": state.get("user_id")}
# ===================================================================
# ======================= 테스트용 코드 끝 ==========================
# ===================================================================


def run_graph():
    """
    LangGraph 워크플로우를 구성하고 실행합니다.
    """
    # 1. StateGraph 객체를 ExerciseState로 초기화합니다.
    builder = StateGraph(ExerciseState)

    # 2. 각 노드를 그래프 빌더에 추가합니다.
    # 테스트 모드 여부에 따라 시작 노드를 다르게 설정합니다.
    if USE_DUMMY_DATA:
        builder.add_node("fetcher", dummy_fetch_data_node)
    else:
        builder.add_node("fetcher", fetch_data_node)
        
    builder.add_node("analyzer", analyze_records)
    builder.add_node("suggester", suggest_goals)
    builder.add_node("interrupter", wait_for_feedback)
    builder.add_node("refiner", refine_goals)
    builder.add_node("rewarder", provide_reward)

    # 3. 노드 간의 데이터 흐름(엣지)을 정의합니다.
    builder.add_edge("fetcher", "analyzer")
    builder.add_edge("analyzer", "suggester")
    builder.add_edge("suggester", "interrupter")
    builder.add_edge("interrupter", "refiner")
    
    # 4. 조건부 엣지를 추가합니다.
    #    'refiner' 노드 실행 후, 'check_goal_completion' 함수의 반환값에 따라 다음 노드가 결정됩니다.
    builder.add_conditional_edges(
        "refiner",
        check_goal_completion,
        {
            # check_goal_completion이 "goal_achieved"를 반환하면 -> "rewarder" 노드로 이동
            "goal_achieved": "rewarder",
            # "goal_not_achieved"를 반환하면 -> 워크플로우 종료(END)
            "goal_not_achieved": END
        }
    )
    # 'rewarder' 노드 실행 후에는 워크플로우를 종료합니다.
    builder.add_edge("rewarder", END)
    
    # 5. 워크플로우의 시작점을 'fetcher' 노드로 설정합니다.
    builder.set_entry_point("fetcher")

    # 6. 정의된 노드와 엣지를 바탕으로 실행 가능한 그래프를 컴파일합니다.
    graph = builder.compile()

    # 7. 그래프 실행을 위한 초기 입력값을 설정합니다.
    #    워크플로우는 이 user_id를 가지고 시작됩니다.
    inputs = {"user_id": "user123"}

    # 8. stream()을 사용하여 그래프를 실행하고, 각 단계의 결과를 출력합니다.
    print("🚀 SynergyM 운동 목표 제안 워크플로우를 시작합니다.")
    for s in graph.stream(inputs, stream_mode="values"):
        # s는 각 단계가 끝난 후의 ExerciseState 전체를 담고 있습니다.
        print("\n" + "="*50)
        print("상태 업데이트 완료. 현재 상태:")
        # 가장 마지막으로 업데이트된 키와 값을 출력합니다.
        latest_update_key = list(s.keys())[-1]
        print(f"-> {latest_update_key}: {s[latest_update_key]}")
        print("="*50)

    print("\n✅ 워크플로우가 성공적으로 종료되었습니다.")

# 이 스크립트가 직접 실행될 때 run_graph 함수를 호출합니다.
if __name__ == "__main__":
    run_graph()

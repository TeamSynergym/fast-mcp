import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.graph.builder2 import create_achievement_test_graph
from config import DEFAULT_USER_ID, DEFAULT_JWT_TOKEN, OPENAI_API_KEY
import json

def run_test_workflow():
    """
    업적 저장 테스트 워크플로우를 실행합니다.
    """
    if not OPENAI_API_KEY:
        print("🚨 OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        return

    # 테스트 그래프 생성
    graph = create_achievement_test_graph()
    
    # 테스트를 위한 초기 입력값
    inputs = {
        "user_id": DEFAULT_USER_ID,  # 초기화된 user_id
        "jwt_token": DEFAULT_JWT_TOKEN
    }

    print("=" * 60)
    print("🚀 업적 저장 기능 테스트를 시작합니다.")
    print(f"   (대상 사용자 ID: {inputs['user_id']})")
    print("=" * 60)

    # 워크플로우 실행
    for s in graph.stream(inputs, stream_mode="values"):
        latest_update_key = list(s.keys())[-1]
        latest_update_value = s[latest_update_key]

        # state에서 사용자 목표 가져오기
        if latest_update_key == "state" and "user_goals" in latest_update_value:
            user_goals = latest_update_value["user_goals"]
            print(f"\n[사용자 목표] -> {user_goals}")

        print(f"\n[상태 업데이트] -> {latest_update_key}: {latest_update_value}")

    print("\n" + "=" * 60)
    print("✅ 테스트 워크플로우가 종료되었습니다.")
    print("=" * 60)

if __name__ == "__main__":
    run_test_workflow()
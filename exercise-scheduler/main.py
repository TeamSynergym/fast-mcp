# main.py
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.graph.builder import create_graph
from config import DEFAULT_USER_ID, DEFAULT_JWT_TOKEN, OPENAI_API_KEY

def run_workflow():
    """
    SynergyM 워크플로우를 설정하고 실행합니다.
    """
    # API 키 존재 여부 확인
    if not OPENAI_API_KEY:
        print("🚨 OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        print("   .env 파일을 확인해주세요.")
        return

    # 그래프 생성
    graph = create_graph()
    
    # 실행에 필요한 초기 입력값
    inputs = {
        "user_id": DEFAULT_USER_ID,
        "jwt_token": DEFAULT_JWT_TOKEN,
    }

    print("=" * 60)
    print("🚀 SynergyM 운동 목표 제안 워크플로우를 시작합니다.")
    print(f"   (대상 사용자 ID: {inputs['user_id']})")
    print("=" * 60)

    # 워크플로우 스트림 실행 및 결과 출력
    for s in graph.stream(inputs, stream_mode="values"):
        latest_update_key = list(s.keys())[-1]
        latest_update_value = s[latest_update_key]
        
        # 상태 업데이트 정보 출력
        print("\n[상태 업데이트] --------------------------------")
        print(f"-> {latest_update_key}: {latest_update_value}")
        print("---------------------------------------------")

    print("\n" + "=" * 60)
    print("✅ 워크플로우가 성공적으로 종료되었습니다.")
    print("=" * 60)

if __name__ == "__main__":
    run_workflow()
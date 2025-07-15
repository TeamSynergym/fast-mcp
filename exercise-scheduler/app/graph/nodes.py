# app/graph/nodes.py
import pandas as pd
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from .state import ExerciseState
from app.services.api_client import get_history_from_backend, get_email_from_backend
from app.services.notification import send_email_notification
from config import LLM_MODEL_NAME

# LLM 모델 초기화
llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0.7)

def fetch_data_node(state: ExerciseState) -> dict:
    """백엔드에서 운동 기록을 가져와 상태를 업데이트합니다."""
    print("--- [Node 1] 운동 기록 가져오기 ---")
    user_id = state["user_id"]
    jwt_token = state["jwt_token"]
    history = get_history_from_backend(user_id, jwt_token)
    if history:
        print(f"✅ 총 {len(history)}개의 운동 기록을 가져왔습니다.")
    else:
        print("🚨 운동 기록이 없거나 가져오는데 실패했습니다.")
    return {"exercise_history": history}

def fetch_email_node(state: ExerciseState) -> dict:
    """백엔드에서 사용자 이메일을 가져와 상태를 업데이트합니다."""
    print("--- [Node 2] 이메일 주소 가져오기 ---")
    user_id = state["user_id"]
    jwt_token = state["jwt_token"]
    email = get_email_from_backend(user_id, jwt_token)
    if email:
        print(f"✅ 사용자 이메일({email})을 확인했습니다.")
    else:
        print("🚨 사용자 이메일을 가져오는데 실패했습니다.")
    return {"user_email": email}

def analyze_records_node(state: ExerciseState) -> dict:
    """운동 기록을 분석하여 결과를 생성합니다."""
    print("--- [Node 3] 운동 기록 분석 ---")
    history = state.get('exercise_history')
    if not history:
        return {"analysis_result": "분석할 운동 기록이 없습니다."}
    
    df = pd.DataFrame(history)
    df['exerciseDate'] = pd.to_datetime(df['exerciseDate'])
    
    avg_completion = df['completionRate'].mean()
    total_sessions = len(df)
    period = (df['exerciseDate'].max() - df['exerciseDate'].min()).days
    
    analysis = (f"최근 약 {period}일간 총 {total_sessions}회 운동하셨고, "
                f"평균 완료율은 {avg_completion:.1f}% 입니다.")
    print(f"📊 분석 결과: {analysis}")
    return {"analysis_result": analysis}

def suggest_goals_node(state: ExerciseState) -> dict:
    """분석 결과를 바탕으로 LLM에게 목표 제안을 요청합니다."""
    print("--- [Node 4] LLM 목표 제안 ---")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 동기를 부여하는 운동 코치입니다. 분석 결과를 바탕으로 '주간 목표'와 '월간 목표'를 제안해주세요. 결과는 반드시 'weekly_goal', 'monthly_goal' 키를 가진 JSON 형식 문자열로, 한글로 작성해주세요."),
        ("human", "분석 결과: {analysis}")
    ])
    chain = prompt | llm
    result = chain.invoke({"analysis": state['analysis_result']})
    print(f"🤖 LLM 제안 (Raw): {result.content}")
    return {"suggested_goals": result.content}

def wait_for_feedback_node(state: ExerciseState) -> dict:
    """사용자로부터 제안된 목표에 대한 피드백을 입력받습니다."""
    print("\n--- [Node 5] 사용자 피드백 대기 ---")
    try:
        goals = json.loads(state['suggested_goals'])
        print("\n🤖 AI 코치가 제안하는 목표:")
        print(f"  - 주간: {goals.get('weekly_goal')}")
        print(f"  - 월간: {goals.get('monthly_goal')}\n")
    except (json.JSONDecodeError, KeyError):
        print("🚨 제안된 목표의 형식이 잘못되었습니다.")

    while True:
        print("마음에 드시나요? 옵션을 선택해주세요.")
        print("  1. 네, 좋아요! 이대로 설정할게요.")
        print("  2. 조금 더 쉬운 목표로 수정해주세요.")
        print("  3. 더 도전적인 목표로 수정해주세요.")
        choice = input("> ")
        if choice in ["1", "2", "3"]:
            return {"feedback": {"choice": choice}}
        else:
            print("🚨 1, 2, 3 중 하나를 입력해주세요.")

def refine_goals_node(state: ExerciseState) -> dict:
    """사용자 피드백을 반영하여 최종 목표를 확정하거나 수정합니다."""
    print("--- [Node 6] 목표 수정 및 확정 ---")
    feedback = state['feedback']
    choice = feedback.get("choice")
    
    if choice == "1":
        print("✅ 사용자가 제안을 수락했습니다. 목표를 최종 확정합니다.")
        final_goals = state['suggested_goals']
    else:
        print("⚠️ 사용자가 수정을 요청했습니다. LLM을 통해 목표를 재조정합니다.")
        request_map = {"2": "더 쉽게", "3": "더 어렵게(도전적으로)"}
        user_request = request_map.get(choice)

        refine_prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 운동 목표를 수정하는 코치입니다. 기존 제안과 사용자 피드백을 바탕으로 목표를 수정해주세요. 결과는 반드시 JSON 형식의 문자열로, 'weekly_goal'과 'monthly_goal' 키를 사용해 한글로 작성해주세요."),
            ("human", "기존 제안: {suggested_goals}\n사용자 요청: {user_request}\n\n수정된 목표(JSON)를 알려주세요.")
        ])
        refine_chain = refine_prompt | llm
        result = refine_chain.invoke({"suggested_goals": state['suggested_goals'], "user_request": user_request})
        final_goals = result.content
        print(f"🤖 LLM 수정 제안: {final_goals}")
        
    return {"final_goals": final_goals}

def provide_reward_node(state: ExerciseState) -> dict:
    """목표 달성 시 사용자에게 보상(이메일)을 제공합니다."""
    print("--- [Node 8] 보상 제공 ---")
    user_email = state.get('user_email')
    if not user_email:
        print("🚨 이메일 주소가 없어 보상을 제공할 수 없습니다.")
        return {"is_goal_achieved": False}

    subject = "🎉 축하합니다! 운동 목표를 성공적으로 달성하셨습니다!"
    try:
        goals_dict = json.loads(state['final_goals'])
        goals_text = f"  - 주간: {goals_dict.get('weekly_goal', 'N/A')}\n  - 월간: {goals_dict.get('monthly_goal', 'N/A')}"
    except (json.JSONDecodeError, AttributeError):
        goals_text = state['final_goals']

    body = (f"SynergyM의 {state['user_id']}님, 정말 대단해요!\n\n"
            f"꾸준한 노력으로 설정하신 아래의 목표를 달성하셨습니다.\n\n"
            f"✔ 달성한 목표:\n{goals_text}\n\n"
            "앞으로의 여정도 SynergyM이 함께 응원하겠습니다! 💪")
    
    send_email_notification(subject, body, user_email)
    return {"is_goal_achieved": True}
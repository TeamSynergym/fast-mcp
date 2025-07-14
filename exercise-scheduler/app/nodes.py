import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# app 폴더 내의 다른 모듈에서 필요한 클래스와 함수를 가져옵니다.
from app.state import ExerciseState
from app.api_client import get_history_from_backend

# 보상 제공을 위한 더미 함수 (실제로는 tools.py 등에 구현)
def send_email_notification(subject: str, body: str, to_email: str):
    """(시뮬레이션) 이메일 알림을 보내는 더미 함수입니다."""
    print("=" * 30)
    print("🏆 보상 알림 전송 (시뮬레이션)")
    print(f"[받는 사람]: {to_email}")
    print(f"[제       목]: {subject}")
    print(f"[내       용]:\n{body}")
    print("=" * 30)
    return "알림 전송 완료"

# LLM 모델 초기화 (환경 변수에서 OPENAI_API_KEY를 자동으로 읽어옵니다)
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)


def fetch_data_node(state: ExerciseState) -> dict:
    """API 클라이언트를 사용해 백엔드에서 운동 기록을 가져옵니다."""
    print("--- [Node] 1. 데이터 가져오기 실행 ---")
    user_id = state['user_id']
    exercise_history = get_history_from_backend(user_id)
    
    if exercise_history:
        print(f"✅ 총 {len(exercise_history)}개의 운동 기록을 API로부터 받았습니다.")
    else:
        print("🚨 API로부터 운동 기록을 가져오는 데 실패했거나 기록이 없습니다.")
        
    return {"exercise_history": exercise_history}


def analyze_records(state: ExerciseState) -> dict:
    """가져온 운동 기록을 분석하고 텍스트로 요약합니다."""
    print("--- [Node] 2. 운동 기록 분석 실행 ---")
    history = state.get('exercise_history')
    if not history:
        return {"analysis_result": "분석할 운동 기록이 없습니다. 먼저 운동을 시작해보세요!"}
        
    df = pd.DataFrame(history)
    # 날짜 형식 변환
    df['exercise_date'] = pd.to_datetime(df['exercise_date'])
    
    # 분석
    avg_completion = df['completion_rate'].mean()
    total_sessions = len(df)
    recent_period = (df['exercise_date'].max() - df['exercise_date'].min()).days

    analysis = (
        f"최근 약 {recent_period}일 동안 총 {total_sessions}회 운동하셨군요! "
        f"평균 운동 완료율은 {avg_completion:.1f}%로 아주 잘하고 계십니다."
    )
    print(f"분석 결과: {analysis}")
    return {"analysis_result": analysis}


def suggest_goals(state: ExerciseState) -> dict:
    """분석 결과를 LLM에 보내 목표를 제안받습니다."""
    print("--- [Node] 3. 목표 제안 생성 실행 ---")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 사용자의 동기를 부여하는 전문 운동 코치입니다. 사용자의 운동 분석 결과를 바탕으로, 달성 가능하고 구체적인 '주간 목표'와 '월간 목표'를 제안해주세요. 결과는 반드시 JSON 형식의 문자열로, 'weekly_goal'과 'monthly_goal' 키를 사용하여 한글로 작성해주세요."),
        ("human", "분석 결과: {analysis}")
    ])
    chain = prompt | llm
    result = chain.invoke({"analysis": state['analysis_result']})
    
    print(f"LLM 제안 목표 (JSON 문자열): {result.content}")
    return {"suggested_goals": result.content}


def wait_for_feedback(state: ExerciseState) -> dict:
    """사용자 피드백을 받는 부분을 시뮬레이션합니다."""
    print("--- [Node] 4. 사용자 피드백 대기 (시뮬레이션) ---")
    print("AI 코치: 제안된 목표가 마음에 드시나요? (예 / 아니오, 수정해주세요: [내용])")
    
    # 실제 애플리케이션에서는 이 부분에서 LangGraph의 interrupt()를 사용해 실행을 멈추고 입력을 기다립니다.
    # 여기서는 시뮬레이션을 위해 사용자가 긍정적인 피드백을 주었다고 가정합니다.
    feedback = "네, 아주 마음에 들어요. 이대로 진행하고 싶습니다!" 
    print(f"사용자 응답 (시뮬레이션): {feedback}")
    return {"feedback": feedback}

    
def refine_goals(state: ExerciseState) -> dict:
    """피드백을 반영하여 최종 목표를 확정합니다."""
    print("--- [Node] 5. 목표 확정 실행 ---")
    feedback = state.get('feedback', '')
    
    # 긍정적인 피드백일 경우, 제안된 목표를 최종 목표로 확정합니다.
    if "예" in feedback or "마음에" in feedback or "좋아요" in feedback:
        print("✅ 사용자가 제안을 수락하여 목표를 최종 확정합니다.")
        final_goals = state['suggested_goals']
    else:
        # 수정 요청이 있을 경우, LLM을 다시 호출하여 목표를 수정해야 합니다.
        # 여기서는 시뮬레이션을 위해 기존 목표를 그대로 사용합니다.
        print("⚠️ 사용자가 수정을 요청했습니다. (현재 시뮬레이션에서는 기존 목표를 유지합니다)")
        final_goals = state['suggested_goals']
        
    print(f"최종 확정된 목표: {final_goals}")
    return {"final_goals": final_goals}


def check_goal_completion(state: ExerciseState) -> str:
    """목표 달성 여부를 확인하여 다음 경로를 결정합니다."""
    print("--- [Node] 6. 목표 달성 여부 확인 실행 ---")
    
    # 이 함수는 다음 노드를 결정하는 '조건부 엣지'의 역할을 합니다.
    # 실제 서비스에서는 사용자의 새로운 운동 기록과 final_goals를 비교하는 로직이 필요합니다.
    # 여기서는 데모를 위해 목표가 확정되면 항상 '달성'했다고 가정합니다.
    if state.get("final_goals"):
        print("🎉 목표 달성! '보상 제공' 노드로 이동합니다.")
        return "goal_achieved"
    else:
        print("... 아직 목표가 설정되지 않았습니다. 워크플로우를 종료합니다.")
        return "goal_not_achieved"


def provide_reward(state: ExerciseState) -> dict:
    """목표 달성 시 이메일 알림 등의 보상을 제공합니다."""
    print("--- [Node] 7. 보상 제공 실행 ---")
    user_email = f"{state['user_id']}@synergym.com" # 예시 이메일 주소
    subject = "🎉 축하합니다! 운동 목표를 성공적으로 달성하셨습니다!"
    body = (
        f"SynergyM의 {state['user_id']}님, 정말 대단해요!\n\n"
        "꾸준한 노력으로 설정하신 운동 목표를 달성하신 것을 진심으로 축하드립니다.\n\n"
        f"✔ 달성한 목표:\n{state['final_goals']}\n\n"
        "작은 성공이 모여 큰 변화를 만듭니다. 저희 SynergyM이 앞으로의 여정도 함께 응원하겠습니다! 💪"
    )
    
    # 이메일 전송 함수 호출 (시뮬레이션)
    send_email_notification(subject, body, user_email)
    
    return {"is_goal_achieved": True}

import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from .state import ExerciseState

# LLM 모델 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def check_goal_completion_edge(state: ExerciseState) -> str:
    """
    LLM을 이용해 목표 달성 여부를 판단하여 다음 노드를 결정합니다.
    'goal_achieved' 또는 'goal_not_achieved'를 반환합니다.
    """
    print("--- [Edge 10] 목표 달성 여부 확인 ---")
   
    if not state.get("final_goals") or not state.get("exercise_history"):
        print("... 목표 또는 운동 기록이 없어 판단 불가. 워크플로우를 종료합니다.")
        return "goal_not_achieved"

    print("🧠 LLM으로 목표 달성 여부를 엄격하게 판단합니다...")
    history_str = pd.DataFrame(state['exercise_history']).to_string()

    eval_prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 엄격한 심판입니다. '달성 목표'와 '운동 기록'을 비교하여 목표 달성 여부를 판단해주세요. 답변은 오직 'goal_achieved' 또는 'goal_not_achieved' 둘 중 하나로만 해야 합니다."),
        ("human", "달성 목표 (JSON):\n{final_goals}\n\n운동 기록:\n{history}\n\n판단 결과는 무엇입니까?")
    ])
   
    eval_chain = eval_prompt | llm
    result = eval_chain.invoke({
        "final_goals": state['final_goals'],
        "history": history_str
    })
   
    decision = result.content.strip()
   
    if "goal_achieved" in decision:
        print("판단 결과: 🎉 목표 달성! '뱃지 생성' 노드로 이동합니다.")
        return "goal_achieved"
    else:
        print("판단 결과: 💪 목표 미달성. 워크플로우를 종료합니다.")
        return "goal_not_achieved"

# --- 신규 추가된 엣지 ---

def check_fatigue_condition_edge(state: ExerciseState) -> str:
    """
    피로/지루함 분석 결과에 따라 다음 노드를 결정합니다.
    """
    print("--- [Edge 4] 피로/지루함 상태 확인 ---")
    status = state.get("fatigue_analysis", {}).get("status", "normal")
    if status in ["fatigued", "bored"]:
        print(f"판단 결과: 😕 '{status}' 상태 감지. '루틴 추천'으로 분기합니다.")
        return "needs_intervention"
    else:
        print("판단 결과: 👍 상태 양호. '슬럼프 예측'으로 계속 진행합니다.")
        return "continue_normal_flow"

import pandas as pd
import re
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from app.graph2.state import ExerciseState

# LLM 모델 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
llm_strict = ChatOpenAI(model="gpt-4o-mini", temperature=0) # 분석/판단용

def detect_fatigue_boredom_node(state: ExerciseState) -> dict:
    """운동 기록을 보고 피로 또는 지루함 징후를 감지합니다."""
    print("--- [Node 3] 사용자 상태 감지 ---")
    history = state.get("exercise_history")
    if not history or len(history) < 5:
        return {"fatigue_analysis": {"status": "normal", "reason": "데이터 부족"}}

    history_str = pd.DataFrame(history).to_string()
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 운동 데이터 분석가입니다. 사용자의 최근 운동 기록을 보고 '피로' 또는 '지루함'의 징후가 있는지 판단해주세요. 
        
반드시 아래 형식의 JSON으로만 답변해주세요:
{{
  "status": "fatigued" | "bored" | "normal",
  "reason": "판단 근거를 한글로 간단히 설명"
}}

추가 설명이나 다른 텍스트는 포함하지 마세요."""),
        ("human", "운동 기록:\n{history}\n\n분석 결과(JSON):")
    ])
    chain = prompt | llm_strict
    
    try:
        result = chain.invoke({"history": history_str})
        print(f"🤖 LLM 원본 응답: {result.content}")
        
        # JSON 추출 시도
        content = result.content.strip()
        
        # JSON 블록이 있는 경우 추출
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            if end != -1:
                content = content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            if end != -1:
                content = content[start:end].strip()
        
        # 첫 번째 중괄호부터 마지막 중괄호까지 추출
        start_brace = content.find("{")
        end_brace = content.rfind("}")
        if start_brace != -1 and end_brace != -1:
            content = content[start_brace:end_brace+1]
        
        analysis = json.loads(content)
        
        # 필수 키 검증
        if "status" not in analysis or "reason" not in analysis:
            raise ValueError("필수 키 누락")
        
        # status 값 검증
        if analysis["status"] not in ["fatigued", "bored", "normal"]:
            analysis["status"] = "normal"
            analysis["reason"] = "잘못된 상태값으로 인한 기본값 적용"
        
        print(f"🧠 피로/지루함 분석 결과: {analysis}")
        return {"fatigue_analysis": analysis}
        
    except json.JSONDecodeError as e:
        print(f"🚨 JSON 파싱 오류: {e}")
        print(f"🚨 원본 응답: {result.content}")
        return {"fatigue_analysis": {"status": "normal", "reason": "JSON 파싱 실패"}}
    except Exception as e:
        print(f"🚨 일반적인 오류: {e}")
        return {"fatigue_analysis": {"status": "normal", "reason": f"분석 오류: {str(e)}"}}

def persona_selection_node(state: ExerciseState) -> dict:
    """사용자로부터 AI 코치 페르소나를 선택받습니다."""
    print("\n--- [Node 4] AI 코치 페르소나 선택 ---")
    print("데이터 분석을 완료했습니다. 어떤 스타일의 코칭을 원하시나요?")
    personas = {
        "1": "다정하고 동기부여 넘치는 코치",
        "2": "데이터를 중시하는 엄격한 트레이너",
        "3": "재미와 습관 형성을 강조하는 친구 같은 코치"
    }
    while True:
        for key, value in personas.items():
            print(f"  {key}. {value}")
        choice = input("> ")
        if choice in personas:
            selected_persona = personas[choice]
            print(f"✅ '{selected_persona}' 코치와 함께 목표를 제안해드릴게요.")
            return {"coach_persona": selected_persona}
        else:
            print("🚨 1, 2, 3 중 하나를 입력해주세요.")

def recommend_new_routine_node(state: ExerciseState) -> dict:
    """피로/지루함이 감지된 사용자에게 새로운 루틴이나 휴식을 제안합니다."""
    print("--- [분기] 새로운 루틴 추천 ---")
    analysis = state['fatigue_analysis']
    
    if analysis.get('status') == 'fatigued':
        recommendation = "최근 운동량이 많아 피로가 누적된 것 같아요. 오늘은 가벼운 스트레칭이나 충분한 휴식을 취해보는 건 어떨까요?"
    elif analysis.get('status') == 'bored':
        recommendation = "매일 비슷한 운동만 해서 조금 지루해지셨나요? 새로운 활력을 위해 '상체 근력 강화' 또는 '유산소 인터벌' 같은 새로운 루틴을 추천해 드릴까요?"
    else:
        recommendation = "분석 중 오류가 발생했습니다."
        
    print(f"🤖 AI 코치 제안: {recommendation}")
    return {}

def predict_slump_node(state: ExerciseState) -> dict:
    """주간/월간 슬럼프 가능성을 예측합니다."""
    print("--- [Node 4a] 주간/월간 슬럼프 예측 ---")
    history = state.get("exercise_history")
    if not history:
        return {"slump_prediction": {"risk": "low", "reason": "데이터 부족"}}

    history_str = pd.DataFrame(history).to_string()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 운동 심리학자입니다. 사용자의 운동 기록 패턴(주기, 완료율 변화 등)을 분석하여 다음 주 또는 다음 달에 슬럼프에 빠질 위험도를 'low', 'medium', 'high'로 예측하고, 그 이유를 간략하게 JSON 형식으로만 답변해주세요. 반드시 아래 형식만 사용하세요:\n\n"
                   "{{\n  \"risk\": \"low\" | \"medium\" | \"high\",\n  \"reason\": \"간단한 설명\"\n}}\n\n"
                   "추가 텍스트는 절대 포함하지 마세요."),
        ("human", "운동 기록:\n{history}\n\n예측 결과(JSON):")
    ])
    chain = prompt | llm_strict
    
    try:
        result = chain.invoke({"history": history_str})
        print(f"🔍 LLM 원본 응답: {result.content}")  # LLM 응답 출력
        prediction = json.loads(result.content)
        print(f"🔮 슬럼프 예측: {prediction}")
        return {"slump_prediction": prediction}
    except json.JSONDecodeError as e:
        print(f"🚨 JSON 파싱 오류: {e}")
        print(f"🚨 원본 응답: {result.content}")  # 원본 응답 출력
        return {"slump_prediction": {"risk": "low", "reason": "분석 실패"}}

def analyze_records_node(state: ExerciseState) -> dict:
    """운동 기록, 슬럼프 예측, 비교 데이터를 종합 분석합니다."""
    print("--- [Node 5] 종합 분석 ---")
    history = state.get('exercise_history')
    if not history:
        return {"analysis_result": "분석할 운동 기록이 없습니다."}
    
    # Java에서 넘어온 날짜 배열(예: [2025, 7, 19])을
    # pandas가 인식할 수 있는 문자열(예: "2025-07-19")로 변환합니다.
    for record in history:
        if isinstance(record.get('exerciseDate'), list) and len(record['exerciseDate']) == 3:
            year, month, day = record['exerciseDate']
            record['exerciseDate'] = f"{year}-{month:02d}-{day:02d}"
    
    df = pd.DataFrame(history)
    df['exerciseDate'] = pd.to_datetime(df['exerciseDate'])
    
    avg_completion = df['completionRate'].mean()
    total_sessions = len(df)
    period = (df['exerciseDate'].max() - df['exerciseDate'].min()).days if total_sessions > 1 else 0
    
    slump_info = state.get("slump_prediction", {}).get("reason", "특별한 징후 없음")
    comparison_info = state.get("comparison_stats", {}).get("comment", "현재 꾸준히 운동 습관을 만들어가고 계시는군요! 잘하고 있어요")
    persona = state.get("coach_persona", "동기부여 전문가")  # 페르소나 가져오기

    # AI 생성형 응답 추가
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"당신은 '{persona}'입니다. 사용자의 운동 기록과 분석 결과를 바탕으로, 사용자를 격려하거나 동기부여할 수 있는 짧은 메시지를 작성해주세요."),
        ("human", f"운동 기록 분석 결과:\n"
                  f"최근 약 {period}일간 총 {total_sessions}회 운동하셨고, "
                  f"평균 완료율은 {avg_completion:.1f}% 입니다. "
                  f"슬럼프 예측 분석 결과는 '{slump_info}'입니다.\n\n"
                  "격려 메시지:")
    ])
    chain = prompt | llm
    try:
        ai_comment = chain.invoke({}).content.strip()
    except Exception as e:
        print(f"🚨 AI 생성형 응답 실패: {e}")
        ai_comment = "운동을 꾸준히 이어가고 계신 점 정말 대단합니다! 앞으로도 화이팅입니다!"

    # 최종 분석 결과
    analysis = (f"최근 약 {period}일간 총 {total_sessions}회 운동하셨고, "
                f"평균 완료율은 {avg_completion:.1f}% 입니다. "
                f"슬럼프 예측 분석 결과는 '{slump_info}'이며, {comparison_info}\n\n"
                f"🤖 AI 코멘트: {ai_comment}")
    print(f"📊 분석 결과: {analysis}")
    return {"analysis_result": analysis}


def suggest_goals_node(state: ExerciseState) -> dict:
    """분석 결과를 바탕으로 LLM에게 목표 제안을 요청합니다."""
    print("--- [Node 7] LLM 목표 제안 ---")
    persona = state.get("coach_persona", "동기를 부여하는 운동 코치")  # 페르소나 가져오기
    
    # 지시사항을 훨씬 더 명확하고 강력하게 수정
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         f"당신은 '{persona}' 성격의 AI 코치입니다. "
         "분석 결과를 바탕으로 '주간 목표'와 '월간 목표'를 제안해주세요. "
         "결과는 반드시 'weekly_goal', 'monthly_goal' 키를 가진 JSON 형식이어야 합니다. "
         "당신의 답변은 오직 유효한 JSON 객체 하나여야 합니다. "
         "어떠한 설명, 인사, 추가 텍스트도 절대 포함하지 마세요."),
        ("human", "분석 결과: {analysis}\n\nJSON:")
    ])
    chain = prompt | llm
    result = chain.invoke({"analysis": state['analysis_result']})
    print(f"🤖 LLM 제안 (Raw): {result.content}")

    # Ensure 'suggested_goals' is always a valid JSON object
    try:
        suggested_goals = json.loads(result.content)
    except json.JSONDecodeError:
        print("🚨 LLM 응답이 유효한 JSON이 아닙니다. 기본값을 사용합니다.")
        suggested_goals = {"weekly_goal": "기본 주간 목표", "monthly_goal": "기본 월간 목표"}

    print(f"🤖 LLM 제안 (Parsed): {suggested_goals}")
    return {"final_goals": suggested_goals}

def clean_json_string(s: str) -> str:
    """LLM이 반환한 문자열에서 JSON 객체만 정확히 추출합니다."""
    # 중괄호 { ... } 사이의 내용만 찾습니다.
    # LLM이 JSON 앞뒤에 추가 텍스트를 붙이는 경우에 대비합니다.
    match = re.search(r'\{.*\}', s, re.DOTALL)
    if match:
        return match.group(0)
    return s

def finalize_goal_node(state: ExerciseState) -> dict:
    """최종 목표를 사용자에게 알리고 결과를 정리합니다."""
    print("\n--- [Node 10] 최종 목표 확정 및 결과 정리 ---")
    final_goals_input = state.get('final_goals')
    
    goals_dict = {}

    # 💡 [핵심 수정] 입력 데이터의 타입에 따라 다르게 처리
    if isinstance(final_goals_input, dict):
        print("... 목표 데이터가 딕셔너리 형태입니다. 그대로 사용합니다.")
        goals_dict = final_goals_input
    elif isinstance(final_goals_input, str):
        print("... 목표 데이터가 문자열입니다. JSON으로 파싱합니다.")
        try:
            cleaned_str = clean_json_string(final_goals_input)
            goals_dict = json.loads(cleaned_str)
        except json.JSONDecodeError:
            print(f"🚨 JSON 파싱 오류 발생. 원본 문자열: {final_goals_input}")
            # 파싱 실패 시, 앱이 중단되지 않도록 기본값 설정
            goals_dict = {"weekly_goal": "파싱 오류", "monthly_goal": final_goals_input}
    else:
        raise TypeError(f"예상치 못한 목표 데이터 타입입니다: {type(final_goals_input)}")

    print("🎉 새로운 목표가 설정되었습니다! 꾸준히 도전해보세요!")
    print(f"  - 주간: {goals_dict.get('weekly_goal', 'N/A')}")
    print(f"  - 월간: {goals_dict.get('monthly_goal', 'N/A')}\n")
    
    # 최종적으로 state의 'final_goals'는 일관성을 위해 딕셔너리 형태로 저장합니다.
    return {"final_goals": goals_dict}

def generate_badge_node(state: ExerciseState) -> dict:
    """목표 달성 시 AI가 개인화된 뱃지를 생성합니다."""
    print("--- [Node 11] AI 뱃지 생성 ---")
    final_goals_data = state.get("final_goals", {})
    
    weekly_goal_description = "주간 목표" # 기본값 설정

    # 💡 [핵심 수정] final_goals 데이터의 타입을 확인하고 처리합니다.
    if isinstance(final_goals_data, dict):
        # 딕셔너리인 경우, 바로 값을 가져옵니다.
        weekly_goal_description = final_goals_data.get("weekly_goal", "주간 목표")
    elif isinstance(final_goals_data, str):
        # 문자열인 경우, 파싱을 시도합니다.
        try:
            goals_dict = json.loads(final_goals_data)
            weekly_goal_description = goals_dict.get("weekly_goal", "주주간 목표")
        except json.JSONDecodeError:
            print(f"🚨 뱃지 생성 중 JSON 파싱 오류. 원본: {final_goals_data}")
            weekly_goal_description = "값진 성과" # 파싱 실패 시 사용할 기본값

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"당신은 창의적이고 유머러스한 동기부여 전문가입니다. 사용자의 운동 기록과 목표를 바탕으로, 유머러스하면서도 운동 기록의 특징을 반영한 '뱃지 이름'과 '뱃지 설명'을 생성해주세요. "
                   "설명은 1~2 문장으로 작성하고, 결과는 'badge_name', 'badge_description' 키를 가진 JSON 형식 문자열로만 답변해주세요."),
        ("human", "달성한 주간 목표: {weekly_goal}\n\n생성된 뱃지 정보(JSON)를 알려주세요.")
    ])
   
    chain = prompt | llm
    result = chain.invoke({"weekly_goal": weekly_goal_description})
   
    try:
        cleaned_content = clean_json_string(result.content)
        badge_info = json.loads(cleaned_content)
        print(f"✨ 생성된 뱃지: {badge_info}")
        return {"generated_badge": badge_info}
    except json.JSONDecodeError:
        print("🚨 뱃지 정보 JSON 파싱 실패")
        return {"generated_badge": {"badge_name": "목표 달성!", "badge_description": "월간 목표를 성공적으로 완수하셨습니다."}}
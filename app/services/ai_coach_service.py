def ai_coach_interaction_service(diagnosis_text: str, recommended_exercise: dict, llm) -> dict:
    """
    AI 코치와의 대화(상담/동기부여) 메시지를 생성하는 서비스 함수.
    Args:
        diagnosis_text (str): 한글 진단 내용
        recommended_exercise (dict): 추천 운동 객체 (예: {'name': '스쿼트', ...})
        llm: LLM 인스턴스 (예: langchain_openai.ChatOpenAI)
    Returns:
        dict: { 'ai_coach_response': str }
    """
    print("[AICoachService] AI 코치 상담 메시지 생성 중...")
    try:
        exercise_name = recommended_exercise.get('name', '운동')
        prompt = f"""
        당신은 AI 피트니스 코치입니다. 아래 정보를 바탕으로 사용자에게 동기부여와 상담 메시지를 생성하세요.
        [진단 내용]
        {diagnosis_text}
        [추천 운동]
        {exercise_name}
        사용자에게 운동의 중요성과 자세 교정의 필요성을 설명하고, 긍정적이고 격려하는 어조로 동기부여를 제공하세요.
        반드시 한국어로 2~3문장으로 답변하세요.
        """
        response = llm.invoke(prompt).content.strip()
        print(f"  > AI 코치 응답: {response}")
        return {"ai_coach_response": response}
    except Exception as e:
        print(f"[AICoachService] 오류: {e}")
        return {"error": f"AI 코치 서비스 오류: {e}"} 
def ai_coach_interaction_service(diagnosis_text: str, recommended_exercise: dict, llm) -> dict:
    print("[AICoachService] AI 코치 상담 메시지 생성 중...")
    """
    AI 코치와의 대화(상담/동기부여) 메시지를 생성하는 서비스 함수.
    Args:
        diagnosis_text (str): 한글 진단 내용
        recommended_exercise (dict): 추천 운동
        llm: LLM 인스턴스 (예: langchain_openai.ChatOpenAI)
    Returns:
        dict: { 'ai_coach_response': str, 'exercise_info': dict }
    """
    try:
        print("[AICoachService] recommended_exercise:", recommended_exercise)
        exercise_name = recommended_exercise.get('name', '운동')
        
        # 운동 정보 추출 - 더 자세한 정보 포함
        exercise_info = {
            'name': exercise_name,
            'thumbnail_url': recommended_exercise.get('thumbnail_url'),
            'description': recommended_exercise.get('description'),
            'url': recommended_exercise.get('url'),
            'difficulty': recommended_exercise.get('difficulty'),
            'category': recommended_exercise.get('category'),
            'target_body_parts': recommended_exercise.get('target_body_parts'),
            'equipment': recommended_exercise.get('equipment'),
            'calories_burn': recommended_exercise.get('calories_burn')
        }
        
        # thumbnail_url이 없는 경우 기본 이미지 설정
        if not exercise_info['thumbnail_url']:
            exercise_info['thumbnail_url'] = "https://res.cloudinary.com/die6y2sez/image/upload/v1751511217/1.jpg"
        
        # URL 처리 디버깅
        print(f"  > [DEBUG] exercise_info['url'] 값: {exercise_info['url']}")
        print(f"  > [DEBUG] exercise_info['url'] 타입: {type(exercise_info['url'])}")
        print(f"  > [DEBUG] exercise_info['url'] is None: {exercise_info['url'] is None}")
        print(f"  > [DEBUG] exercise_info['url'] == '': {exercise_info['url'] == ''}")
        
        # url이 없는 경우에만 기본 URL 설정 (Exercise 테이블에 실제 URL이 있다면 사용하지 않음)
        if not exercise_info['url']:
            exercise_info['url'] = f"https://terms.naver.com/entry.naver?docId=938949&cid=51034&categoryId=51034"
        else:
            print(f"  > Exercise 테이블의 실제 URL 사용: {exercise_info['url']}")
        
        prompt = f"""
        당신은 AI 피트니스 코치입니다. 
        [진단 내용]
        {diagnosis_text}
        [추천 운동]
        {exercise_name}
        위 정보를 바탕으로 사용자에 대한 진단을 자세히 알려주고, 추천운동을 명시해주세요.
        **중요: [추천 운동]에 명시된 운동명을 반드시 그대로 사용하여, 해당 운동을 추천운동으로 안내하세요. 임의로 다른 운동명을 생성하지 마세요.**
        사용자에게 운동의 중요성과 자세 교정의 필요성을 설명하고, 긍정적이고 격려하는 어조로 동기부여를 제공하세요.
        마지막에는 추천운동을 루틴에 추가할거냐는 질문을 해주세요.
        각 소제목은 굵게 표시해주세요.
        """
        response = llm.invoke(prompt).content.strip()
        print(f"  > AI 코치 응답: {response}")
        print(f"  > 응답 길이: {len(response) if response else 0}")
        print(f"  > 응답 타입: {type(response)}")
        
        result = {
            "ai_coach_response": response,
            "exercise_info": exercise_info
        }
        print(f"  > 최종 반환 결과: {result}")
        return result
    except Exception as e:
        print(f"[AICoachService] 오류: {e}")
        return {"error": f"AI 코치 서비스 오류: {e}"} 
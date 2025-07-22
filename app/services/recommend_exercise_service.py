def recommend_exercise_node(diagnosis_text: str, vector_db, llm) -> dict:
    """
    진단 내용을 바탕으로 LLM을 통해 운동 추천 검색어를 생성하고, VectorDB에서 추천 운동을 반환합니다.
    Args:
        diagnosis_text (str): 한글 진단 내용
        vector_db: 운동 벡터 DB 인스턴스
        llm: LLM 인스턴스 (예: langchain_openai.ChatOpenAI)
    Returns:
        dict: { 'recommended_exercise': 운동 객체, 'search_query': 생성된 검색어 }
    """
    print("[RecommendExerciseService] 맞춤 운동 추천 중 (from VectorDB)...")
    try:
        # LLM을 사용해 진단 내용에서 핵심 키워드를 추출하여 검색 쿼리 생성
        prompt = f"""아래의 자세 진단 내용에 가장 적합한 '단 한 가지'의 검색어을 추천해줘. 
        ~난이도, ~효과를 가진, ~부위의, ~운동의 순서로 검색어를 작성해야해.
        VectorDB 검색에 사용할 키워드 문장 오직 한개만 간결하게 한 줄로 답해줘.
        
        [진단 내용]
        {diagnosis_text}
        [출력 예시]
        - 중급 난이도의 유연성을 높이는 효과를 가진 골반 부위의 스트레칭 운동
        [생성된 검색어]
        """
        llm_query = llm.invoke(prompt).content.strip()
        print(f"  > LLM 생성 검색어: '{llm_query}'")
        recommended_list = vector_db.search(llm_query, top_k=1)
        if not recommended_list:
            raise ValueError("VectorDB에서 추천 운동을 찾지 못했습니다.")
        retrieved_exercise = recommended_list[0]
        print(f"  > VectorDB 검색 결과 운동명: '{retrieved_exercise['name']}'")
        return {"recommended_exercise": retrieved_exercise, "search_query": llm_query}
    except Exception as e:
        print(f"[RecommendExerciseService] 오류: {e}")
        return {"error": f"운동 추천 노드 오류: {e}"} 
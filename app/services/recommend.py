def recommend_exercise(diagnosis_text: str, vector_db, llm) -> dict:
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
        
        # Exercise 테이블에서 추가 정보 조회 (Spring Boot API 호출)
        print(f"  > [DEBUG] ===== Spring Boot API 호출 시작 =====")
        print(f"  > [DEBUG] 운동명: {retrieved_exercise['name']}")
        print(f"  > [DEBUG] enrich_exercise_with_db_info 함수 호출 직전")
        
        enriched_exercise = enrich_exercise_with_db_info(retrieved_exercise)
        
        print(f"  > [DEBUG] ===== Spring Boot API 호출 완료 =====")
        print(f"  > [DEBUG] 최종 데이터: {enriched_exercise}")
        
        return {"recommended_exercise": enriched_exercise, "search_query": llm_query}
    except Exception as e:
        print(f"[RecommendExerciseService] 오류: {e}")
        return {"error": f"운동 추천 노드 오류: {e}"}

def enrich_exercise_with_db_info(exercise_data: dict) -> dict:
    """
    VectorDB에서 가져온 운동 정보에 Exercise 테이블의 추가 정보를 병합합니다.
    Args:
        exercise_data (dict): VectorDB에서 가져온 운동 정보
    Returns:
        dict: 병합된 운동 정보
    """
    print(f"  > [DEBUG] ===== enrich_exercise_with_db_info 함수 시작 =====")
    print(f"  > [DEBUG] 입력받은 exercise_data: {exercise_data}")
    print(f"  > [DEBUG] 함수 내부 진입 성공")
    
    try:
        import requests
        print(f"  > [DEBUG] requests 모듈 import 성공")
        
        # Spring Boot API를 통해 운동명으로 Exercise 정보 조회
        exercise_name = exercise_data.get('name', '')
        print(f"  > [DEBUG] 추출된 운동명: '{exercise_name}'")
        
        if not exercise_name:
            print(f"  > [DEBUG] 운동명이 비어있음 - 함수 종료")
            return exercise_data
            
        # Spring Boot API 호출 (운동명으로 검색)
        api_url = "http://localhost:8081/api/exercises/search/name"
        params = {"name": exercise_name}
        
        print(f"  > [DEBUG] ===== Spring Boot API 호출 시작 =====")
        print(f"  > [DEBUG] API URL: {api_url}")
        print(f"  > [DEBUG] API 파라미터: {params}")
        print(f"  > [DEBUG] 요청 전송 중...")
        
        response = requests.get(api_url, params=params, timeout=10)
        
        print(f"  > [DEBUG] ===== Spring Boot API 응답 수신 =====")
        print(f"  > [DEBUG] 응답 상태 코드: {response.status_code}")
        print(f"  > [DEBUG] 응답 헤더: {dict(response.headers)}")
        
        if response.status_code == 200:
            print(f"  > [DEBUG] API 호출 성공 (200)")
            try:
                exercises = response.json()
                print(f"  > [DEBUG] JSON 파싱 성공: {type(exercises)}")
                print(f"  > [DEBUG] 응답 데이터 길이: {len(exercises) if exercises else 0}")
                
                if exercises and len(exercises) > 0:
                    # 첫 번째 매칭되는 운동 정보 사용
                    db_exercise = exercises[0]
                    print(f"  > [DEBUG] 첫 번째 운동 데이터: {db_exercise}")
                    print(f"  > [DEBUG] db_exercise의 모든 키: {list(db_exercise.keys())}")
                    print(f"  > [DEBUG] ===== 각 키별 값 확인 =====")
                    for key in db_exercise.keys():
                        print(f"  > [DEBUG] {key}: {db_exercise.get(key)} (타입: {type(db_exercise.get(key))})")
                    
                    # 기존 VectorDB 데이터와 병합
                    enriched_data = exercise_data.copy()
                    enriched_data.update({
                        'thumbnail_url': db_exercise.get('thumbnailUrl') or db_exercise.get('thumbnail_url'),
                        'url': db_exercise.get('url'),
                        'target_body_parts': db_exercise.get('bodyPart') or db_exercise.get('body_part'),
                        'equipment': db_exercise.get('equipment', '없음'),
                        'calories_burn': db_exercise.get('caloriesBurn') or db_exercise.get('calories_burn', '정보 없음')
                    })
                    
                    # 디버깅: Spring Boot API 응답의 url 필드 상세 확인
                    print(f"  > [DEBUG] ===== URL 필드 상세 분석 =====")
                    print(f"  > [DEBUG] db_exercise.get('url'): {db_exercise.get('url')}")
                    print(f"  > [DEBUG] db_exercise.get('url') 타입: {type(db_exercise.get('url'))}")
                    print(f"  > [DEBUG] db_exercise.get('url') is None: {db_exercise.get('url') is None}")
                    print(f"  > [DEBUG] db_exercise.get('url') == '': {db_exercise.get('url') == ''}")
                    print(f"  > [DEBUG] 'url' in db_exercise: {'url' in db_exercise}")
                    print(f"  > [DEBUG] db_exercise.keys()에 'url' 포함 여부: {'url' in list(db_exercise.keys())}")
                    
                    print(f"  > [DEBUG] ===== 데이터 병합 완료 =====")
                    print(f"  > [DEBUG] - thumbnail_url: {enriched_data.get('thumbnail_url')}")
                    print(f"  > [DEBUG] - url: {enriched_data.get('url')}")
                    print(f"  > [DEBUG] - bodyPart: {enriched_data.get('target_body_parts')}")
                    print(f"  > [DEBUG] - equipment: {enriched_data.get('equipment')}")
                    print(f"  > [DEBUG] - calories_burn: {enriched_data.get('calories_burn')}")
                    
                    return enriched_data
                else:
                    print(f"  > [DEBUG] Exercise 테이블에서 '{exercise_name}' 운동을 찾을 수 없음")
                    print(f"  > [DEBUG] 응답 데이터가 비어있음: {exercises}")
            except Exception as json_error:
                print(f"  > [DEBUG] JSON 파싱 오류: {json_error}")
                print(f"  > [DEBUG] 응답 텍스트: {response.text}")
        else:
            print(f"  > [DEBUG] Spring Boot API 호출 실패: {response.status_code}")
            print(f"  > [DEBUG] 응답 텍스트: {response.text}")
            
    except Exception as e:
        print(f"  > [DEBUG] ===== 오류 발생 =====")
        print(f"  > [DEBUG] 오류 타입: {type(e).__name__}")
        print(f"  > [DEBUG] 오류 메시지: {str(e)}")
        print(f"  > [DEBUG] Exercise 테이블 정보 병합 중 오류: {e}")
    
    print(f"  > [DEBUG] ===== 함수 종료 (기본 데이터 반환) =====")
    # 오류 발생 시 기존 데이터 반환
    return exercise_data 
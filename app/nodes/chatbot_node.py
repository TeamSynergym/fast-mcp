import os
import httpx
import requests
import asyncio
from transformers import pipeline, AutoTokenizer
from pydantic import BaseModel
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

load_dotenv()

# MODEL_NAME = "google/flan-t5-small"
# text_generator = pipeline("text2text-generation", model=MODEL_NAME, device=-1)
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# OLLAMA_API_URL = "http://192.168.2.6:11434/api/chat"
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# def truncate_text_tokenwise(text: str, max_tokens: int = 1024) -> str:
#     tokens = tokenizer.encode(text, truncation=True, max_length=max_tokens)
#     return tokenizer.decode(tokens, clean_up_tokenization_spaces=True)

# --- LangGraph 노드로 작동할 클래스 정의 ---
class ChatbotActionNode:
    """
    유튜브 검색 로직을 수행하는 클래스.
    """
    # async def _ask_model(self, prompt: str, model_name: str) -> str:
    #     messages = [
    #         {"role": "system", "content": "You are a friendly health trainer chatbot. Based on the user's posture diagnosis, provide specific exercise or stretching advice. Respond in English."},
    #         {"role": "user", "content": prompt}
    #     ]
    #     payload = {"model": model_name, "messages": messages, "stream": False}
    #     async with httpx.AsyncClient(timeout=300) as client:
    #         resp = await client.post(OLLAMA_API_URL, json=payload)
    #         resp.raise_for_status()
    #         data = resp.json()
    #         return data.get("message", {}).get("content", "")

    # def _combine_answers(self, answer1: str, answer2: str) -> str:
    #     prompt = f"Combine these two health advice answers into one fluent English answer:\n\nAnswer 1:\n{answer1}\n\nAnswer 2:\n{answer2}\n\nCombined answer:"
    #     prompt = truncate_text_tokenwise(prompt, max_tokens=1024)
    #     result = text_generator(prompt, max_new_tokens=1024, do_sample=False)
    #     combined_text = result[0]['generated_text']
    #     # Truncate the combined answer to 500 tokens
    #     truncated_text = truncate_text_tokenwise(combined_text, max_tokens=500)
    #     return truncated_text

    # def _translate_to_korean(self, text: str) -> str:
    #     url = "https://api.mymemory.translated.net/get"
    #     params = {"q": text, "langpair": "en|ko"}
    #     resp = requests.get(url, params=params)
    #     return resp.json().get("responseData", {}).get("translatedText", "")

    # def _generate_youtube_query(self, combined_answer: str) -> str:
    #     prompt = f"Based on the following health advice, generate a short and specific Youtube keyword in English. Keep it under 6 words and only include exercises or stretches:\n\n{truncate_text_tokenwise(combined_answer, 256)}\n\nSearch query:"
    #     result = text_generator(prompt, max_new_tokens=16, do_sample=False)
    #     query = result[0]['generated_text'].strip()
        
    #     # 생성된 검색어가 비어 있거나, 부적절할 경우 기본값을 반환하도록 수정
    #     if not query or len(query.split()) > 8 or "I'm" in query or "chat" in query:
    #         print("⚠️ 생성된 검색어가 비어있거나 부적절하여 기본 검색어를 사용합니다.")
    #         # 진단 내용이 긍정적일 경우, 자세 유지 운동을 추천
    #         if "great job" in combined_answer.lower() or "good posture" in combined_answer.lower():
    #             return "posture maintenance exercises"
    #         # 그 외의 경우, 일반적인 교정 운동을 추천
    #         return "stretches for posture correction"
    #     return query

    def _search_youtube(self, query: str) -> str:
        params = {
            "part": "snippet",
            "q": query,
            "key": YOUTUBE_API_KEY,
            "type": "video",
            "maxResults": 5,
            "videoEmbeddable": "true",
            "relevanceLanguage": "ko",  # 검색 결과의 언어 한국어
            "regionCode": "KR"        # 검색 지역 대한민국
        }
        try:
            response = requests.get("https://www.googleapis.com/youtube/v3/search", params=params)
            response.raise_for_status()
            items = response.json().get("items", [])
            if not items:
                return "No video found."
            
            for item in items:
                video_id = item["id"]["videoId"]
                if not video_id:
                    continue
                
                try:
                    # 자막 존재 여부 확인 (한국어 또는 영어)
                    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                    transcript_list.find_transcript(['ko', 'en'])
                    
                    # 자막이 있으면 해당 영상 ID를 사용하고 루프 종료
                    print(f"  > 자막이 있는 영상 찾음: {video_id}")
                    youtube_url = f"https://www.youtube.com/watch?v={video_id}"
                    return youtube_url
                
                except (TranscriptsDisabled, NoTranscriptFound):
                    # 자막이 없으면 다음 영상으로 넘어감
                    print(f"  > 자막 없음 (ID: {video_id}). 다음 영상 확인...")
                    continue
            
        except requests.exceptions.RequestException as e:
            # Log for debugging
            print(f"Error during YouTube API call: {e}")
            return "Error fetching video."

    async def run(self, prompt: str) -> dict:
        """클래스의 모든 로직을 실행하는 메인 메소드"""
        search_phrase = prompt.strip()
        
        print(f"  > 유튜브 검색 실행: '{search_phrase}'")
        youtube_url = self._search_youtube(search_phrase)

        return {
            "youtube_url": youtube_url,
            "search_phrase": search_phrase,
        }
import os
import httpx
import requests
import asyncio
from transformers import pipeline, AutoTokenizer
from pydantic import BaseModel
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from typing import Optional, List

load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")


class ChatbotActionNode:
    """
    유튜브 검색 로직을 수행하는 클래스.
    재검색 시 이전에 찾았던 영상을 제외하는 기능 추가
    """
    def _search_youtube(self, query: str, exclude_urls: Optional[List[str]] = None) -> str:
        """
        유튜브에서 자막이 있는 영상을 검색합니다.
        exclude_urls에 포함된 영상은 검색 결과에서 제외됩니다.
        """
        if exclude_urls is None:
            exclude_urls = []
            
        params = {
            "part": "snippet",
            "q": query,
            "key": YOUTUBE_API_KEY,
            "type": "video",
            "maxResults": 10,
            "videoEmbeddable": "true",
            "relevanceLanguage": "ko",
            "regionCode": "KR"
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
                
                # 제외 목록과 비교
                youtube_url = f"https://www.youtube.com/watch?v={video_id}"
                if youtube_url in exclude_urls:
                    print(f"  > 이미 확인한 영상입니다 (ID: {video_id}). 건너뜁니다.")
                    continue
                
                try:
                    # 자막 존재 여부 확인 (한국어 또는 영어)
                    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                    transcript_list.find_transcript(['ko', 'en'])
                    
                    # 자막이 있으면 해당 영상 URL을 반환하고 루프 종료
                    print(f"  > 자막이 있는 새 영상 찾음: {video_id}")
                    return youtube_url
                
                except (TranscriptsDisabled, NoTranscriptFound):
                    # 자막이 없으면 다음 영상으로 넘어감
                    print(f"  > 자막 없음 (ID: {video_id}). 다음 영상 확인...")
                    continue
            
            # 모든 영상을 확인했지만 적합한 영상이 없는 경우
            return "No suitable video found after checking all results."

        except requests.exceptions.RequestException as e:
            print(f"Error during YouTube API call: {e}")
            return "Error fetching video."


    async def run(self, prompt: str, exclude_urls: Optional[List[str]] = None) -> dict:
        """클래스의 모든 로직을 실행하는 메인 메소드"""
        search_phrase = prompt.strip()
        
        print(f"  > 유튜브 검색 실행: '{search_phrase}'")
        youtube_url = self._search_youtube(search_phrase, exclude_urls=exclude_urls)

        return {
            "youtube_url": youtube_url,
            "search_phrase": search_phrase,
        }
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from transformers import pipeline


label_mapping = {
    '기쁨': 'JOY',
    '슬픔': 'SAD',
    '분노': 'ANGER',
    '불안': 'ANXIETY',
    '혐오': 'HATRED',
    '중립': 'NEUTRAL'
}


class EmotionRequest(BaseModel):
    memo: str


class EmotionResponse(BaseModel):
    label: str
    score: float


try:
    emotion_classifier = pipeline("text-classification", model="iuj92/synergym_emotion", tokenizer="iuj92/synergym_emotion")
except Exception as e:
    emotion_classifier = None
    print(f"Error model!!:{e}")


router = APIRouter()


@router.post("/emotion", response_model=EmotionResponse)
def classify_emotion(request: EmotionRequest):
    if not emotion_classifier:
        raise HTTPException(status_code=500, detail="Emotion classifier model error!!")

    # 💡 변경점 1: 빈 메모에 대한 예외 처리 추가
    # memo가 비어있거나 공백만 있으면 'NEUTRAL'을 기본값으로 반환합니다.
    if not request.memo or not request.memo.strip():
        return EmotionResponse(label='NEUTRAL', score=1.0)

    try:
        result = emotion_classifier(request.memo)[0]
        
        # 💡 변경점 2: .get()을 사용하여 KeyError를 방지하고, 예상치 못한 레이블이 나오면 'NEUTRAL'로 처리합니다.
        model_label = result.get('label')
        english_label = label_mapping.get(model_label, 'NEUTRAL')
        
        result['label'] = english_label # 변환된 값으로 업데이트
        print(result)
        
        return EmotionResponse(label=result["label"], score=float(result["score"]))
    except Exception as e:
        # 그 외 예측 불가능한 에러가 발생해도 500 에러를 반환합니다.
        raise HTTPException(status_code=500, detail=str(e))
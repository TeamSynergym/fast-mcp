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
        raise HTTPException(status_code=500, detail="Emotion classifier model error!!");
    try:
        result = emotion_classifier(request.memo)[0]
        result["label"] = label_mapping[result["label"]]
        print(result)
        
        return EmotionResponse(label=result["label"], score=float(result["score"]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

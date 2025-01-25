from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import torch
from textblob import TextBlob
import re

app = FastAPI()

# Paths for the model and tokenizer
model_path = 'itsmrwick/sentiment_model'
tokenizer_path = 'itsmrwick/sentiment_model'

# Load model and tokenizer
try:
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
except Exception as e:
    raise RuntimeError(f"Failed to load model or tokenizer: {str(e)}")


def correct_spelling(text):
    """Correct spelling using TextBlob."""
    corrected_text = TextBlob(text).correct()
    return str(corrected_text)


class ReviewInput(BaseModel):
    text: str


@app.post("/predict")
def predict_sentiment(input: ReviewInput):
    # Validate for empty input
    if not input.text or not input.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    # Reject if special characters exist
    if not re.match(r'^[a-zA-Z0-9\s]+$', input.text.strip()):
        raise HTTPException(status_code=400, detail="Input text must contain only alphanumeric characters and spaces.")

    # Limit input length
    max_length = 512
    if len(input.text) > max_length:
        raise HTTPException(status_code=400, detail=f"Input text cannot be longer than {max_length} characters.")

    # Correct spelling
    corrected_text = correct_spelling(input.text)

    # Tokenize the corrected input
    inputs = tokenizer(corrected_text, return_tensors='pt', truncation=True, max_length=128, padding=True)

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()

    # Map predicted class to sentiment
    sentiment_map = {0: 'Positive', 1: 'Negative', 2: 'Neutral'}
    sentiment = sentiment_map[predicted_class]

    return {
        'original_text': input.text,
        'corrected_text': corrected_text,
        'sentiment': sentiment
    }

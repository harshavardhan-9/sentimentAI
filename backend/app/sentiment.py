import re
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict

# Load tokenizer and model
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Use GPU if available
device = 0 if torch.cuda.is_available() else -1
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)

def clean_text(text: str) -> str:
    """Preprocess text for sentiment analysis."""
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^A-Za-z0-9\s]+', '', text)  # Remove special chars
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespaces
    return text

def analyze_text(text: str) -> Dict:
    """Analyze sentiment with cleaned input and HuggingFace model."""
    try:
        cleaned_text = clean_text(text)
        if not cleaned_text:
            return {"text": text, "sentiment": "neutral", "polarity": 0.0}
        
        result = sentiment_pipeline(cleaned_text)[0]
        sentiment = result['label'].lower()
        polarity = round(result['score'], 4)

        return {
            "text": text,
            "sentiment": sentiment,
            "polarity": polarity
        }

    except Exception as e:
        return {
            "text": text,
            "sentiment": "error",
            "polarity": 0.0,
            "error": str(e)
        }

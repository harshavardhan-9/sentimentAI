from textblob import TextBlob

def analyze_text(text: str) -> dict:
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    
    if polarity > 0:
        sentiment = "positive"
    elif polarity < 0:
        sentiment = "negative"
    else:
        sentiment = "neutral"
        
    return {
        "text": text,
        "sentiment": sentiment,
        "polarity": polarity
    }

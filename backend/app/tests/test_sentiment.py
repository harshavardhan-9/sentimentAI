from ..sentiment import analyze_text

def test_analyze_text():
    result = analyze_text("I love this product")
    assert result["sentiment"] == "positive"
    assert result["polarity"] > 0

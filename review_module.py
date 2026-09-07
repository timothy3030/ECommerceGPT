from pathlib import Path
import pandas as pd
import ollama
from transformers import pipeline

BASE_DIR = Path(__file__).parent
REVIEW_FILE = BASE_DIR / "data" / "reviews.csv"

reviews = pd.read_csv(REVIEW_FILE)

sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)


def analyze_reviews(product_name):
    product_reviews = reviews[
        reviews["ProductName"].str.contains(product_name, case=False, na=False)
    ].copy()

    if product_reviews.empty:
        return "No reviews found for that product."

    def get_sentiment(row):
        result = sentiment_model(row["Review"][:512])[0]

        if row["Rating"] == 3:
            return "NEUTRAL", result["score"]

        return result["label"], result["score"]

    results = product_reviews.apply(get_sentiment, axis=1)
    product_reviews["Sentiment"] = [item[0] for item in results]
    product_reviews["Confidence"] = [item[1] for item in results]

    counts = product_reviews["Sentiment"].value_counts().to_dict()

    review_text = "\n".join(product_reviews["Review"].head(20).tolist())

    prompt = f"""
Summarize customer opinions about {product_name}.

Give:
1. Overall opinion
2. Strengths
3. Weaknesses
4. Buying suggestion

Reviews:
{review_text}
"""

    try:
        response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}]
        )
        summary = response["message"]["content"]

    except Exception:
        summary = "Review summary could not be generated because Ollama is unavailable."

    return (
        f"Positive: {counts.get('POSITIVE', 0)}\n"
        f"Neutral: {counts.get('NEUTRAL', 0)}\n"
        f"Negative: {counts.get('NEGATIVE', 0)}\n\n"
        f"Sample Reviews:\n"
        f"{product_reviews[['Review', 'Sentiment']].head(5).to_string(index=False)}\n\n"
        f"Summary:\n{summary}"
    )
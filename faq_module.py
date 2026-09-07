from pathlib import Path
import numpy as np
import pandas as pd
import ollama
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).parent
FAQ_FILE = BASE_DIR / "data" / "faqs.csv"
EMBEDDING_FILE = BASE_DIR / "models" / "faq_embeddings.npy"

faqs = pd.read_csv(FAQ_FILE)
faq_embeddings = np.load(EMBEDDING_FILE)
model = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve_faq(user_question):
    query_embedding = model.encode([user_question])
    scores = cosine_similarity(query_embedding, faq_embeddings)[0]

    best_index = scores.argmax()

    return {
        "question": faqs.iloc[best_index]["Question"],
        "answer": faqs.iloc[best_index]["Answer"],
        "score": float(scores[best_index])
    }


def answer_faq(user_question, threshold=0.55):
    result = retrieve_faq(user_question)

    if result["score"] < threshold:
        return None, result["score"]

    prompt = f"""
Answer naturally using only this FAQ answer.

Customer Question: {user_question}
FAQ Answer: {result["answer"]}
"""

    try:
        response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"], result["score"]

    except Exception:
        return result["answer"], result["score"]
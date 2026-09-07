from pathlib import Path
import pandas as pd
import ollama
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).parent
PRODUCT_FILE = BASE_DIR / "data" / "products.csv"

products = pd.read_csv(PRODUCT_FILE)

products["search_text"] = (
    products["ProductName"] + " " +
    products["Category"] + " " +
    products["Brand"] + " " +
    products["Description"]
)

vectorizer = TfidfVectorizer(stop_words="english")
product_vectors = vectorizer.fit_transform(products["search_text"])


def retrieve_product(user_question):
    query_vector = vectorizer.transform([user_question])
    scores = cosine_similarity(query_vector, product_vectors)[0]

    best_index = scores.argmax()
    return products.iloc[best_index], float(scores[best_index])


def answer_product_question(user_question, current_product=None):
    if current_product is None:
        product, score = retrieve_product(user_question)
    else:
        product = current_product
        score = 1.0

    context = f"""
Product Name: {product["ProductName"]}
Category: {product["Category"]}
Brand: {product["Brand"]}
Price: ₹{product["Price"]}
Rating: {product["Rating"]}
Description: {product["Description"]}
"""

    prompt = f"""
Answer only from this product context.
If the answer is not present, say:
"I do not have that information."

Product Context:
{context}

Customer Question: {user_question}
"""

    try:
        response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"], product, score

    except Exception:
        return context, product, score
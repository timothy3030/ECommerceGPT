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


def get_top_products(user_query, top_n=3):
    query_vector = vectorizer.transform([user_query])
    scores = cosine_similarity(query_vector, product_vectors)[0]

    top_indices = scores.argsort()[-top_n:][::-1]
    return products.iloc[top_indices]


def recommend_product(user_query):
    top_products = get_top_products(user_query)

    context = top_products[
        ["ProductName", "Brand", "Price", "Rating", "Description"]
    ].to_string(index=False)

    prompt = f"""
Recommend the best product for this request: {user_query}

Use only these retrieved products:
{context}

Mention product name, brand, price, and a short reason.
"""

    try:
        response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]

    except Exception:
        product = top_products.iloc[0]
        return (
            f"I recommend {product['ProductName']} by {product['Brand']}. "
            f"Price: ₹{product['Price']}. Rating: {product['Rating']}."
        )
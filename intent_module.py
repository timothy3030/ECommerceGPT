import ollama


def fallback_intent(question):
    question = question.lower()

    if any(word in question for word in ["review", "opinion", "customers think"]):
        return "REVIEW"

    if any(word in question for word in ["recommend", "suggest", "best laptop", "best mobile"]):
        return "RECOMMENDATION"

    if any(word in question for word in ["price", "brand", "tell me about", "specification"]):
        return "PRODUCT"

    return "FAQ"


def detect_intent(question):
    prompt = f"""
Classify this customer question into exactly one category:

FAQ
RECOMMENDATION
PRODUCT
REVIEW

Question: {question}

Reply with only one category name.
"""

    try:
        response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}]
        )

        intent = response["message"]["content"].strip().upper()

        if intent in ["FAQ", "RECOMMENDATION", "PRODUCT", "REVIEW"]:
            return intent

    except Exception:
        pass

    return fallback_intent(question)
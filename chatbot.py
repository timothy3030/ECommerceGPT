from faq_module import answer_faq
from recommendation_module import recommend_product
from product_module import answer_product_question
from review_module import analyze_reviews
from intent_module import detect_intent


def chatbot():
    print("E-Commerce GPT Chatbot started.")
    print("Type Exit whenever you want to stop.")

    current_product = None

    while True:
        question = input("\nYou: ").strip()

        if question.lower() == "exit":
            print("Assistant: Bye gng!")
            break

        if current_product is not None and any(
            word in question.lower()
            for word in ["its price", "its brand", "its rating", "this product"]
        ):
            intent = "PRODUCT"
        else:
            intent = detect_intent(question)

        print("Detected Intent:", intent)

        if intent == "FAQ":
            response, score = answer_faq(question)

            if response:
                print(f"Similarity Score: {score:.2f}")
                print("Assistant:", response)
            else:
                print("Assistant: Sorry, I could not find a matching FAQ.")

        elif intent == "RECOMMENDATION":
            print("Assistant:", recommend_product(question))

        elif intent == "PRODUCT":
            response, current_product, score = answer_product_question(
                question,
                current_product
            )
            print(f"Retrieved Product: {current_product['ProductName']}")
            print(f"Similarity: {score:.2f}")
            print("Assistant:", response)

        elif intent == "REVIEW":
            product_name = input("Enter product name: ")
            print(analyze_reviews(product_name))


if __name__ == "__main__":
    chatbot()
# AI-Powered E-Commerce Chatbot using Ollama

This project is a modular RAG-based e-commerce chatbot developed for the GPT-4 lab experiment. It answers FAQs, recommends products, answers product-related questions, analyzes customer reviews, and detects the user’s intent using Ollama.

## Features

- Semantic FAQ search using embeddings and cosine similarity
- Product recommendation using TF-IDF
- Product question answering using retrieved product context
- Customer review sentiment analysis
- Review summary and buying suggestion
- Intent detection: FAQ, Recommendation, Product, or Review
- Continuous chatbot interaction until the user types `Exit`

## Project Structure

```text
ECommerceGPT/
├── chatbot.py
├── create_embeddings.py
├── faq_module.py
├── recommendation_module.py
├── product_module.py
├── review_module.py
├── intent_module.py
├── requirements.txt
├── data/
│   ├── faqs.csv
│   ├── products.csv
│   └── reviews.csv
├── models/
└── screenshots/
```

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

Download the Ollama model:

```bash
ollama pull llama3.2
```

## Run the Project

First create FAQ embeddings:

```bash
python create_embeddings.py
```

Then start the chatbot:

```bash
python chatbot.py
```

## Sample Queries

```text
Where is my parcel?
Recommend a laptop
Tell me about Lenovo Laptop
What is its price?
What do customers think about Lenovo Laptop?
Exit
```

## Lab Question Mapping

| Questions | Files |
|---|---|
| Q1-Q3 | `create_embeddings.py` |
| Q4-Q5 | `faq_module.py` |
| Q6-Q7 | `recommendation_module.py` |
| Q8-Q9 | `product_module.py` |
| Q10-Q11 | `review_module.py` |
| Q12 | `intent_module.py` |
| Q13 | `chatbot.py` |

## Screenshots

Execution screenshots are available in the `screenshots` folder.
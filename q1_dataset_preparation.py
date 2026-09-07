import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# Load datasets
faqs = pd.read_csv(DATA_DIR / "faqs.csv")
products = pd.read_csv(DATA_DIR / "products.csv")
reviews = pd.read_csv(DATA_DIR / "reviews.csv")

# FAQ dataset
print("\n========== FAQ DATASET ==========")
print("\nFirst 5 records:")
print(faqs.head())

print("\nRows and Columns:")
print(faqs.shape)

print("\nAttributes:")
print(list(faqs.columns))


# Products dataset
print("\n========== PRODUCTS DATASET ==========")
print("\nFirst 5 records:")
print(products.head())

print("\nRows and Columns:")
print(products.shape)

print("\nAttributes:")
print(list(products.columns))


# Reviews dataset
print("\n========== REVIEWS DATASET ==========")
print("\nFirst 5 records:")
print(reviews.head())

print("\nRows and Columns:")
print(reviews.shape)

print("\nAttributes:")
print(list(reviews.columns))
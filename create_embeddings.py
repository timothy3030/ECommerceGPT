from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).parent
FAQ_FILE = BASE_DIR / "data" / "faqs.csv"
MODEL_DIR = BASE_DIR / "models"
EMBEDDING_FILE = MODEL_DIR / "faq_embeddings.npy"

faqs = pd.read_csv(FAQ_FILE)

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(faqs["Question"].tolist(), convert_to_numpy=True)

MODEL_DIR.mkdir(exist_ok=True)
np.save(EMBEDDING_FILE, embeddings)

print("FAQ embeddings created successfully.")
print("Saved at:", EMBEDDING_FILE)
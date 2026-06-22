import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

vectorizer_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "models",
    "vectorizer.pkl"
)
model_path = os.path.join(BASE_DIR, "..", "models", "model.pkl")

vectorizer = pickle.load(open(vectorizer_path, "rb"))
model = pickle.load(open(model_path, "rb"))



print("Path:", vectorizer_path)
print("Exists:", os.path.exists(vectorizer_path))
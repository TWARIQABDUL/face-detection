import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_trained_model(model_path="models/classifier.pkl"):
    return joblib.load(model_path)

def predict_embedding(embedding: np.ndarray, artifact: dict, sim_threshold: float = 0.55):
    sims = cosine_similarity(embedding.reshape(1, -1), artifact["embeddings"])
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])
    if best_sim < sim_threshold:
        return "Unknown", best_sim
    if artifact["model_type"] == "svm":
        return str(artifact["classifier"].predict(embedding.reshape(1, -1))), best_sim
    return str(artifact["labels"][best_idx]), best_sim

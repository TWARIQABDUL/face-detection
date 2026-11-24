import os
import time
import shutil
import tempfile
import cv2
import joblib
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from sklearn.metrics.pairwise import cosine_similarity

from src.detector import FaceDetectorEmbedder
from src.train import train_classifier

MODEL_PATH = "models/classifier.pkl"
KNOWN_DIR = "data/known_faces"
START_TIME = time.time()

app = FastAPI(
    title="HITL Face Recognition API",
    description="REST API for single-face inference, bulk data ingestion, and on-demand model retraining.",
    version="1.0.0",
)

detector = FaceDetectorEmbedder(det_size=(640, 640), min_face_size=50, blur_threshold=15.0)


def get_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


@app.get("/health")
def health_check():
    """Returns model uptime, status, and loaded classes."""
    artifact = get_model()
    uptime_sec = round(time.time() - START_TIME, 2)
    return {
        "status": "online",
        "uptime_seconds": uptime_sec,
        "model_loaded": artifact is not None,
        "classes": artifact["classes"] if artifact else [],
        "total_training_samples": len(artifact["labels"]) if artifact else 0,
    }


@app.post("/predict")
def predict_single(file: UploadFile = File(...), sim_threshold: float = Form(0.55)):
    """Predicts identity for a single uploaded image datapoint."""
    artifact = get_model()
    if artifact is None:
        raise HTTPException(status_code=400, detail="Model not trained yet.")

    contents = file.file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    t0 = time.time()
    faces = detector.extract_faces(frame)
    predictions = []

    for face in faces:
        emb = face["embedding"].reshape(1, -1)
        sims = cosine_similarity(emb, artifact["embeddings"])
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim < sim_threshold:
            label = "Unknown"
        elif artifact["model_type"] == "svm":
            label = str(artifact["classifier"].predict(emb))
        else:
            label = str(artifact["labels"][best_idx])

        predictions.append({
            "label": label,
            "similarity_score": round(best_sim, 4),
            "detection_confidence": round(face["det_score"], 4),
            "bbox": face["bbox"],
        })

    latency_ms = round((time.time() - t0) * 1000, 2)
    return {
        "filename": file.filename,
        "faces_detected": len(predictions),
        "predictions": predictions,
        "inference_latency_ms": latency_ms,
    }


@app.post("/upload-and-retrain")
async def upload_and_retrain(
    person_name: str = Form(...),
    files: list[UploadFile] = File(...),
):
    """Bulk uploads labeled images for a person and triggers classifier retraining."""
    clean_name = person_name.strip().replace(" ", "_")
    if not clean_name:
        raise HTTPException(status_code=400, detail="person_name cannot be empty.")

    dest_dir = os.path.join(KNOWN_DIR, clean_name)
    os.makedirs(dest_dir, exist_ok=True)

    saved_count = 0
    for f in files:
        contents = await f.read()
        np_arr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is not None:
            out_path = os.path.join(dest_dir, f"{int(time.time() * 1000)}_{f.filename}")
            cv2.imwrite(out_path, frame)
            saved_count += 1

    train_res = train_classifier(known_dir=KNOWN_DIR, model_path=MODEL_PATH, detector=detector)
    return {
        "uploaded_for": clean_name,
        "images_saved": saved_count,
        "retrain_result": train_res,
    }

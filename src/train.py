import os
import cv2
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from src.detector import FaceDetectorEmbedder


def train_classifier(
    known_dir="data/known_faces",
    model_path="models/classifier.pkl",
    embeddings_path="data/embeddings/face_embeddings.pkl",
    detector: FaceDetectorEmbedder = None,
):
    print("Starting training...")
    
    if detector is None:
        detector = FaceDetectorEmbedder()

    X, y = [], []

    for person_name in sorted(os.listdir(known_dir)):
        person_dir = os.path.join(known_dir, person_name)
        if not os.path.isdir(person_dir):
            continue

        for img_name in os.listdir(person_dir):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue

            img_path = os.path.join(person_dir, img_name)
            frame = cv2.imread(img_path)
            if frame is None:
                continue

            # Try detecting on the image as-is first
            faces = detector.app.get(frame)

            # If it's a tight face crop and detector misses it, add padding around the image
            if len(faces) == 0:
                h, w, _ = frame.shape
                pad_y, pad_x = int(h * 0.5), int(w * 0.5)
                padded = cv2.copyMakeBorder(
                    frame, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=[0, 0, 0]
                )
                faces = detector.app.get(padded)

            if len(faces) > 0:
                largest = max(
                    faces,
                    key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                )
                X.append(largest.normed_embedding)
                y.append(person_name)

    if len(X) == 0:
        return {"status": "empty", "message": "No valid face embeddings found in known_faces."}

    X = np.array(X)
    y = np.array(y)
    unique_classes = np.unique(y)

    if len(unique_classes) >= 2:
        print("Using Support Vector Classifier")
        clf = SVC(kernel="linear", probability=True, C=1.0)
        clf.fit(X, y)
        model_type = "svm"
    else:
        print("Using KNN")
        clf = KNeighborsClassifier(n_neighbors=1, metric="cosine")
        clf.fit(X, y)
        model_type = "knn"

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)

    artifact = {
        "classifier": clf,
        "model_type": model_type,
        "classes": unique_classes.tolist(),
        "embeddings": X,
        "labels": y,
    }
    joblib.dump(artifact, model_path)
    joblib.dump({"X": X, "y": y}, embeddings_path)

    return {
        "status": "success",
        "num_samples": len(X),
        "classes": unique_classes.tolist(),
        "model_type": model_type,
    }
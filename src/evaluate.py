import os
import json
import cv2
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Safe non-GUI backend for CLI & Streamlit
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from sklearn.metrics.pairwise import cosine_similarity
from src.detector import FaceDetectorEmbedder


def generate_evaluation_plots(y_true, y_pred, best_sims, sim_threshold, plot_dir="data/logs/plots"):
    os.makedirs(plot_dir, exist_ok=True)
    labels = sorted(list(set(y_true) | set(y_pred)))
    saved_paths = {}

    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
    )
    plt.title("Face Recognition Confusion Matrix")
    plt.xlabel("Predicted Identity")
    plt.ylabel("Actual Identity")
    plt.tight_layout()
    cm_path = os.path.join(plot_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    saved_paths["confusion_matrix"] = cm_path

    # 2. Cosine Similarity Distribution (Known vs Unknown)
    plt.figure(figsize=(7, 4))
    known_sims = [s for s, t in zip(best_sims, y_true) if t.lower() != "unknown"]
    unknown_sims = [s for s, t in zip(best_sims, y_true) if t.lower() == "unknown"]

    if known_sims:
        sns.histplot(known_sims, bins=15, kde=len(known_sims) > 1, color="#2b8cbe", label="Known Persons", alpha=0.6)
    if unknown_sims:
        sns.histplot(unknown_sims, bins=15, kde=len(unknown_sims) > 1, color="#e34a33", label="Unknown / Strangers", alpha=0.6)

    plt.axvline(sim_threshold, color="black", linestyle="--", linewidth=1.5, label=f"Threshold ({sim_threshold})")
    plt.title("Max Cosine Similarity Distribution")
    plt.xlabel("Cosine Similarity Score")
    plt.ylabel("Image Count")
    plt.xlim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    sim_path = os.path.join(plot_dir, "similarity_distribution.png")
    plt.savefig(sim_path, dpi=300)
    plt.close()
    saved_paths["similarity_distribution"] = sim_path

    # 3. Per-Class Precision, Recall, F1 Bar Chart
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(7, 4))
    plt.bar(x - width, prec, width, label="Precision", color="#1b9e77")
    plt.bar(x, rec, width, label="Recall", color="#d95f02")
    plt.bar(x + width, f1, width, label="F1-Score", color="#7570b3")

    plt.xticks(x, labels, rotation=15)
    plt.ylim(0, 1.1)
    plt.ylabel("Score")
    plt.title("Per-Class Evaluation Metrics")
    plt.legend(loc="lower right")
    plt.tight_layout()
    metrics_path = os.path.join(plot_dir, "class_metrics.png")
    plt.savefig(metrics_path, dpi=300)
    plt.close()
    saved_paths["class_metrics"] = metrics_path

    return saved_paths


def evaluate_model(
    test_dir="data/test_set",
    model_path="models/classifier.pkl",
    sim_threshold=0.55,
    detector: FaceDetectorEmbedder = None,
    plot_dir="data/logs/plots",
):
    if not os.path.exists(model_path):
        return {"status": "error", "message": "No trained model found. Train the model first."}

    if not os.path.exists(test_dir) or not os.listdir(test_dir):
        return {
            "status": "error",
            "message": f"Test directory '{test_dir}' is empty. Add subfolders per person (and 'Unknown' for strangers).",
        }

    if detector is None:
        detector = FaceDetectorEmbedder()

    artifact = joblib.load(model_path)
    known_embs = artifact["embeddings"]
    known_labels = artifact["labels"]
    clf = artifact["classifier"]
    model_type = artifact["model_type"]

    y_true, y_pred, best_sims = [], [], []

    for folder_name in sorted(os.listdir(test_dir)):
        folder_path = os.path.join(test_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        for img_name in os.listdir(folder_path):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue

            frame = cv2.imread(os.path.join(folder_path, img_name))
            if frame is None:
                continue

            faces = detector.app.get(frame)
            if not faces:
                continue

            largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            emb = largest.normed_embedding.reshape(1, -1)

            sims = cosine_similarity(emb, known_embs)[0]
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])

            if best_sim < sim_threshold:
                pred = "Unknown"
            elif model_type == "svm":
                pred = str(clf.predict(emb)[0])
            else:
                pred = str(known_labels[best_idx])

            y_true.append(folder_name)
            y_pred.append(pred)
            best_sims.append(best_sim)

    if not y_true:
        return {"status": "error", "message": "No detectable faces found in the test set images."}

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Calculate Unknown Rejection Rate & False Acceptance Rate (FAR)
    unknown_indices = [i for i, label in enumerate(y_true) if label.lower() == "unknown"]
    if unknown_indices:
        correctly_rejected = sum(1 for i in unknown_indices if y_pred[i].lower() == "unknown")
        unknown_rejection_rate = correctly_rejected / len(unknown_indices)
        false_acceptance_rate = 1.0 - unknown_rejection_rate
    else:
        unknown_rejection_rate = None
        false_acceptance_rate = None

    report_text = classification_report(y_true, y_pred, zero_division=0)
    plot_paths = generate_evaluation_plots(y_true, y_pred, best_sims, sim_threshold, plot_dir=plot_dir)

    metrics = {
        "status": "success",
        "total_test_samples": len(y_true),
        "sim_threshold": sim_threshold,
        "accuracy": round(float(acc), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "unknown_rejection_rate": round(float(unknown_rejection_rate), 4) if unknown_rejection_rate is not None else "N/A (No Unknown folder)",
        "false_acceptance_rate": round(float(false_acceptance_rate), 4) if false_acceptance_rate is not None else "N/A",
        "classification_report": report_text,
        "plots": plot_paths,
    }

    os.makedirs("data/logs", exist_ok=True)
    with open("data/logs/latest_evaluation.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    res = evaluate_model()
    print(json.dumps(res, indent=2))

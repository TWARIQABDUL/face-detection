import os
import time
import cv2
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from src.detector import FaceDetectorEmbedder
from src.ingestion import MediaIngestor
from src.queue_manager import UnknownQueueManager
from src.train import train_classifier
from src.logger import RecognitionLogger
from src.evaluate import evaluate_model

MODEL_PATH = "models/classifier.pkl"
KNOWN_DIR = "data/known_faces"
QUEUE_DIR = "data/unknown_queue"
TEST_DIR = "data/test_set"
START_TIME = time.time()

detector = FaceDetectorEmbedder(det_size=(640, 640), min_face_size=50, blur_threshold=15.0)
queue_manager = UnknownQueueManager(queue_dir=QUEUE_DIR, known_dir=KNOWN_DIR)
logger = RecognitionLogger()


def load_model():
    return joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None


def get_status_markdown():
    hrs, rem = divmod(int(time.time() - START_TIME), 3600)
    mins, secs = divmod(rem, 60)
    uptime = f"{hrs:02d}h {mins:02d}m {secs:02d}s"
    artifact = load_model()
    if artifact:
        classes = ", ".join(artifact["classes"])
        return (
            f"**Model Uptime:** `{uptime}` | **Status:** `Online ({artifact['model_type'].upper()})` | "
            f"**Classes ({len(artifact['classes'])}):** {classes} | **Samples:** {len(artifact['labels'])}"
        )
    return f"**Model Uptime:** `{uptime}` | **Status:** `No trained model yet`"


def predict_face(embedding: np.ndarray, artifact: dict, sim_threshold: float):
    if artifact is None:
        return "Unknown", 0.0
    sims = cosine_similarity(embedding.reshape(1, -1), artifact["embeddings"])[0]
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])
    if best_sim < sim_threshold:
        return "Unknown", best_sim
    if artifact["model_type"] == "svm":
        return str(artifact["classifier"].predict(embedding.reshape(1, -1))[0]), best_sim
    return str(artifact["labels"][best_idx]), best_sim


def process_uploaded_images(files, sim_threshold, blur_threshold):
    if not files:
        return "No images uploaded.", [], get_status_markdown()
    detector.blur_threshold = blur_threshold
    ingestor = MediaIngestor(detector=detector)
    paths = [f.name if hasattr(f, "name") else f for f in files]
    faces = ingestor.process_images(paths)

    artifact = load_model()
    gallery = []
    queued = 0
    for face in faces:
        label, score = predict_face(face["embedding"], artifact, sim_threshold)
        face["pred_label"] = label
        face["pred_score"] = score
        if label == "Unknown":
            st_msg = queue_manager.add_unknown_face(face)
            face["queue_status"] = st_msg
            if not st_msg.startswith("skipped_duplicate"):
                queued += 1
        else:
            face["queue_status"] = "recognized"
        rgb = cv2.cvtColor(face["crop"], cv2.COLOR_BGR2RGB)
        gallery.append((rgb, f"{label} ({score:.2f})"))

    logger.log_batch(faces)
    summary = f"Detected **{len(faces)}** faces | Recognized: **{len(faces) - queued}** | Queued Unknown: **{queued}**"
    return summary, gallery, get_status_markdown()


def process_uploaded_video(video_path, sim_threshold, blur_threshold, sample_fps):
    if not video_path:
        return "No video uploaded.", [], get_status_markdown()
    detector.blur_threshold = blur_threshold
    ingestor = MediaIngestor(detector=detector, sample_fps=sample_fps)
    faces = ingestor.process_video(video_path)

    artifact = load_model()
    gallery = []
    queued = 0
    for face in faces:
        label, score = predict_face(face["embedding"], artifact, sim_threshold)
        face["pred_label"] = label
        face["pred_score"] = score
        if label == "Unknown":
            st_msg = queue_manager.add_unknown_face(face)
            face["queue_status"] = st_msg
            if not st_msg.startswith("skipped_duplicate"):
                queued += 1
        else:
            face["queue_status"] = "recognized"
        rgb = cv2.cvtColor(face["crop"], cv2.COLOR_BGR2RGB)
        gallery.append((rgb, f"{label} ({score:.2f}) @ {face['timestamp_sec']}s"))

    logger.log_batch(faces)
    summary = f"Extracted **{len(faces)}** video face crops | Recognized: **{len(faces) - queued}** | Queued Unknown: **{queued}**"
    return summary, gallery, get_status_markdown()


def get_unknown_clusters():
    if not os.path.exists(QUEUE_DIR):
        return gr.Dropdown(choices=[], value=None), []
    clusters = sorted([d for d in os.listdir(QUEUE_DIR) if os.path.isdir(os.path.join(QUEUE_DIR, d))])
    gallery = []
    for cid in clusters:
        cpath = os.path.join(QUEUE_DIR, cid)
        imgs = sorted([f for f in os.listdir(cpath) if f.endswith((".jpg", ".png", ".jpeg"))])
        for img_name in imgs[:3]:
            bgr = cv2.imread(os.path.join(cpath, img_name))
            if bgr is not None:
                gallery.append((cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), f"Cluster: {cid}"))
    return gr.Dropdown(choices=clusters, value=clusters[0] if clusters else None), gallery


def label_and_retrain(cluster_id, person_name):
    if not cluster_id or not person_name.strip():
        dd, gal = get_unknown_clusters()
        return "Select a cluster and enter a person name.", dd, gal, get_status_markdown()
    queue_manager.promote_cluster(cluster_id, person_name)
    res = train_classifier(known_dir=KNOWN_DIR, model_path=MODEL_PATH, detector=detector)
    dd, gal = get_unknown_clusters()
    return f"Labeled '{person_name}' and retrained! Status: {res}", dd, gal, get_status_markdown()


def force_retrain():
    res = train_classifier(known_dir=KNOWN_DIR, model_path=MODEL_PATH, detector=detector)
    return f"Retrain result: {res}", get_status_markdown()


def generate_visualizations():
    records = []
    if os.path.exists(KNOWN_DIR):
        for person in sorted(os.listdir(KNOWN_DIR)):
            pdir = os.path.join(KNOWN_DIR, person)
            if not os.path.isdir(pdir):
                continue
            for fname in os.listdir(pdir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    bgr = cv2.imread(os.path.join(pdir, fname))
                    if bgr is not None:
                        sharp = float(cv2.Laplacian(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
                        records.append({"Person": person, "Sharpness": sharp})

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    if records:
        df = pd.DataFrame(records)
        counts = df["Person"].value_counts()
        sns.barplot(x=counts.index, y=counts.values, hue=counts.index, ax=axes[0], palette="viridis", legend=False)
        axes[0].set_title("1. Class Sample Distribution")
        axes[0].tick_params(axis="x", rotation=25)

        sns.histplot(df["Sharpness"], bins=12, kde=True, ax=axes[1], color="#2b8cbe")
        axes[1].axvline(15.0, color="red", linestyle="--", label="Blur Gate (15.0)")
        axes[1].set_title("2. Laplacian Sharpness Distribution")
        axes[1].legend()

    artifact = load_model()
    if artifact and len(artifact["embeddings"]) >= 2:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(artifact["embeddings"])
        var_pct = pca.explained_variance_ratio_.sum() * 100
        pca_df = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "Identity": artifact["labels"]})
        sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Identity", s=90, ax=axes[2])
        axes[2].set_title(f"3. 2D PCA Embeddings ({var_pct:.1f}% Var)")

    plt.tight_layout()
    interp = (
        "### Feature Interpretations\n"
        "1. **Class Sample Distribution:** Tracks sample balance across enrolled identities to avoid SVM hyperplane bias.\n"
        "2. **Laplacian Sharpness Distribution:** Verifies training images exceed the sharpness quality gate (`15.0`).\n"
        "3. **2D PCA Projection of 512-D Embeddings:** Demonstrates tight intra-class clustering and clear inter-class separation."
    )
    return fig, interp


def run_eval_and_logs():
    res = evaluate_model(test_dir=TEST_DIR, model_path=MODEL_PATH, sim_threshold=0.55, detector=detector)
    logs_df = logger.load_logs()
    return str(res), logs_df.tail(25)


with gr.Blocks(title="HITL Face Recognition Pipeline") as demo:
    gr.Markdown("# Human-in-the-Loop Face Recognition Pipeline")
    status_md = gr.Markdown(get_status_markdown())
    refresh_btn = gr.Button("Refresh Uptime & Model Status", size="sm")
    refresh_btn.click(fn=get_status_markdown, outputs=status_md)

    with gr.Row():
        sim_slider = gr.Slider(0.35, 0.85, value=0.55, step=0.05, label="Recognition Threshold (Cosine Similarity)")
        blur_slider = gr.Slider(5.0, 100.0, value=15.0, step=5.0, label="Min Sharpness (Blur Threshold)")
        fps_slider = gr.Slider(0.5, 5.0, value=1.5, step=0.5, label="Video Sampling Rate (FPS)")

    with gr.Tab("1. Single/Bulk Image & Video Inference"):
        with gr.Row():
            with gr.Column():
                img_input = gr.File(file_count="multiple", label="Upload Single or Bulk Images")
                img_btn = gr.Button("Predict / Process Images", variant="primary")
            with gr.Column():
                vid_input = gr.Video(label="Upload Video Clip")
                vid_btn = gr.Button("Extract & Recognize Video Faces", variant="primary")
        inf_summary = gr.Markdown()
        inf_gallery = gr.Gallery(label="Detected & Recognized Faces", columns=5, height=300)

        img_btn.click(process_uploaded_images, [img_input, sim_slider, blur_slider], [inf_summary, inf_gallery, status_md])
        vid_btn.click(process_uploaded_video, [vid_input, sim_slider, blur_slider, fps_slider], [inf_summary, inf_gallery, status_md])

    with gr.Tab("2. Label Unknown Faces & Retrain"):
        scan_btn = gr.Button("Refresh Unknown Queue")
        cluster_dd = gr.Dropdown(label="Select Unknown Cluster ID")
        queue_gallery = gr.Gallery(label="Queued Unknown Face Crops", columns=5, height=250)
        with gr.Row():
            name_box = gr.Textbox(label="Assign Person Name", placeholder="e.g., Alice")
            retrain_btn = gr.Button("Label & Retrain Model", variant="primary")
            force_btn = gr.Button("Force Retrain on Known Dataset")
        retrain_out = gr.Textbox(label="Training Output")
        scan_btn.click(get_unknown_clusters, outputs=[cluster_dd, queue_gallery])
        retrain_btn.click(label_and_retrain, [cluster_dd, name_box], [retrain_out, cluster_dd, queue_gallery, status_md])
        force_btn.click(force_retrain, outputs=[retrain_out, status_md])

    with gr.Tab("3. Dataset Feature Visualizations"):
        viz_btn = gr.Button("Generate Dataset Visualizations", variant="primary")
        viz_plot = gr.Plot(label="Dataset Feature Charts")
        viz_text = gr.Markdown()
        viz_btn.click(generate_visualizations, outputs=[viz_plot, viz_text])

    with gr.Tab("4. Evaluation Metrics & Logs"):
        eval_btn = gr.Button("Run Evaluation on Test Set & Load Logs", variant="primary")
        eval_out = gr.Textbox(label="Evaluation Metrics JSON", lines=8)
        logs_table = gr.Dataframe(label="Recent Inference & Timestamp Logs")
        eval_btn.click(run_eval_and_logs, outputs=[eval_out, logs_table])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

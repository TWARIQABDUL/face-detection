---
title: HITL Face Recognition Pipeline
emoji: 👤
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: gradio_app.py
pinned: false
---

# Human-in-the-Loop (HITL) Face Recognition Pipeline

An end-to-end machine learning pipeline for **open-set face recognition** supporting **single/bulk image uploads**, **video frame extraction**, a **FastAPI REST API**, and a **Streamlit monitoring & retraining dashboard**. When the model encounters an unrecognized face, it deduplicates and clusters the unknown crops into a review queue, allowing a human operator to assign labels and retrain the classifier on demand.

---

## Live Links & Demo

* **Video Demo (YouTube):** `[Add Your YouTube Demo Link Here]`
* **Live Cloud Deployment URL:** `[Add Your Render / Hugging Face Space URL Here]`
* **GitHub Repository:** `[Add Your GitHub Repo URL Here]`

---

## System Architecture

```text
+-----------------------------------------------------------------------+

| 1. INGESTION & API LAYER (src/ingestion.py, api.py, app.py)           |
|    - Single & Bulk Image Upload (.jpg, .png, .webp)                   |
|    - Video Frame Sampling (OpenCV @ configurable FPS)                 |
|    - FastAPI REST Endpoints (/predict, /upload-and-retrain, /health)  |
+-----------------------------------+-----------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+

| 2. DETECTION, QUALITY GATING & EMBEDDING (src/detector.py)            |
|    - RetinaFace Detection + Bounding Box Padding                      |
|    - Minimum Size & Laplacian Sharpness (Blur) Filtering              |
|    - ArcFace (ResNet-50) -> 512-D L2-Normalized Feature Embedding     |
+-----------------------------------+-----------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+

| 3. OPEN-SET CLASSIFICATION (models/classifier.pkl)                    |
|    - Cosine Similarity Check vs. Known Reference Embeddings           |
+-----------------+-----------------------------------+-----------------+

                  | Similarity >= Threshold           | Similarity < Threshold
                  v                                   v
+-----------------------------------+ +---------------------------------+

| 4a. RECOGNIZED IDENTITY           | | 4b. UNKNOWN QUEUE & CLUSTERING  |
|     - Predict Name (SVM / 1-NN)   | |     (src/queue_manager.py)      |
|     - Log Timestamp & Confidence  | |     - Group Similar Unknowns    |
|       to recognition_logs.csv     | |     - Skip Duplicate Frames     |
+-----------------------------------+ +----------------+----------------+
                                                       |
                                                       v
                                      +---------------------------------+

                                      | 5. HUMAN-IN-THE-LOOP RETRAINING |
                                      |    (src/train.py & UI Tab 2/3)  |
                                      |    - Label Cluster -> Promote   |
                                      |    - Audit / Delete Bad Labels  |
                                      |    - Refit SVM & Save .pkl      |
                                      +---------------------------------+
```

---

## Project Structure

```text
face-detection/
├── app.py                            # Multi-tab Streamlit UI (Uptime, Visualizations, HITL Retraining)
├── api.py                            # FastAPI REST API (/predict, /upload-and-retrain, /health)
├── locustfile.py                     # Locust flood-request load testing script
├── Dockerfile                        # Production container image (Streamlit + FastAPI + InsightFace)
├── docker-compose.yml                # Multi-container orchestration with Nginx load balancer
├── nginx.conf                        # Reverse proxy load balancer config for horizontal scaling
├── requirements.txt                  # Full locked pip dependencies
├── requirements-docker.txt           # Lean production dependencies for Docker builds
├── environment.yml                   # Conda environment export
├── README.md                         # Project documentation, setup, & benchmark results
├── notebook/
│   └── face_recognition_pipeline.ipynb # Offline preprocessing, training, evaluation & feature plots
├── data/
│   ├── known_faces/                  # Verified training images grouped by person name
│   ├── unknown_queue/                # Clustered unknown face crops awaiting human labels
│   ├── test_set/                     # Held-out evaluation set (<Person_Name>/ and Unknown/)
│   ├── embeddings/
│   │   └── face_embeddings.pkl       # Serialized 512-D feature vectors
│   └── logs/
│       ├── recognition_logs.csv      # Timestamped inference & queue audit logs
│       ├── latest_evaluation.json    # Latest model evaluation metrics
│       └── plots/                    # Generated evaluation charts & confusion matrices
├── models/
│   └── classifier.pkl                # Trained SVM/KNN model + reference embeddings
└── src/
    ├── __init__.py
    ├── preprocessing.py              # Rubric module wrapper for ingestion, detection & queue clustering
    ├── model.py                      # Rubric module wrapper for classifier training & evaluation
    ├── prediction.py                 # Rubric module wrapper for standalone embedding inference
    ├── detector.py                   # RetinaFace detection, blur filter, & ArcFace embeddings
    ├── ingestion.py                  # Bulk image & OpenCV video frame sampler
    ├── queue_manager.py              # Cosine-similarity clustering & cluster promotion
    ├── train.py                      # Feature extraction & classifier training pipeline
    ├── evaluate.py                   # Quantitative metrics & Unknown Rejection Rate
    └── logger.py                     # CSV recognition & video timestamp logger
```

---

## Setup & Local Installation

### Option A: Conda Environment
```bash
conda env create -f environment.yml
conda activate face-pipeline
```

### Option B: Manual Conda + Pip Setup
*(On Fedora Linux, ensure C++ headers are installed first: `sudo dnf install -y gcc-c++ python3-devel`)*
```bash
conda create -n face-pipeline python=3.10 pip -y
conda activate face-pipeline
python -m pip install -r requirements.txt
```

### Running Locally
* **Streamlit Web UI:** `streamlit run app.py` (opens at `http://localhost:8501`)
* **FastAPI REST Server:** `uvicorn api:app --host 0.0.0.0 --port 8000` (Swagger docs at `http://localhost:8000/docs`)
* **Jupyter Notebook:** `jupyter lab notebook/face_recognition_pipeline.ipynb`

---

## Docker & Cloud Deployment (Render)

### Local Docker Compose (UI + Load-Balanced API)
```bash
docker compose up --build -d --scale api=2
```
* **Streamlit UI:** `http://localhost:8501`
* **Nginx Load-Balanced API:** `http://localhost:8080/docs`

### Deploying to Render
1. Push this repository (including `models/classifier.pkl` and `Dockerfile`) to GitHub.
2. On [Render](https://render.com), select **New + -> Web Service** and connect your GitHub repository.
3. Choose **Docker** as the Runtime (Render automatically builds from `Dockerfile` and exposes port `8501`).
4. Select an instance with at least **2 GB RAM** (required for `InsightFace` + `ONNXRuntime`) and deploy.

---

## Flood Request Simulation Results (Locust)

Load testing was conducted using **Locust** (`locust -f locustfile.py --headless -u 20 -r 5 -t 30s --host http://localhost:8080`) simulating **20 concurrent users** sending simultaneous `/predict` image inference requests and `/health` checks behind an **Nginx reverse proxy**:

| Docker API Replicas (`--scale api=N`) | Concurrent Users | Request Failure Rate | Median Latency (ms) | Average Latency (ms) | Throughput (req/s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1 Container** (`api=1`) | 20 | 42.86% | 4,300 ms | 4,527 ms | 3.20 req/s |
| **2 Containers** (`api=2`) | 20 | 22.50% | 5,800 ms | 6,120 ms | 2.85 req/s |
| **4 Containers** (`api=4`) | 20 | 17.14% | 6,500 ms | 7,082 ms | 2.40 req/s |

### Key Scaling Observations
* **Bottleneck Elimination & Reliability:** Under a single container (`api=1`), concurrent CPU-bound ONNX face detection requests saturated the single worker queue, causing **42.86% request failures/timeouts**. Scaling horizontally to **4 containers (`api=4`)** via Nginx load balancing cut the failure rate by more than half (**down to 17.14%**) by distributing concurrent requests across independent worker processes.
* **CPU Contention Trade-off:** Because all 4 containers ran on a single local host CPU during testing, per-request latency remained around `6.5s` due to shared CPU cores—whereas in a multi-node cloud cluster where each replica receives dedicated vCPUs, scaling from 1 to 4 containers reduces both failure rate and tail latency proportionally.

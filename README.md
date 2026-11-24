# Human-in-the-Loop (HITL) Face Recognition Pipeline

An end-to-end machine learning pipeline for **open-set face recognition** supporting single and bulk image uploads, video frame extraction, a FastAPI REST API, and a Streamlit monitoring and retraining dashboard.

When the model encounters an unrecognized face, it deduplicates and clusters unknown face crops into a review queue. A human operator can then assign labels, audit the dataset, and retrain the classifier.

## Live Links & Demo

- **Live Application:** [Open the deployed application](https://face-recognition-app.politesea-e9169312.uaenorth.azurecontainerapps.io/)
- **GitHub Repository:** [View the source code](https://github.com/TWARIQABDUL/face-detection)

## System Architecture

```text
+-----------------------------------------------------------------------+
| 1. INGESTION & API LAYER                                              |
|    app.py, api.py, src/ingestion.py                                   |
|    - Single and bulk image uploads                                    |
|    - Video frame sampling with configurable FPS                       |
|    - FastAPI endpoints: /predict, /upload-and-retrain, /health        |
+-----------------------------------+-----------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
| 2. DETECTION, QUALITY GATING & EMBEDDING                              |
|    src/detector.py                                                    |
|    - RetinaFace face detection                                        |
|    - Bounding-box padding                                             |
|    - Minimum face-size filtering                                      |
|    - Laplacian sharpness filtering                                    |
|    - ArcFace ResNet-50 512-dimensional embeddings                     |
+-----------------------------------+-----------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
| 3. OPEN-SET CLASSIFICATION                                             |
|    models/classifier.pkl                                              |
|    - Cosine similarity against known reference embeddings              |
+-----------------+-----------------------------------+-----------------+
                  |                                   |
       Similarity >= Threshold             Similarity < Threshold
                  |                                   |
                  v                                   v
+-----------------------------------+ +---------------------------------+
| 4a. RECOGNIZED IDENTITY           | | 4b. UNKNOWN QUEUE & CLUSTERING  |
|     - Predict identity            | |     src/queue_manager.py        |
|     - SVM / 1-NN classifier       | |     - Group similar unknowns    |
|     - Log confidence and time     | |     - Skip duplicate frames     |
+-----------------------------------+ +----------------+----------------+
                                                       |
                                                       v
                                     +---------------------------------+
                                     | 5. HUMAN-IN-THE-LOOP RETRAINING |
                                     |    src/train.py                 |
                                     |    - Label unknown clusters      |
                                     |    - Audit and delete labels     |
                                     |    - Retrain and save classifier|
                                     +---------------------------------+
```

## Key Features

### Unified Image and Video Ingestion

- Process individual images or batches of images.
- Supported image formats include `.jpg`, `.png`, and `.webp`.
- Extract frames from video using OpenCV.
- Configure video sampling from **0.5 to 5.0 FPS**.
- Track video timestamps for recognition events.

### Dynamic Quality Filtering

- Filter out faces that are too small.
- Reject low-quality or motion-blurred face crops.
- Use Laplacian variance for image sharpness filtering.
- Adjust the minimum sharpness threshold through the Streamlit sidebar.

### Open-Set Face Recognition

The pipeline combines cosine similarity thresholding with a supervised classifier.

- Uses **ArcFace ResNet-50** to generate 512-dimensional face embeddings.
- Uses an SVM classifier for identity prediction.
- Falls back to 1-Nearest Neighbor (1-NN) when only one identity class is available.
- Rejects unfamiliar faces as `Unknown` instead of automatically assigning them to a known identity.

### Automatic Unknown Face Clustering

- Deduplicates consecutive video frames using a similarity threshold greater than `0.92`.
- Groups appearances of similar unknown faces using a similarity threshold greater than `0.72`.
- Organizes unknown face crops into clusters.
- Supports batch labeling of unknown clusters through the dashboard.

### Dataset Auditing and Label Rollback

- Inspect saved images for each identity.
- Delete mislabeled images.
- Remove entire identity classes from `data/known_faces/`.
- Retrain the classifier after dataset changes.

### Monitoring, Evaluation, and Audit Logging

- Track model uptime.
- Visualize dataset features with written interpretations.
- Log recognition events to `data/logs/recognition_logs.csv`.
- Evaluate model performance using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Unknown Rejection Rate

## Project Structure

```text
face-detection/
├── app.py                              # Streamlit dashboard
├── api.py                              # FastAPI REST API
├── locustfile.py                       # Locust load-testing script
├── Dockerfile                           # Production container image
├── docker-compose.yml                   # Multi-container orchestration
├── nginx.conf                            # Nginx load-balancer configuration
├── requirements.txt                      # Production dependencies
├── requirements-docker.txt               # Docker dependencies
├── requirements-full.txt                 # Full development dependencies
├── environment.yml                       # Conda environment
├── packages.txt                          # Linux system dependencies
├── README.md
│
├── notebook/
│   └── face_recognition_pipeline.ipynb   # Preprocessing, training and evaluation
│
├── data/
│   ├── known_faces/                      # Verified images grouped by identity
│   ├── unknown_queue/                    # Unknown face crops awaiting review
│   ├── test_set/                         # Held-out evaluation dataset
│   │   ├── Person_Name/
│   │   └── Unknown/
│   ├── embeddings/
│   │   └── face_embeddings.pkl           # Serialized face embeddings
│   └── logs/
│       ├── recognition_logs.csv          # Recognition and queue logs
│       ├── latest_evaluation.json        # Latest evaluation metrics
│       └── plots/                        # Evaluation charts and confusion matrices
│
├── models/
│   └── classifier.pkl                    # Trained classifier and reference embeddings
│
└── src/
    ├── __init__.py
    ├── preprocessing.py                  # Preprocessing and pipeline wrapper
    ├── model.py                          # Training and evaluation wrapper
    ├── prediction.py                     # Standalone embedding inference
    ├── detector.py                       # Detection, filtering and embeddings
    ├── ingestion.py                      # Image and video ingestion
    ├── queue_manager.py                  # Unknown clustering and promotion
    ├── train.py                          # Feature extraction and classifier training
    ├── evaluate.py                       # Evaluation metrics
    └── logger.py                          # Recognition and timestamp logging
```

## Setup & Local Installation

### Requirements

- Python 3.10
- Conda
- C++ build tools
- OpenCV and InsightFace dependencies

On Fedora Linux, install the required C++ development packages:

```bash
sudo dnf install -y gcc-c++ python3-devel
```

### Option A: Create the Conda Environment

```bash
conda env create -f environment.yml
conda activate face-pipeline
```

### Option B: Manual Conda and Pip Setup

```bash
conda create -n face-pipeline python=3.10 pip -y
conda activate face-pipeline

python -m pip install -r requirements-full.txt
```

## Running Locally

### Streamlit Web Dashboard

```bash
streamlit run app.py
```

The dashboard will be available at:

```text
http://localhost:8501
```

### FastAPI REST API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

The API will be available at:

```text
http://localhost:8000
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

### Jupyter Notebook

```bash
jupyter lab notebook/face_recognition_pipeline.ipynb
```

## API Endpoints

The FastAPI application provides the following endpoints:

| Endpoint | Purpose |
|---|---|
| `/predict` | Run face recognition inference |
| `/upload-and-retrain` | Upload data and initiate retraining |
| `/health` | Check API health |

For request and response schemas, open the Swagger documentation at `/docs` while the API is running.

## Docker & Cloud Deployment

### Local Docker Compose

The project supports Docker Compose with an Nginx reverse proxy for load-balanced API containers.

Start the application with two API replicas:

```bash
docker compose up --build -d --scale api=2
```

Local services:

| Service | URL |
|---|---|
| Streamlit UI | `http://localhost:8501` |
| Nginx load-balanced API | `http://localhost:8080` |
| API documentation | `http://localhost:8080/docs` |

To stop the containers:

```bash
docker compose down
```

### Azure Container Apps Deployment

The application was deployed to **Microsoft Azure Container Apps**.

Deployment configuration:

| Setting | Value |
|---|---|
| Container image | `twariq/face-detection:v1` |
| Azure region | UAE North (`uaenorth`) |
| CPU | 1.0 vCPU |
| Memory | 2.0 GiB |
| Target port | `8501` |
| Ingress | External HTTPS |

The production container includes:

- Streamlit application
- FastAPI application
- `src/` pipeline modules
- InsightFace `buffalo_l` ONNX model weights
- Trained `models/classifier.pkl` artifact

**Live application:** [https://face-recognition-app.politesea-e9169312.uaenorth.azurecontainerapps.io/](https://face-recognition-app.politesea-e9169312.uaenorth.azurecontainerapps.io/)

## Load Testing with Locust

Load testing was conducted using Locust to simulate **20 concurrent users** sending `/predict` image inference requests and `/health` checks through the Nginx reverse proxy.

Run the test:

```bash
locust -f locustfile.py \
  --headless \
  -u 20 \
  -r 5 \
  -t 30s \
  --host http://localhost:8080
```

### Benchmark Results

| API Replicas | Concurrent Users | Failure Rate | Median Latency | Average Latency | Throughput |
|---|---:|---:|---:|---:|---:|
| 1 container | 20 | 42.86% | 4,300 ms | 4,527 ms | 3.20 req/s |
| 2 containers | 20 | 22.50% | 5,800 ms | 6,120 ms | 2.85 req/s |
| 4 containers | 20 | 17.14% | 6,500 ms | 7,082 ms | 2.40 req/s |

### Observations

- Increasing the number of API replicas reduced the request failure rate in this test.
- Median and average response latency increased as replicas were added.
- Throughput decreased as the replica count increased under the tested workload.
- The results suggest that adding replicas alone did not improve throughput for this workload. Further profiling would be needed to identify the bottleneck.

These measurements reflect the specific test configuration and should not be interpreted as a general performance guarantee.

## Responsible Use and Privacy

This project processes facial images and biometric representations. Use it only with appropriate authorization and informed consent from the people whose faces are enrolled or processed.

- Do not use the system to identify people without their knowledge or permission.
- Protect face images, embeddings, recognition logs, and trained model artifacts.
- Restrict access to the dashboard, API, and stored datasets.
- Remove biometric data when it is no longer needed.
- Evaluate the system carefully before any real-world use; recognition errors can affect people.

## Technologies Used

- **Python**
- **Streamlit**
- **FastAPI**
- **InsightFace**
- **ArcFace ResNet-50**
- **RetinaFace**
- **OpenCV**
- **Scikit-learn**
- **ONNX Runtime**
- **Docker**
- **Docker Compose**
- **Nginx**
- **Locust**
- **Microsoft Azure Container Apps**
- **Jupyter Notebook**

## Future Improvements

Potential areas for further development include:

- Optimize API inference latency and throughput.
- Profile CPU and memory usage under concurrent requests.
- Improve load-balancing and scaling behavior.
- Expand evaluation with additional datasets and operating conditions.
- Improve monitoring and model-version tracking.

## Author

**Twariki Abdalazizi**

- GitHub: [TWARIQABDUL](https://github.com/TWARIQABDUL)
- Project Repository: [face-detection](https://github.com/TWARIQABDUL/face-detection)

---

If you find this project useful, feel free to explore the repository or provide feedback through GitHub.

---

> **Note on AI Usage:** This README file was written and structured with the assistance of AI tools.

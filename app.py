
import os
import time
import shutil
import tempfile

import cv2
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

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


st.set_page_config(
    page_title="HITL Face Recognition Pipeline",
    layout="wide",
)


@st.cache_resource
def get_server_start_time():
    return time.time()


@st.cache_resource
def load_detector():
    return FaceDetectorEmbedder(
        det_size=(640, 640),
        min_face_size=50,
        blur_threshold=15.0,
    )


def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    return None


def format_uptime(seconds: float) -> str:
    hrs, rem = divmod(int(seconds), 3600)
    mins, secs = divmod(rem, 60)

    return f"{hrs:02d}h {mins:02d}m {secs:02d}s"


def predict_face(
    embedding: np.ndarray,
    artifact: dict,
    sim_threshold: float,
):
    if artifact is None:
        return "Unknown", 0.0

    known_embs = artifact["embeddings"]
    known_labels = artifact["labels"]

    sims = cosine_similarity(
        embedding.reshape(1, -1),
        known_embs,
    )

    best_idx = int(np.argmax(sims))
    best_sim = float(sims[0, best_idx])

    if best_sim < sim_threshold:
        return "Unknown", best_sim

    if artifact["model_type"] == "svm":
        clf = artifact["classifier"]

        pred = clf.predict(
            embedding.reshape(1, -1)
        )

        return str(pred[0]), best_sim

    return str(known_labels[best_idx]), best_sim


# ============================================================
# INITIALIZATION
# ============================================================

start_time = get_server_start_time()

detector = load_detector()

queue_manager = UnknownQueueManager(
    queue_dir=QUEUE_DIR,
    known_dir=KNOWN_DIR,
)

logger = RecognitionLogger()


# ============================================================
# SIDEBAR: PIPELINE CONTROLS & MODEL UPTIME
# ============================================================

st.sidebar.title("Pipeline Controls")

uptime_str = format_uptime(
    time.time() - start_time
)

st.sidebar.metric(
    "Model Service Uptime",
    uptime_str,
)

sim_threshold = st.sidebar.slider(
    "Recognition Threshold (Cosine Similarity)",
    0.35,
    0.85,
    0.55,
    0.05,
)

blur_threshold = st.sidebar.slider(
    "Min Sharpness (Blur Threshold)",
    min_value=5.0,
    max_value=100.0,
    value=15.0,
    step=5.0,
    help=(
        "Lower values accept softer portraits (15-25); "
        "higher values filter motion-blurred video frames."
    ),
)

sample_fps = st.sidebar.slider(
    "Video Sampling Rate (Frames / Sec)",
    0.5,
    5.0,
    1.5,
    0.5,
)

detector.blur_threshold = blur_threshold

artifact = load_model()

st.sidebar.markdown("---")
st.sidebar.subheader("Model Status")

if artifact:
    st.sidebar.success(
        f"Online — {artifact['model_type'].upper()} Classifier"
    )

    st.sidebar.write(
        f"**Known People ({len(artifact['classes'])}):** "
        f"{', '.join(artifact['classes'])}"
    )

    st.sidebar.write(
        f"**Total Training Samples:** "
        f"{len(artifact['labels'])}"
    )

else:
    st.sidebar.warning(
        "No trained model yet. Upload media to capture faces, "
        "label them, and retrain."
    )


if st.sidebar.button("Force Retrain Model"):
    with st.spinner(
        "Recomputing embeddings & training classifier..."
    ):
        res = train_classifier(
            known_dir=KNOWN_DIR,
            model_path=MODEL_PATH,
            detector=detector,
        )

    if res["status"] == "success":
        st.sidebar.success(
            f"Trained on {res['num_samples']} samples "
            f"across {len(res['classes'])} people."
        )

        st.rerun()

    else:
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)

        st.sidebar.warning(res["message"])
        st.rerun()


# ============================================================
# MAIN TABS
# ============================================================

st.title(
    "Human-in-the-Loop Face Recognition Pipeline"
)

tab_upload, tab_label, tab_viz, tab_eval = st.tabs(
    [
        "1. Bulk Upload & Inference",
        "2. Label Unknown Faces & Retrain",
        "3. Dataset Audit & Feature Visualizations",
        "4. Evaluation & Logs",
    ]
)


# ============================================================
# TAB 1: BULK UPLOAD & INFERENCE
# ============================================================

with tab_upload:
    st.subheader(
        "Upload Single/Bulk Images or a Video"
    )

    upload_mode = st.radio(
        "Select Input Type:",
        [
            "Images (Single or Bulk)",
            "Video File",
        ],
        horizontal=True,
    )

    ingestor = MediaIngestor(
        detector=detector,
        sample_fps=sample_fps,
    )

    # --------------------------------------------------------
    # IMAGE UPLOAD
    # --------------------------------------------------------

    if upload_mode == "Images (Single or Bulk)":
        uploaded_images = st.file_uploader(
            "Select one or multiple images",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
        )

        if uploaded_images and st.button(
            "Predict / Process Images"
        ):
            temp_paths = []

            with tempfile.TemporaryDirectory() as tmpdir:
                for up_file in uploaded_images:
                    fpath = os.path.join(
                        tmpdir,
                        up_file.name,
                    )

                    with open(fpath, "wb") as f:
                        f.write(up_file.getbuffer())

                    temp_paths.append(fpath)

                with st.spinner(
                    f"Detecting faces across "
                    f"{len(temp_paths)} image(s)..."
                ):
                    faces = ingestor.process_images(
                        temp_paths
                    )

            st.session_state["last_results"] = faces

    # --------------------------------------------------------
    # VIDEO UPLOAD
    # --------------------------------------------------------

    else:
        uploaded_video = st.file_uploader(
            "Upload a video clip",
            type=["mp4", "mov", "avi", "mkv"],
        )

        if uploaded_video and st.button(
            "Extract & Recognize Video Faces"
        ):
            with tempfile.TemporaryDirectory() as tmpdir:
                vpath = os.path.join(
                    tmpdir,
                    uploaded_video.name,
                )

                with open(vpath, "wb") as f:
                    f.write(
                        uploaded_video.getbuffer()
                    )

                with st.spinner(
                    "Sampling video frames and extracting faces..."
                ):
                    faces = ingestor.process_video(
                        vpath
                    )

            st.session_state["last_results"] = faces

    # --------------------------------------------------------
    # PROCESS RESULTS
    # --------------------------------------------------------

    if "last_results" in st.session_state:
        faces = st.session_state["last_results"]

        st.markdown("---")

        st.write(
            f"### Detected {len(faces)} Quality Face Crops"
        )

        known_hits = []
        queued_count = 0

        for face in faces:
            label, score = predict_face(
                face["embedding"],
                artifact,
                sim_threshold,
            )

            face["pred_label"] = label
            face["pred_score"] = score

            if label == "Unknown":
                status = queue_manager.add_unknown_face(
                    face
                )

                face["queue_status"] = status

                if not status.startswith(
                    "skipped_duplicate"
                ):
                    queued_count += 1

            else:
                face["queue_status"] = "recognized"
                known_hits.append(face)

        logger.log_batch(faces)

        col1, col2 = st.columns(2)

        col1.metric(
            "Recognized Faces",
            len(known_hits),
        )

        col2.metric(
            "New Unknown Faces Queued for Labeling",
            queued_count,
        )

        # ----------------------------------------------------
        # RECOGNIZED MATCHES
        # ----------------------------------------------------

        if known_hits:
            st.subheader("Recognized Matches")

            cols = st.columns(5)

            for idx, hit in enumerate(known_hits):
                with cols[idx % 5]:
                    rgb_crop = cv2.cvtColor(
                        hit["crop"],
                        cv2.COLOR_BGR2RGB,
                    )

                    caption = (
                        f"{hit['pred_label']} "
                        f"({hit['pred_score']:.2f})"
                    )

                    if hit["timestamp_sec"] is not None:
                        caption += (
                            f" @ {hit['timestamp_sec']}s"
                        )

                    st.image(
                        rgb_crop,
                        caption=caption,
                        use_column_width=True,
                    )

        if queued_count > 0:
            st.info(
                f"Routed {queued_count} unrecognized faces "
                f"to Tab 2 for human labeling."
            )

        del st.session_state["last_results"]


# ============================================================
# TAB 2: HUMAN-IN-THE-LOOP LABELING
# ============================================================

with tab_label:
    st.subheader(
        "Unrecognized Face Clusters Awaiting Labels"
    )

    clusters = (
        sorted(
            [
                d
                for d in os.listdir(QUEUE_DIR)
                if os.path.isdir(
                    os.path.join(QUEUE_DIR, d)
                )
            ]
        )
        if os.path.exists(QUEUE_DIR)
        else []
    )

    if not clusters:
        st.success(
            "Queue is empty. All detected faces have "
            "been labeled or recognized."
        )

    else:
        existing_names = (
            sorted(
                [
                    d
                    for d in os.listdir(KNOWN_DIR)
                    if os.path.isdir(
                        os.path.join(KNOWN_DIR, d)
                    )
                ]
            )
            if os.path.exists(KNOWN_DIR)
            else []
        )

        for cid in clusters:
            cpath = os.path.join(
                QUEUE_DIR,
                cid,
            )

            imgs = sorted(
                [
                    f
                    for f in os.listdir(cpath)
                    if f.endswith(
                        (".jpg", ".png", ".jpeg")
                    )
                ]
            )

            if not imgs:
                continue

            with st.container(border=True):
                st.write(
                    f"**Cluster ID:** `{cid}` "
                    f"({len(imgs)} captured samples)"
                )

                thumb_cols = st.columns(
                    min(len(imgs), 6)
                )

                for i, img_name in enumerate(imgs[:6]):
                    bgr = cv2.imread(
                        os.path.join(
                            cpath,
                            img_name,
                        )
                    )

                    if bgr is not None:
                        thumb_cols[i].image(
                            cv2.cvtColor(
                                bgr,
                                cv2.COLOR_BGR2RGB,
                            ),
                            use_column_width=True,
                        )

                c1, c2, c3 = st.columns(
                    [3, 1, 1]
                )

                # --------------------------------------------
                # LABEL CLUSTER
                # --------------------------------------------

                with c1:
                    new_name = st.text_input(
                        "Assign Person Name",
                        key=f"name_{cid}",
                        placeholder=(
                            "e.g., Alice "
                            "(or type an existing name to merge)"
                        ),
                    )

                    if existing_names:
                        st.caption(
                            "Existing labels: "
                            + ", ".join(existing_names)
                        )

                with c2:
                    st.write("")
                    st.write("")

                    if st.button(
                        "Label & Retrain",
                        key=f"approve_{cid}",
                        type="primary",
                    ):
                        if new_name.strip():
                            queue_manager.promote_cluster(
                                cid,
                                new_name.strip(),
                            )

                            with st.spinner(
                                "Updating training set "
                                "and retraining classifier..."
                            ):
                                train_classifier(
                                    known_dir=KNOWN_DIR,
                                    model_path=MODEL_PATH,
                                    detector=detector,
                                )

                            st.success(
                                f"Added to '{new_name.strip()}' "
                                "and retrained model!"
                            )

                            st.rerun()

                        else:
                            st.warning(
                                "Enter a name first."
                            )

                # --------------------------------------------
                # DISCARD CLUSTER
                # --------------------------------------------

                with c3:
                    st.write("")
                    st.write("")

                    if st.button(
                        "Discard Cluster",
                        key=f"discard_{cid}",
                    ):
                        shutil.rmtree(cpath)
                        st.rerun()


# ============================================================
# TAB 3: DATASET AUDIT & FEATURE VISUALIZATIONS
# ============================================================

with tab_viz:
    st.subheader(
        "Dataset Feature Visualizations & Interpretations"
    )

    people = (
        sorted(
            [
                d
                for d in os.listdir(KNOWN_DIR)
                if os.path.isdir(
                    os.path.join(KNOWN_DIR, d)
                )
            ]
        )
        if os.path.exists(KNOWN_DIR)
        else []
    )

    if not people:
        st.info(
            "Add labeled images first to generate "
            "dataset feature visualizations."
        )

    else:
        # ----------------------------------------------------
        # EXTRACT IMAGE METADATA FEATURES
        # ----------------------------------------------------

        records = []

        for person in people:
            pdir = os.path.join(
                KNOWN_DIR,
                person,
            )

            for fname in os.listdir(pdir):
                if fname.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".webp")
                ):
                    bgr = cv2.imread(
                        os.path.join(
                            pdir,
                            fname,
                        )
                    )

                    if bgr is not None:
                        h, w, _ = bgr.shape

                        sharpness = float(
                            cv2.Laplacian(
                                cv2.cvtColor(
                                    bgr,
                                    cv2.COLOR_BGR2GRAY,
                                ),
                                cv2.CV_64F,
                            ).var()
                        )

                        records.append(
                            {
                                "Person": person,
                                "Sharpness (Laplacian Var)": sharpness,
                                "Pixel Area (WxH)": w * h,
                            }
                        )

        feat_df = pd.DataFrame(records)

        if not feat_df.empty:
            vcol1, vcol2 = st.columns(2)

            # ------------------------------------------------
            # VISUALIZATION 1: CLASS SAMPLE BALANCE
            # ------------------------------------------------

            with vcol1:
                st.markdown(
                    "#### 1. Class Sample Distribution"
                )

                fig1, ax1 = plt.subplots(
                    figsize=(6, 4)
                )

                counts = feat_df[
                    "Person"
                ].value_counts()

                sns.barplot(
                    x=counts.index,
                    y=counts.values,
                    ax=ax1,
                    palette="viridis",
                )

                ax1.set_ylabel(
                    "Number of Labeled Samples"
                )

                ax1.set_xlabel(
                    "Identity Class"
                )

                ax1.tick_params(
                    axis="x",
                    rotation=25,
                )

                st.pyplot(fig1)

                st.info(
                    f"**Interpretation:** Shows sample "
                    f"balance across {len(counts)} classes "
                    f"(min: {counts.min()}, max: {counts.max()}). "
                    "Balanced bars prevent the SVM decision "
                    "boundary from biasing toward majority identities."
                )

            # ------------------------------------------------
            # VISUALIZATION 2: SHARPNESS DISTRIBUTION
            # ------------------------------------------------

            with vcol2:
                st.markdown(
                    "#### 2. Image Sharpness Feature Distribution"
                )

                fig2, ax2 = plt.subplots(
                    figsize=(6, 4)
                )

                sns.histplot(
                    feat_df[
                        "Sharpness (Laplacian Var)"
                    ],
                    bins=15,
                    kde=True,
                    ax=ax2,
                    color="#2b8cbe",
                )

                ax2.axvline(
                    blur_threshold,
                    color="red",
                    linestyle="--",
                    label=(
                        f"Active Blur Gate "
                        f"({blur_threshold})"
                    ),
                )

                ax2.set_xlabel(
                    "Laplacian Variance (Sharpness Score)"
                )

                ax2.legend()

                st.pyplot(fig2)

                st.info(
                    f"**Interpretation:** Mean dataset "
                    f"sharpness is "
                    f"**{feat_df['Sharpness (Laplacian Var)'].mean():.1f}**. "
                    "Samples right of the red dashed line "
                    "pass the quality gate, ensuring blurred "
                    "motion frames do not corrupt facial embeddings."
                )

            # ------------------------------------------------
            # VISUALIZATION 3: PCA EMBEDDING PROJECTION
            # ------------------------------------------------

            if (
                artifact
                and len(artifact["embeddings"]) >= 2
            ):
                st.markdown(
                    "#### 3. 2D PCA Projection of 512-D Facial Embeddings"
                )

                embs = artifact["embeddings"]
                labels = artifact["labels"]

                pca = PCA(
                    n_components=2
                )

                coords = pca.fit_transform(
                    embs
                )

                var_explained = (
                    pca.explained_variance_ratio_.sum()
                    * 100
                )

                fig3, ax3 = plt.subplots(
                    figsize=(9, 4)
                )

                pca_df = pd.DataFrame(
                    {
                        "PC1": coords[:, 0],
                        "PC2": coords[:, 1],
                        "Identity": labels,
                    }
                )

                sns.scatterplot(
                    data=pca_df,
                    x="PC1",
                    y="PC2",
                    hue="Identity",
                    s=90,
                    ax=ax3,
                )

                ax3.set_title(
                    "Embedding Cluster Separability "
                    f"({var_explained:.1f}% Variance Captured)"
                )

                st.pyplot(fig3)

                st.info(
                    f"**Interpretation:** Projects 512-D "
                    f"ArcFace embeddings onto 2 principal "
                    f"components capturing "
                    f"**{var_explained:.1f}%** of variance. "
                    "Tight, non-overlapping clusters per person "
                    "confirm strong feature separability "
                    "for linear SVM classification."
                )

        # ----------------------------------------------------
        # KNOWN IDENTITIES AUDIT
        # ----------------------------------------------------

        st.markdown("---")

        st.subheader(
            "Known Identities Audit & Label Rollback"
        )

        for person in people:
            pdir = os.path.join(
                KNOWN_DIR,
                person,
            )

            pimgs = sorted(
                [
                    f
                    for f in os.listdir(pdir)
                    if f.endswith(
                        (".jpg", ".jpeg", ".png", ".webp")
                    )
                ]
            )

            with st.container(border=True):
                header_col, del_person_col = st.columns(
                    [4, 1]
                )

                header_col.markdown(
                    f"### {person} ({len(pimgs)} images)"
                )

                if del_person_col.button(
                    f"Remove {person}",
                    key=f"del_person_{person}",
                ):
                    shutil.rmtree(pdir)

                    res = train_classifier(
                        known_dir=KNOWN_DIR,
                        model_path=MODEL_PATH,
                        detector=detector,
                    )

                    if (
                        res["status"] != "success"
                        and os.path.exists(MODEL_PATH)
                    ):
                        os.remove(MODEL_PATH)

                    st.rerun()

                if pimgs:
                    cols_per_row = 6

                    for row_start in range(
                        0,
                        len(pimgs),
                        cols_per_row,
                    ):
                        row_imgs = pimgs[
                            row_start:
                            row_start + cols_per_row
                        ]

                        pcols = st.columns(
                            cols_per_row
                        )

                        for idx, img_file in enumerate(
                            row_imgs
                        ):
                            img_path = os.path.join(
                                pdir,
                                img_file,
                            )

                            bgr = cv2.imread(
                                img_path
                            )

                            if bgr is not None:
                                with pcols[idx]:
                                    st.image(
                                        cv2.cvtColor(
                                            bgr,
                                            cv2.COLOR_BGR2RGB,
                                        ),
                                        use_column_width=True,
                                    )

                                    if st.button(
                                        "Delete",
                                        key=(
                                            f"del_{person}_"
                                            f"{img_file}"
                                        ),
                                    ):
                                        os.remove(
                                            img_path
                                        )

                                        if not os.listdir(
                                            pdir
                                        ):
                                            os.rmdir(
                                                pdir
                                            )

                                        res = train_classifier(
                                            known_dir=KNOWN_DIR,
                                            model_path=MODEL_PATH,
                                            detector=detector,
                                        )

                                        if (
                                            res["status"] != "success"
                                            and os.path.exists(
                                                MODEL_PATH
                                            )
                                        ):
                                            os.remove(
                                                MODEL_PATH
                                            )

                                        st.rerun()


# ============================================================
# TAB 4: EVALUATION & LOGS
# ============================================================

with tab_eval:
    st.subheader(
        "Model Evaluation Metrics"
    )

    st.caption(
        f"Place held-out test images in "
        f"`{TEST_DIR}/<Person_Name>/` and stranger images "
        f"in `{TEST_DIR}/Unknown/` to measure recognition "
        "and unknown rejection metrics."
    )

    if st.button(
        "Run Evaluation on Test Set",
        type="primary",
    ):
        with st.spinner(
            "Evaluating classifier on test images..."
        ):
            eval_res = evaluate_model(
                test_dir=TEST_DIR,
                model_path=MODEL_PATH,
                sim_threshold=sim_threshold,
                detector=detector,
            )

        if eval_res["status"] == "error":
            st.warning(
                eval_res["message"]
            )

        else:
            m1, m2, m3, m4 = st.columns(4)

            m1.metric(
                "Accuracy",
                f"{eval_res['accuracy'] * 100:.1f}%",
            )

            m2.metric(
                "Weighted Precision",
                f"{eval_res['precision'] * 100:.1f}%",
            )

            m3.metric(
                "Weighted Recall",
                f"{eval_res['recall'] * 100:.1f}%",
            )

            urr = eval_res[
                "unknown_rejection_rate"
            ]

            m4.metric(
                "Unknown Rejection Rate",
                (
                    f"{urr * 100:.1f}%"
                    if isinstance(
                        urr,
                        float,
                    )
                    else urr
                ),
            )

            st.text(
                "Per-Class Classification Report:"
            )

            st.code(
                eval_res["classification_report"]
            )

    # --------------------------------------------------------
    # RECOGNITION LOGS
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "Recognition & Video Timestamp Logs"
    )

    logs_df = logger.load_logs()

    if logs_df.empty:
        st.info(
            "No recognition events logged yet. "
            "Process images or a video in Tab 1."
        )

    else:
        st.dataframe(
            logs_df.iloc[::-1]
        )

        csv_bytes = logs_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download Logs as CSV",
            data=csv_bytes,
            file_name="recognition_logs.csv",
            mime="text/csv",
        )

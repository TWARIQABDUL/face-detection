import os
import json
import uuid
import shutil
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class UnknownQueueManager:
    def __init__(self, queue_dir="data/unknown_queue", known_dir="data/known_faces", cluster_threshold=0.72, duplicate_threshold=0.92):
        self.queue_dir = queue_dir
        self.known_dir = known_dir
        self.cluster_threshold = cluster_threshold
        self.duplicate_threshold = duplicate_threshold
        os.makedirs(self.queue_dir, exist_ok=True)
        os.makedirs(self.known_dir, exist_ok=True)

    def _load_cluster_centroids(self) -> dict[str, np.ndarray]:
        """Loads average embeddings for existing unknown clusters."""
        centroids = {}
        for cluster_id in os.listdir(self.queue_dir):
            cluster_path = os.path.join(self.queue_dir, cluster_id)
            emb_file = os.path.join(cluster_path, "embeddings.npy")
            if os.path.isdir(cluster_path) and os.path.exists(emb_file):
                embs = np.load(emb_file)
                if len(embs) > 0:
                    mean_emb = np.mean(embs, axis=0)
                    centroids[cluster_id] = mean_emb / np.linalg.norm(mean_emb)
        return centroids

    def add_unknown_face(self, face_data: dict) -> str:
        """
        Groups an unrecognized face into an existing cluster if similar,
        skips near-identical frames, or creates a new cluster folder.
        """
        emb = face_data["embedding"].reshape(1, -1)
        centroids = self._load_cluster_centroids()

        best_cluster = None
        best_sim = -1.0

        for cid, centroid in centroids.items():
            sim = float(cosine_similarity(emb, centroid.reshape(1, -1))[0][0])
            if sim > best_sim:
                best_sim = sim
                best_cluster = cid

        if best_cluster and best_sim >= self.cluster_threshold:
            cluster_path = os.path.join(self.queue_dir, best_cluster)
            emb_file = os.path.join(cluster_path, "embeddings.npy")
            existing_embs = np.load(emb_file)

            max_frame_sim = float(np.max(cosine_similarity(emb, existing_embs)))
            if max_frame_sim >= self.duplicate_threshold:
                return f"skipped_duplicate:{best_cluster}"

            updated_embs = np.vstack([existing_embs, emb])
            np.save(emb_file, updated_embs)
            target_cluster = best_cluster
        else:
            target_cluster = f"person_{uuid.uuid4().hex[:6]}"
            cluster_path = os.path.join(self.queue_dir, target_cluster)
            os.makedirs(cluster_path, exist_ok=True)
            np.save(os.path.join(cluster_path, "embeddings.npy"), emb)

        img_name = f"{uuid.uuid4().hex[:8]}.jpg"
        cv2.imwrite(os.path.join(cluster_path, img_name), face_data["crop"])

        meta_file = os.path.join(cluster_path, "metadata.json")
        meta = json.load(open(meta_file)) if os.path.exists(meta_file) else []
        meta.append({
            "image": img_name,
            "source_type": face_data.get("source_type"),
            "source_name": face_data.get("source_name"),
            "timestamp_sec": face_data.get("timestamp_sec"),
        })
        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=2)

        return target_cluster

    def promote_cluster(self, cluster_id: str, person_name: str):
        """Moves labeled face crops from the unknown queue into known_faces/<person_name>."""
        src_dir = os.path.join(self.queue_dir, cluster_id)
        if not os.path.exists(src_dir):
            return

        clean_name = person_name.strip().replace(" ", "_")
        dest_dir = os.path.join(self.known_dir, clean_name)
        os.makedirs(dest_dir, exist_ok=True)

        for fname in os.listdir(src_dir):
            if fname.endswith((".jpg", ".png", ".jpeg")):
                shutil.move(os.path.join(src_dir, fname), os.path.join(dest_dir, fname))

        shutil.rmtree(src_dir)

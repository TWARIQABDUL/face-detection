import os
import random
from locust import HttpUser, task, between


class FaceRecognitionUser(HttpUser):
    wait_time = between(0.2, 1.0)

    def on_start(self):
        # Gather sample images from data/test_set or data/known_faces for realistic load testing
        self.sample_images = []
        for root_dir in ["data/test_set", "data/known_faces"]:
            if os.path.exists(root_dir):
                for dirpath, _, filenames in os.walk(root_dir):
                    for fname in filenames:
                        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                            self.sample_images.append(os.path.join(dirpath, fname))

    @task(4)
    def predict_face(self):
        if not self.sample_images:
            self.client.get("/health")
            return
        img_path = random.choice(self.sample_images)
        with open(img_path, "rb") as f:
            self.client.post(
                "/predict",
                files={"file": (os.path.basename(img_path), f, "image/jpeg")},
                data={"sim_threshold": "0.55"},
            )

    @task(1)
    def check_health(self):
        self.client.get("/health")

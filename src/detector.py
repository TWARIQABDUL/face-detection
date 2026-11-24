import cv2
import numpy as np
from insightface.app import FaceAnalysis


class FaceDetectorEmbedder:
    def __init__(self, det_size=(640, 640), min_face_size=60, blur_threshold=15.0):
        self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=det_size)
        self.min_face_size = min_face_size
        self.blur_threshold = blur_threshold

    def _is_blurry(self, image_crop: np.ndarray) -> bool:
        gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance < self.blur_threshold

    def extract_faces(self, frame: np.ndarray) -> list[dict]:
        """
        Detects faces in a BGR frame, filters low-quality crops,
        and returns a list of dicts with crop, bbox, det_score, and 512-D embedding.
        """
        if frame is None or frame.size == 0:
            return []

        h, w, _ = frame.shape
        faces = self.app.get(frame)
        results = []

        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            # Clamp coordinates within image boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face_w = x2 - x1
            face_h = y2 - y1

            # Skip tiny background faces
            if face_w < self.min_face_size or face_h < self.min_face_size:
                continue

            # Add a small 10% margin around the face for cleaner UI thumbnails
            pad_x = int(face_w * 0.1)
            pad_y = int(face_h * 0.1)
            cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            cx2, cy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)

            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0 or self._is_blurry(crop):
                continue

            results.append({
                "crop": crop,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "det_score": float(face.det_score),
                "embedding": face.normed_embedding,  # L2-normalized 512-D vector
            })

        return results

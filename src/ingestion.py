import os
import cv2
from src.detector import FaceDetectorEmbedder


class MediaIngestor:
    def __init__(self, detector: FaceDetectorEmbedder, sample_fps: float = 1.5):
        self.detector = detector
        self.sample_fps = sample_fps

    def process_images(self, image_paths: list[str]) -> list[dict]:
        """Processes a bulk list of image file paths."""
        extracted = []
        for path in image_paths:
            frame = cv2.imread(path)
            if frame is None:
                continue

            faces = self.detector.extract_faces(frame)
            for idx, face in enumerate(faces):
                face["source_type"] = "image"
                face["source_name"] = os.path.basename(path)
                face["timestamp_sec"] = None
                face["face_index"] = idx
                extracted.append(face)

        return extracted

    def process_video(self, video_path: str) -> list[dict]:
        """Samples frames from a video at `sample_fps` and extracts clean faces."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(1, int(round(native_fps / self.sample_fps)))

        extracted = []
        frame_idx = 0
        video_name = os.path.basename(video_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                timestamp_sec = round(frame_idx / native_fps, 2)
                faces = self.detector.extract_faces(frame)
                for idx, face in enumerate(faces):
                    face["source_type"] = "video"
                    face["source_name"] = video_name
                    face["timestamp_sec"] = timestamp_sec
                    face["face_index"] = idx
                    extracted.append(face)

            frame_idx += 1

        cap.release()
        return extracted

import os
from datetime import datetime
import pandas as pd


class RecognitionLogger:
    def __init__(self, log_path="data/logs/recognition_logs.csv"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.columns = [
            "logged_at",
            "source_type",
            "source_name",
            "timestamp_sec",
            "predicted_label",
            "similarity_score",
            "detection_score",
            "queue_status",
        ]
        if not os.path.exists(self.log_path):
            pd.DataFrame(columns=self.columns).to_csv(self.log_path, index=False)

    def log_batch(self, records: list[dict]):
        """Appends a batch of inference results to the CSV log file."""
        if not records:
            return

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        rows = []
        for r in records:
            rows.append({
                "logged_at": now_str,
                "source_type": r.get("source_type"),
                "source_name": r.get("source_name"),
                "timestamp_sec": r.get("timestamp_sec") if r.get("timestamp_sec") is not None else "",
                "predicted_label": r.get("pred_label"),
                "similarity_score": round(float(r.get("pred_score", 0.0)), 4),
                "detection_score": round(float(r.get("det_score", 0.0)), 4),
                "queue_status": r.get("queue_status", "recognized"),
            })

        df = pd.DataFrame(rows, columns=self.columns)
        df.to_csv(self.log_path, mode="a", header=False, index=False)

    def load_logs(self) -> pd.DataFrame:
        if os.path.exists(self.log_path):
            return pd.read_csv(self.log_path)
        return pd.DataFrame(columns=self.columns)

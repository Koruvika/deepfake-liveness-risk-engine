"""uniface RetinaFace wrapper: largest face -> bbox (xyxy) + 5 keypoints."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Det:
    bbox: np.ndarray   # (4,) x1, y1, x2, y2
    kps: np.ndarray    # (5, 2)
    score: float


class FaceDetector:
    def __init__(self):
        from uniface import RetinaFace

        self._net = RetinaFace()

    def __call__(self, frame):
        faces = self._net.detect(frame)
        if not faces:
            return None
        f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return Det(np.asarray(f.bbox, np.float32)[:4], np.asarray(f.landmarks, np.float32).reshape(5, 2),
                   float(f.confidence))

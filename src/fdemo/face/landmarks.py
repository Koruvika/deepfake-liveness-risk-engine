"""MediaPipe Tasks FaceLandmarker wrapper returning 478 landmarks in pixel coordinates."""
import cv2
import numpy as np


class Landmarker:
    def __init__(self, model_path):
        from mediapipe.tasks.python import BaseOptions, vision

        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
        )
        self._lm = vision.FaceLandmarker.create_from_options(opts)

    def __call__(self, frame_bgr):
        import mediapipe as mp

        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        res = self._lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not res.face_landmarks:
            return None
        h, w = frame_bgr.shape[:2]
        return np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]], np.float32)

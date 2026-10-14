"""ArcFace w600k_r50 (insightface buffalo_l) embeddings, L2-normalised."""
import cv2
import numpy as np

from fdemo.face.align import norm_crop


class ArcFace:
    def __init__(self, session):
        self._sess = session
        self._inp = session.get_inputs()[0].name

    def embed(self, img, kps) -> np.ndarray:
        crop, _ = norm_crop(img, kps, 112)
        blob = cv2.dnn.blobFromImage(crop, 1 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        e = self._sess.run(None, {self._inp: blob})[0].ravel()
        return e / (np.linalg.norm(e) + 1e-9)

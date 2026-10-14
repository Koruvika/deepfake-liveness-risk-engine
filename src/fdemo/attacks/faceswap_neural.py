"""A2: inswapper_128 face swap (the engine behind roop / Deep-Live-Cam), per frame, soft paste-back."""
import cv2
import numpy as np

from fdemo import models
from fdemo.face.align import norm_crop


def source_latent(source_frame) -> np.ndarray:
    det = models.detector()(source_frame)
    if det is None:
        raise ValueError("no face found in the source frame")
    emb = models.arcface().embed(source_frame, det.kps)
    _, emap = models.inswapper()
    lat = emb.reshape(1, -1) @ emap
    return (lat / np.linalg.norm(lat)).astype(np.float32)


def paste_back(frame, face128, M, erode=15, blur=31) -> np.ndarray:
    H, W = frame.shape[:2]
    IM = cv2.invertAffineTransform(M)
    back = cv2.warpAffine(face128, IM, (W, H))
    mask = cv2.warpAffine(np.full(face128.shape[:2], 255, np.uint8), IM, (W, H))
    mask = cv2.GaussianBlur(cv2.erode(mask, np.ones((erode, erode), np.uint8)), (blur, blur), 0)
    a = mask.astype(np.float32)[..., None] / 255
    return (back * a + frame * (1 - a)).astype(np.uint8)


def swap_frame(frame, latent, det) -> np.ndarray:
    sess, _ = models.inswapper()
    crop, M = norm_crop(frame, det.kps, 128)
    blob = cv2.dnn.blobFromImage(crop, 1 / 255.0, (128, 128), (0, 0, 0), swapRB=True)
    pred = sess.run(None, {"target": blob, "source": latent})[0]
    face = (pred[0].transpose(1, 2, 0)[:, :, ::-1].clip(0, 1) * 255).astype(np.uint8)
    return paste_back(frame, face, M)


def generate(target_frames, rng, source_frame) -> np.ndarray:
    latent = source_latent(source_frame)
    det = models.detector()
    out = []
    for f in target_frames:
        d = det(f)
        out.append(swap_frame(f, latent, d) if d is not None else f)
    return np.stack(out)

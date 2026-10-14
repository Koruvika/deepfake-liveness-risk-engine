"""insightface-compatible 5-point alignment (needed by ArcFace w600k_r50 and inswapper_128)."""
import cv2
import numpy as np

ARCFACE_DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    np.float32,
)


def estimate_norm(kps, size=112) -> np.ndarray:
    if size % 112 == 0:
        ratio, dx = size / 112.0, 0.0
    else:
        ratio, dx = size / 128.0, 8.0 * size / 128.0
    dst = ARCFACE_DST * ratio
    dst[:, 0] += dx
    M, _ = cv2.estimateAffinePartial2D(np.asarray(kps, np.float32), dst, method=cv2.LMEDS)
    if M is None:
        raise ValueError("degenerate keypoints")
    return M


def norm_crop(img, kps, size=112):
    M = estimate_norm(kps, size)
    return cv2.warpAffine(img, M, (size, size), borderValue=0), M

"""MediaPipe FaceMesh (478 pts) index sets and mask helpers."""
import cv2
import numpy as np

FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
             152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
INNER = [33, 133, 160, 159, 158, 144, 145, 153, 362, 263, 387, 386, 385, 373, 374, 380,
         70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
         1, 2, 98, 327, 168, 6, 197, 195, 5, 4,
         61, 291, 0, 17, 13, 14, 78, 308, 40, 270, 91, 321]
SWAP_POINTS = FACE_OVAL + INNER
LEFT_EYE_OUTER, RIGHT_EYE_OUTER = 33, 263

ROIS = {
    "forehead": [10, 338, 297, 299, 69, 67, 109],
    "left_cheek": [117, 118, 101, 36, 205, 187, 123],
    "right_cheek": [346, 347, 330, 266, 425, 411, 352],
}


def hull_mask(points, shape) -> np.ndarray:
    m = np.zeros(shape[:2], np.uint8)
    pts = np.round(np.asarray(points, np.float32)).astype(np.int32)
    cv2.fillConvexPoly(m, cv2.convexHull(pts), 255)
    return m


def roi_mask(lms, idxs, shape) -> np.ndarray:
    return hull_mask(np.asarray(lms)[idxs, :2], shape)

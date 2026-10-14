"""A1: landmark-based face swap (Delaunay warp on TARGET triangulation, LAB colour transfer,
seamlessClone or feathered alpha blending)."""
import cv2
import numpy as np

from fdemo import models
from fdemo.face.geometry import FACE_OVAL, SWAP_POINTS, hull_mask


def delaunay(points, shape) -> list:
    h, w = shape[:2]
    p = np.clip(np.asarray(points, np.float32), [0, 0], [w - 1, h - 1])
    sub = cv2.Subdiv2D((0, 0, w, h))
    for x, y in p:
        sub.insert((float(x), float(y)))
    tris = []
    for t in sub.getTriangleList():
        v = t.reshape(3, 2)
        if (v < 0).any() or (v[:, 0] >= w).any() or (v[:, 1] >= h).any():
            continue
        idx = tuple(int(np.argmin(np.linalg.norm(p - q, axis=1))) for q in v)
        if len(set(idx)) == 3:
            tris.append(idx)
    return tris


def warp_triangle(src, dst, ts, td) -> None:
    rs = cv2.boundingRect(np.float32([ts]))
    rd = cv2.boundingRect(np.float32([td]))
    if min(rs[2], rs[3], rd[2], rd[3]) == 0:
        return
    ts_o = np.float32([[x - rs[0], y - rs[1]] for x, y in ts])
    td_o = np.float32([[x - rd[0], y - rd[1]] for x, y in td])
    patch = src[rs[1]:rs[1] + rs[3], rs[0]:rs[0] + rs[2]]
    if patch.size == 0:
        return
    M = cv2.getAffineTransform(ts_o, td_o)
    warped = cv2.warpAffine(patch, M, (rd[2], rd[3]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    m = np.zeros((rd[3], rd[2]), np.float32)
    cv2.fillConvexPoly(m, np.int32(np.round(td_o)), 1.0, cv2.LINE_AA)
    y0, y1, x0, x1 = rd[1], rd[1] + rd[3], rd[0], rd[0] + rd[2]
    H, W = dst.shape[:2]
    if y0 < 0 or x0 < 0 or y1 > H or x1 > W:
        return
    region = dst[y0:y1, x0:x1]
    region[:] = region * (1 - m[..., None]) + warped * m[..., None]


def warp_face(src_img, src_pts, dst_shape, dst_pts, tris) -> np.ndarray:
    out = np.zeros(dst_shape, np.float32)
    s = src_img.astype(np.float32)
    for a, b, c in tris:
        warp_triangle(s, out, src_pts[[a, b, c]], dst_pts[[a, b, c]])
    return out


def color_transfer(src, ref, mask) -> np.ndarray:
    m = mask > 0
    s = cv2.cvtColor(np.clip(src, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    r = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    for c in range(3):
        ms, ss = s[..., c][m].mean(), s[..., c][m].std() + 1e-6
        mr, sr = r[..., c][m].mean(), r[..., c][m].std() + 1e-6
        s[..., c] = (s[..., c] - ms) / ss * sr + mr
    return cv2.cvtColor(np.clip(s, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def blend(target, warped, mask, mode, feather) -> np.ndarray:
    if mode == "clone":
        x, y, w, h = cv2.boundingRect(mask)
        try:
            # OpenCV 5 mutates the mask argument of seamlessClone -> pass a copy
            return cv2.seamlessClone(warped, target, mask.copy(), (x + w // 2, y + h // 2), cv2.NORMAL_CLONE)
        except cv2.error:
            pass
    k = feather // 2 + 2
    inner = cv2.erode(mask, np.ones((k, k), np.uint8))
    a = cv2.GaussianBlur(inner, (feather, feather), 0).astype(np.float32)[..., None] / 255
    return (warped * a + target * (1 - a)).astype(np.uint8)


def swap_frame(target, source, src_pts, dst_pts, tris, mode, feather) -> np.ndarray:
    h, w = target.shape[:2]
    dst_pts = np.clip(dst_pts, [0, 0], [w - 1, h - 1])
    warped = warp_face(source, src_pts, target.shape, dst_pts, tris)
    mask = cv2.erode(hull_mask(dst_pts[: len(FACE_OVAL)], target.shape), np.ones((5, 5), np.uint8))
    if mask.sum() == 0:
        return target
    return blend(target, color_transfer(warped, target, mask), mask, mode, feather)


def generate(target_frames, rng, source_frame) -> np.ndarray:
    lm = models.landmarker()
    src_lms = lm(source_frame)
    if src_lms is None:
        raise ValueError("no face found in the source frame")
    src_pts = src_lms[SWAP_POINTS]
    mode = str(rng.choice(["clone", "alpha"]))
    feather = int(rng.choice([7, 11, 15, 21]))
    tris, out = None, []
    for f in target_frames:
        lms = lm(f)
        if lms is None:
            out.append(f)
            continue
        dst_pts = lms[SWAP_POINTS]
        if tris is None:
            tris = delaunay(dst_pts, f.shape)
        out.append(swap_frame(f, source_frame, src_pts, dst_pts, tris, mode, feather))
    return np.stack(out)

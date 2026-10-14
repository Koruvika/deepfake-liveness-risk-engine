"""Video read/write helpers and the single 'SDK capture' normalisation step."""
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from fdemo import config


def read_frames(path, max_frames=None) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while max_frames is None or len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    return np.stack(frames)


def write_frames(frames, path, fps=config.FPS, crf=config.RAW_CRF) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _, h, w, _ = frames.shape
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p", str(path),
    ]
    subprocess.run(cmd, input=np.ascontiguousarray(frames, dtype=np.uint8).tobytes(), check=True)
    return path


def sdk_capture(src, dst) -> Path:
    """Identical final encode for bona fide and attacks: 640x360, 25 fps, 80 frames, H.264 CRF 23."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
        "-vf", (
            f"fps={config.FPS},"
            f"scale={config.WIDTH}:{config.HEIGHT}:force_original_aspect_ratio=decrease,"
            f"pad={config.WIDTH}:{config.HEIGHT}:(ow-iw)/2:(oh-ih)/2,setsar=1"
        ),
        "-frames:v", str(config.N_FRAMES), "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(config.CAPTURE_CRF), "-pix_fmt", "yuv420p",
        str(dst),
    ]
    subprocess.run(cmd, check=True)
    return dst


def probe(path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames:format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    d = json.loads(out.stdout)
    s = d["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return {
        "width": int(s["width"]),
        "height": int(s["height"]),
        "fps": float(num) / float(den),
        "frames": int(s["nb_read_frames"]),
        "duration": float(d["format"]["duration"]),
    }


def sample_indices(n, k) -> np.ndarray:
    return np.linspace(0, n - 1, k).round().astype(int)

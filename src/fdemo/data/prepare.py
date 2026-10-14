"""Fetch 15 video-only speech clips per actor from RAVDESS (HTTP Range, no full zip download),
trim to 3.2 s @ 25 fps and run the SDK-capture step. Also writes the identity folds."""
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from remotezip import RemoteZip

from fdemo import config
from fdemo.io.video import probe

RAVDESS_URL = "https://zenodo.org/records/1188976/files/Video_Speech_Actor_{actor:02d}.zip?download=1"
ZIP_TIMEOUT = (15, 120)


def with_retry(fn, tries=5, wait=15):
    """Zenodo regularly answers 429/5xx under load; retry with linear backoff."""
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == tries:
                raise RuntimeError(f"giving up after {tries} attempts: {e}") from e
            print(f"  retry {attempt}/{tries - 1} after error: {e}")
            time.sleep(wait * attempt)


def make_splits(actors, n_folds, seed) -> dict:
    """Identity folds with one or two actors of each sex per fold (odd actor id = male)."""
    rng = np.random.default_rng(seed)
    males = [a for a in actors if a % 2 == 1]
    females = [a for a in actors if a % 2 == 0]
    rng.shuffle(males)
    rng.shuffle(females)
    folds = {}
    for i, a in enumerate(males):
        folds[int(a)] = i % n_folds
    for i, a in enumerate(females):
        folds[int(a)] = (i + n_folds // 2) % n_folds
    return folds


def is_video_only_speech(name) -> bool:
    base = name.rsplit("/", 1)[-1]
    return base.startswith("02-01-") and base.endswith(".mp4")


def bona_id(actor, k) -> str:
    return f"bona_a{actor:02d}_c{k:02d}"


def trim_clip(src, dst) -> None:
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-t", str(config.CLIP_SEC),
        "-vf", f"fps={config.FPS},scale={config.WIDTH}:{config.HEIGHT}", "-frames:v", str(config.N_FRAMES),
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(config.RAW_CRF), "-pix_fmt", "yuv420p", str(dst),
    ]
    subprocess.run(cmd, check=True)


def write_splits(folds, seed) -> None:
    config.SPLITS.parent.mkdir(parents=True, exist_ok=True)
    config.SPLITS.write_text(json.dumps({"seed": seed, "folds": {str(a): f for a, f in folds.items()}}, indent=2))


def extract_member(url, name, td):
    """Open a fresh RemoteZip for this single extraction (a stale connection can't be retried)."""
    with RemoteZip(url, timeout=ZIP_TIMEOUT) as z:
        return z.extract(name, td)


def _write_csv_atomic(rows, path) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, path)


def fetch_ravdess(actors=config.ACTORS, clips=config.CLIPS_PER_ACTOR, seed=config.SEED) -> pd.DataFrame:
    # Folds are always computed over the full actor roster, not the `actors` subset passed in,
    # so that parallel/per-actor invocations don't assign wrong folds or clobber splits.json.
    folds = make_splits(config.ACTORS, config.N_FOLDS, seed)
    write_splits(folds, seed)
    rows = pd.read_csv(config.BONA_CSV).to_dict("records") if config.BONA_CSV.exists() else []
    for actor in actors:
        have = [r for r in rows if r["actor"] == actor]
        if len(have) >= clips:
            continue
        used = {r["source_name"] for r in have}
        rng = np.random.default_rng([seed, actor])
        url = RAVDESS_URL.format(actor=actor)
        with with_retry(lambda: RemoteZip(url, timeout=ZIP_TIMEOUT)) as z, tempfile.TemporaryDirectory() as td:
            names = sorted(n for n in z.namelist() if is_video_only_speech(n))
            for name in rng.permutation(names):
                if len(have) >= clips:
                    break
                if name in used:
                    continue
                src = Path(with_retry(lambda: extract_member(url, name, td)))
                if probe(src)["duration"] < config.CLIP_SEC + 0.02:
                    continue
                vid = bona_id(actor, len(have))
                raw = config.BONA_RAW / f"{vid}.mp4"
                try:
                    trim_clip(src, raw)
                    if probe(raw)["frames"] != config.N_FRAMES:
                        raw.unlink()
                        continue
                    # Lazy import: fdemo.attacks.runner imports fdemo.models, which would otherwise create
                    # an import cycle with fdemo.data.prepare at module load time.
                    from fdemo.attacks.runner import capture_like_attack

                    capture_like_attack(raw, config.VIDEOS / "bona" / f"{vid}.mp4")
                except subprocess.CalledProcessError as e:
                    print(f"  skipping {name}: {e}")
                    continue
                row = {"vid": vid, "actor": actor, "fold": folds[actor], "source_name": str(name)}
                have.append(row)
                rows.append(row)
                src.unlink()
                _write_csv_atomic(rows, config.BONA_CSV)
        print(f"actor {actor:02d}: {len(have)} clips")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(len(fetch_ravdess()), "bona fide clips")

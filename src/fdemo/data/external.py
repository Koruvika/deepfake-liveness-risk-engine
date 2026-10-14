"""Optional P3 test set: fake videos from TalkingHeadBench (6 generators, test split). Real videos are not
distributed by the dataset, so P3 reports APCER only."""
import logging
import os
import subprocess

import pandas as pd

from fdemo import config
from fdemo.io.video import probe

log = logging.getLogger(__name__)

REPO = "luchaoqi/TalkingHeadBench"
GENERATORS = ["AniPortraitAudio", "AniPortraitVideo", "EmoPortrait", "Hallo", "Hallo2", "LivePortrait"]


def select_files(files, per_gen) -> dict:
    out = {}
    for g in GENERATORS:
        sel = sorted(f for f in files if f.startswith(f"fake/{g}/test/") and f.endswith(".mp4"))
        out[g] = sel[:per_gen]
    return out


def ext_capture(src, dst) -> None:
    """Same SDK capture, but square talking-head videos are centred on a blurred 16:9 background.
    The foreground is cropped to the capture width so wide inputs never overlay at a negative x."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part.mp4")
    fc = (f"[0:v]fps={config.FPS},split[a][b];[a]scale={config.WIDTH}:{config.HEIGHT},boxblur=20[bg];"
          f"[b]scale=-2:{config.HEIGHT},crop='min(iw,{config.WIDTH})':{config.HEIGHT}[fg];"
          f"[bg][fg]overlay=(W-w)/2:0,setsar=1")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-t", str(config.CLIP_SEC), "-filter_complex", fc,
           "-frames:v", str(config.N_FRAMES), "-an", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", str(config.CAPTURE_CRF), "-pix_fmt", "yuv420p", str(tmp)]
    subprocess.run(cmd, check=True)
    os.replace(tmp, dst)


def fetch_external(per_gen=20) -> pd.DataFrame:
    from huggingface_hub import HfApi, hf_hub_download

    files = HfApi().list_repo_files(REPO, repo_type="dataset")
    rows = []
    for g, sel in select_files(files, per_gen).items():
        for i, f in enumerate(sel):
            vid = f"ext_{g}_{i:02d}"
            dst = config.VIDEOS / "EXT" / f"{vid}.mp4"
            try:
                if not dst.exists():
                    src = hf_hub_download(REPO, f, repo_type="dataset", local_dir=config.EXT_SRC)
                    ext_capture(src, dst)
                if probe(dst)["frames"] != config.N_FRAMES:
                    dst.unlink()
                    continue
                rows.append({"vid": vid, "path": f"data/videos/EXT/{vid}.mp4", "label": 1, "attack": f"EXT_{g}",
                             "group": "EXT", "actor": -1, "source_actor": -1, "fold": -1})
            except Exception as e:  # one failing video must not lose the whole fetch
                log.warning("failed to fetch %s (%s): %s", vid, f, e)
                if dst.exists():
                    dst.unlink()
                for stray in dst.parent.glob(f"{vid}*.part.mp4"):
                    stray.unlink()
                continue
    df = pd.DataFrame(rows)
    df.to_csv(config.EXT_CSV, index=False)
    return df

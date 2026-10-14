"""Plan attack jobs (identity-fold aware), run them in parallel, and build the manifest."""
import logging
import multiprocessing as mp
import os
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from fdemo import config, models
from fdemo.io.video import read_frames, sdk_capture, write_frames

MANIFEST_COLS = ["vid", "path", "label", "attack", "group", "actor", "source_actor", "fold"]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Job:
    vid: str
    attack: str
    target_vid: str
    target_actor: int
    fold: int
    source_vid: str | None      # A1/A2: bona clip of another actor in the same fold; A4: the A2 job vid
    source_actor: int | None
    seed: int


def split_counts(total, k) -> list:
    base, extra = divmod(total, k)
    return [base + 1 if i < extra else base for i in range(k)]


def plan_jobs(bona, seed=config.SEED, n_attack=config.N_ATTACK) -> list:
    rng = np.random.default_rng(seed)
    counts = split_counts(n_attack, config.N_FOLDS)
    jobs, a2_jobs = [], []
    for attack in config.ATTACKS:
        if attack == "A4":
            jobs += [Job(f"A4_{i:03d}", "A4", j.target_vid, j.target_actor, j.fold, j.vid, j.source_actor,
                         int(rng.integers(2**31))) for i, j in enumerate(a2_jobs)]
            continue
        cur = []
        for f in range(config.N_FOLDS):
            pool = bona[bona.fold == f].sort_values("vid").reset_index(drop=True)
            picks = pool.iloc[rng.choice(len(pool), counts[f], replace=False)]
            for t in picks.itertuples():
                src_vid = src_actor = None
                if attack in ("A1", "A2"):
                    src_actor = int(rng.choice(sorted(set(pool.actor) - {t.actor})))
                    src_vid = str(rng.choice(sorted(pool[pool.actor == src_actor].vid)))
                cur.append(Job(f"{attack}_{len(cur):03d}", attack, t.vid, int(t.actor), f, src_vid, src_actor,
                               int(rng.integers(2**31))))
        if attack == "A2":
            a2_jobs = cur
        jobs += cur
    return jobs


def _generators():
    from fdemo.attacks import faceswap_classic, faceswap_neural, replay, static_print

    return {"A1": faceswap_classic.generate, "A2": faceswap_neural.generate,
            "A3": replay.generate, "A5": static_print.generate}


def pick_source_frame(frames):
    """Find a frame with a detectable face: try the middle frame first, then 8 frames spread
    across the clip. Raises ValueError only if none of those candidates has a detectable face."""
    n = len(frames)
    candidates = [n // 2, *np.linspace(0, n - 1, 8).round().astype(int).tolist()]
    det = models.detector()
    seen = set()
    for i in candidates:
        if i in seen:
            continue
        seen.add(i)
        if det(frames[i]) is not None:
            return frames[i]
    raise ValueError("no face found in any candidate source frame")


def manifest_row(job) -> dict:
    return {"vid": job.vid, "path": f"data/videos/{job.attack}/{job.vid}.mp4", "label": 1, "attack": job.attack,
            "group": config.ATTACK_GROUP[job.attack], "actor": job.target_actor,
            "source_actor": job.source_actor if job.source_actor is not None else -1, "fold": job.fold}


def bona_rows(bona) -> pd.DataFrame:
    return pd.DataFrame({
        "vid": bona.vid, "path": "data/videos/bona/" + bona.vid + ".mp4", "label": 0, "attack": "bona",
        "group": "bona", "actor": bona.actor, "source_actor": -1, "fold": bona.fold,
    })[MANIFEST_COLS]


def run_job(job):
    """Generate one attack video (resumable, atomic). Returns the manifest row, or None if the
    job was skipped (e.g. an A4 job whose A2 source raw was never produced)."""
    final = config.VIDEOS / job.attack / f"{job.vid}.mp4"
    if final.exists():
        return manifest_row(job)

    rng = np.random.default_rng(job.seed)
    if job.attack == "A4":
        src_raw = config.ATTACK_RAW / "A2" / f"{job.source_vid}.mp4"
        if not src_raw.exists():
            log.warning("skipping A4 job %s: missing A2 raw %s", job.vid, src_raw)
            return None
        from fdemo.attacks import injection

        frames = injection.generate(read_frames(src_raw), rng)
    else:
        target = read_frames(config.BONA_RAW / f"{job.target_vid}.mp4")
        source = None
        if job.source_vid:
            src_frames = read_frames(config.BONA_RAW / f"{job.source_vid}.mp4")
            source = pick_source_frame(src_frames)
        frames = _generators()[job.attack](target, rng, source)

    keep_raw = config.ATTACK_RAW / "A2" / f"{job.vid}.mp4" if job.attack == "A2" else None
    _write_and_capture(frames, final, keep_raw=keep_raw)

    return manifest_row(job)


def _write_and_capture(frames, final, keep_raw=None) -> Path:
    """Write frames (CRF 12) then run the identical SDK-capture encode, atomically. Shared by bona fide
    (via `capture_like_attack`) and every attack (via `run_job`), so the two paths cannot drift. If
    `keep_raw` is given, the intermediate raw file is persisted there instead of a scratch tempdir (A2's
    raw feeds A4)."""
    final = Path(final)
    with tempfile.TemporaryDirectory() as td:
        if keep_raw is not None:
            keep_raw = Path(keep_raw)
            raw_tmp = keep_raw.with_name(keep_raw.stem + ".part.mp4")
            write_frames(frames, raw_tmp, crf=config.RAW_CRF)
            keep_raw.parent.mkdir(parents=True, exist_ok=True)
            os.replace(raw_tmp, keep_raw)
            raw = keep_raw
        else:
            raw = Path(td) / "raw.mp4"
            write_frames(frames, raw, crf=config.RAW_CRF)

        tmp_final = final.with_name(final.stem + ".part.mp4")
        sdk_capture(raw, tmp_final)
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp_final, final)

    return final


def capture_like_attack(raw_path, final_path) -> Path:
    """Send a bona fide raw clip through the identical 'null attack' path: decode -> re-encode
    (CRF 12) -> SDK capture -> atomic replace. This is exactly what every attack does (minus the attack
    generator itself), so bona fide and attacks cannot differ because of an extra generation step."""
    frames = read_frames(raw_path)
    return _write_and_capture(frames, final_path)


def recapture_bona(bona_df, workers=config.WORKERS) -> None:
    """Re-run every bona fide video through `capture_like_attack` so it gets the same decode/re-encode
    generation attacks get. Idempotent: skips entirely if the marker file already exists; writes the
    marker only when every video succeeds."""
    marker = config.VIDEOS / "bona" / ".null_attack_v1"
    if marker.exists():
        return

    jobs = [(config.BONA_RAW / f"{vid}.mp4", config.VIDEOS / "bona" / f"{vid}.mp4") for vid in bona_df.vid]
    failed = []
    if workers == 0:  # in-process (tests, debugging)
        for raw, final in tqdm(jobs, desc="recapture"):
            if not _recapture_job_safe(raw, final):
                failed.append(str(final))
    else:
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("forkserver"), initializer=_init_worker) as ex:
            futs = {ex.submit(_recapture_job_safe, raw, final): final for raw, final in jobs}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="recapture"):
                if not fut.result():
                    failed.append(str(futs[fut]))

    if failed:
        msg = f"recapture_bona: {len(failed)} failed: {failed}"
        log.warning(msg)
        print(msg)
        return

    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("")


def _recapture_job_safe(raw, final) -> bool:
    try:
        capture_like_attack(raw, final)
        return True
    except Exception as e:
        log.exception("recapture failed for %s: %s", final, e)
        return False


def run_job_safe(job):
    """Wrap run_job so one failing job never kills the whole batch."""
    try:
        return run_job(job)
    except Exception as e:
        log.exception("job %s (%s) failed: %s", job.vid, job.attack, e)
        return None


def _init_worker():
    import cv2

    cv2.setNumThreads(config.THREADS)


def generate_all(jobs, workers=config.WORKERS) -> list:
    rows, failed = [], []
    for stage in ([j for j in jobs if j.attack != "A4"], [j for j in jobs if j.attack == "A4"]):
        if not stage:
            continue
        if workers == 0:  # in-process (tests, debugging)
            for j in tqdm(stage, desc="generate"):
                r = run_job_safe(j)
                (rows if r is not None else failed).append(r if r is not None else j.vid)
        else:
            with ProcessPoolExecutor(workers, mp_context=mp.get_context("forkserver"), initializer=_init_worker) as ex:
                futs = {ex.submit(run_job_safe, j): j for j in stage}
                for fut in tqdm(as_completed(futs), total=len(futs), desc="generate"):
                    j = futs[fut]
                    r = fut.result()
                    (rows if r is not None else failed).append(r if r is not None else j.vid)

    counts = Counter(r["attack"] for r in rows)
    msg = f"generated {dict(counts)}; failed {len(failed)}" + (f": {failed}" if failed else "")
    log.info(msg)
    print(msg)
    return rows


def build_manifest(bona, attack_rows, ext_rows=None) -> pd.DataFrame:
    parts = [bona_rows(bona), pd.DataFrame(attack_rows, columns=MANIFEST_COLS)]
    if ext_rows is not None and len(ext_rows):
        parts.append(pd.DataFrame(ext_rows)[MANIFEST_COLS])
    m = pd.concat(parts, ignore_index=True)
    config.MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(config.MANIFEST, index=False)
    return m

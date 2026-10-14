"""Run every signal on every manifest video (parallel, resumable) -> cache/features.csv."""
import logging
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from fdemo import config
from fdemo.ctx import build_ctx
from fdemo.signals import SIGNALS

log = logging.getLogger(__name__)


def extract_video(vid, path, cache_dir=None) -> dict:
    row = {"vid": vid}
    try:
        st = os.stat(path)
        row["src_mtime_ns"] = st.st_mtime_ns
        row["src_size"] = st.st_size
    except OSError:
        pass
    t = time.perf_counter()
    ctx = build_ctx(path, vid, cache_dir=cache_dir)
    row["lat_ctx"] = time.perf_counter() - t
    row["no_face"] = ctx.no_face
    for name, fn in SIGNALS.items():
        t = time.perf_counter()
        try:
            row.update(fn(ctx))
        except Exception as e:  # one failing signal must not lose the whole video
            log.warning("signal %s failed on %s: %s", name, vid, e)
            row[f"err_{name}"] = 1
        row[f"lat_{name}"] = time.perf_counter() - t
    return row


def _init_worker():
    import cv2

    cv2.setNumThreads(1)


def _job(args) -> dict:
    vid = args[0]
    try:
        return extract_video(*args)
    except Exception as e:  # one failing video must not kill the whole batch
        log.exception("extract_video failed on %s", vid)
        return {"vid": vid, "error": repr(e)}


def _is_stale(row, path) -> bool:
    """A cached row is stale (must be re-extracted) if it predates src_mtime_ns/src_size tracking, or the
    source video at `path` has changed since it was extracted."""
    try:
        st = os.stat(path)
    except OSError:
        return False  # source gone: nothing to compare against, leave the cached row alone
    for col, cur in (("src_mtime_ns", st.st_mtime_ns), ("src_size", st.st_size)):
        val = row.get(col)
        if val is None or pd.isna(val) or int(val) != cur:
            return True
    return False


SRC_STAT_DTYPE = {"src_mtime_ns": "Int64", "src_size": "Int64"}


def extract_all(manifest, out_csv=config.FEATURES, workers=config.WORKERS, flush_every=25) -> pd.DataFrame:
    # Nullable Int64 (not the default float64) so a single NaN in src_mtime_ns/src_size doesn't force the
    # whole column to float and silently truncate precision on these ~1.8e18 nanosecond timestamps, which
    # would make unrelated unchanged rows look stale.
    existing = pd.read_csv(out_csv, dtype=SRC_STAT_DTYPE) if out_csv.exists() else pd.DataFrame(columns=["vid"])
    paths = {r.vid: str(config.ROOT / r.path) for r in manifest.itertuples()}
    done = set(existing["vid"])
    existing_by_vid = existing.set_index("vid", drop=False).to_dict("index") if len(existing) else {}
    stale = {vid for vid in done if vid in paths and _is_stale(existing_by_vid.get(vid, {}), paths[vid])}
    if stale:
        existing = existing[~existing["vid"].isin(stale)].reset_index(drop=True)
    todo = [(vid, path) for vid, path in paths.items() if vid not in done or vid in stale]
    new = []
    failed = []
    n_ok = 0

    def flush():
        nonlocal existing, new
        if new:
            existing = pd.concat([existing, pd.DataFrame(new)], ignore_index=True)
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            tmp = out_csv.with_suffix(".tmp")
            existing.to_csv(tmp, index=False)
            os.replace(tmp, out_csv)
            new = []

    def handle(row):
        nonlocal n_ok
        if "error" in row:
            failed.append(row["vid"])
        else:
            new.append(row)
            n_ok += 1

    try:
        if workers == 0:  # in-process (tests, debugging)
            for a in tqdm(todo, desc="extract"):
                vid = a[0]
                try:
                    handle(extract_video(*a))
                except Exception as e:  # one failing video must not kill the whole batch
                    log.exception("extract_video failed on %s", vid)
                    handle({"vid": vid, "error": repr(e)})
        else:
            with ProcessPoolExecutor(workers, mp_context=mp.get_context("forkserver"), initializer=_init_worker) as ex:
                futs = [ex.submit(_job, a) for a in todo]
                for fut in tqdm(as_completed(futs), total=len(futs), desc="extract"):
                    handle(fut.result())
                    if len(new) >= flush_every:
                        flush()
    finally:
        flush()

    msg = f"extracted {n_ok}, failed {len(failed)}" + (f": {failed}" if failed else "")
    log.info(msg)
    print(msg)
    return existing

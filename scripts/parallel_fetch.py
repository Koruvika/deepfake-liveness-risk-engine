"""Ops helper (not part of the package; run with `uv run python scripts/parallel_fetch.py 6`):
run fetch_ravdess per actor in parallel processes (Zenodo throttles per connection),
each writing its own CSV, then merge into data/bona.csv."""
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from fdemo import config
from fdemo.data import prepare


def one(actor):
    per = config.DATA / f"bona_{actor:02d}.csv"
    if config.BONA_CSV.exists() and not per.exists():
        rows = pd.read_csv(config.BONA_CSV)
        rows[rows.actor == actor].to_csv(per, index=False) if (rows.actor == actor).any() else None
    prepare.config.BONA_CSV = per
    df = prepare.fetch_ravdess(actors=[actor])
    return len(df)


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    with ProcessPoolExecutor(workers) as ex:
        for a, n in zip(config.ACTORS, ex.map(one, config.ACTORS)):
            print(f"actor {a:02d}: {n}", flush=True)
    parts = [pd.read_csv(config.DATA / f"bona_{a:02d}.csv") for a in config.ACTORS if (config.DATA / f"bona_{a:02d}.csv").exists()]
    merged = pd.concat(parts, ignore_index=True).sort_values("vid")
    prepare._write_csv_atomic(merged.to_dict("records"), config.BONA_CSV)
    print(len(merged), "bona fide clips")

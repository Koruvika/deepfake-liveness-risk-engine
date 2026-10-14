# fdemo — CPU-only multi-layer deepfake / spoof / injection risk engine

Demo for hardening a FaceTec-based identity-verification flow. It generates attacks (classic and neural face swap,
screen replay, virtual-camera injection, printed photo) from public RAVDESS videos, scores them with two open-source
baselines (MiniFASNet, SBI) plus our own forensic signals (blend-boundary noise, spectrum/moiré, stream timing and
noise correlation, identity drift, rPPG), fuses everything into a risk engine with allow / review / deny routing
and a FaceTec-aware vendor adapter, and reports ISO/IEC 30107-3-style metrics.

## Quick start

```bash
uv sync
uv run fdemo fetch          # models (~900 MB), 180 RAVDESS clips via HTTP Range, TalkingHeadBench fakes
uv run fdemo generate       # 750 attacks
uv run fdemo extract        # features for every video
uv run fdemo bench          # reports/benchmark_report.md + figures + models/fusion_all.joblib
uv run fdemo recommend      # reports/recommendations.md
uv run fdemo scan some.mp4 --vendor facetec_session.json --photo-id --expected-ref <server-issued-ref>
```

Realistic timings on a modest machine (16 logical cores, ~5 GB free RAM) — all strongly dependent on Zenodo/network
conditions and on CPU thermal throttling, so treat these as orientation, not an SLA:
- `fetch`: ≈ 15–60 min when Zenodo cooperates; `fdemo fetch` downloads actors sequentially, which can take hours
  if Zenodo is slow or flaky. `uv run python scripts/parallel_fetch.py 6` fetches the RAVDESS clips per actor in
  parallel (≈ 15 min in our run). Both are resumable — finished clips are kept, re-run on failure.
- `generate`: ≈ 2–3 h; the A2 neural face-swap (inswapper) attack dominates the wall-clock time (~1 s/frame on
  CPU). Resumable — finished videos are kept, failed jobs are logged and retried on the next run.
- `extract`: ≈ 1–1.5 h (SBI is the slowest signal per video). Resumable, outputs written atomically.

Notes:
- `fetch` is resumable; Zenodo is sometimes slow or returns 504 — re-run it. RAVDESS clips are fetched with HTTP
  Range requests (no full-zip download).
- Generation and extraction are resumable: finished videos/rows are kept, failed jobs are logged and retried on
  the next run; outputs are written atomically.
- On machines with less than 8 GB free RAM, run `fdemo extract --workers 2` — the default worker count can push
  concurrent SBI/MiniFASNet model loads past available memory and trigger OOM kills.
- `fdemo scan` runs the same "SDK capture" normalisation step the pipeline applies to every generated video before
  scoring, so pass it a raw/camera-captured video (e.g. straight off a phone or webcam) — not a file already
  under `data/videos/`, which has already been through that normalisation and would be double-processed.
- `--expected-ref` binds the vendor session JSON to a specific server-issued `externalDatabaseRefID`: every
  response in the session must carry that exact value or the scan is rejected, instead of only checking that the
  responses are internally consistent with each other.
- `fdemo scan` needs `fdemo bench` to have produced `models/fusion_all.joblib`.

Results: [`deliverables/evaluation_summary.md`](deliverables/evaluation_summary.md) — data, baselines, method,
metrics, results vs. open-source baselines, limitations. Running `bench` / `recommend` regenerates the full test
report (`reports/benchmark_report.md`) and the thresholds / monitoring / manual-review rules
(`reports/recommendations.md`) locally.
import json
from pathlib import Path

import pandas as pd
import typer

from fdemo import config

app = typer.Typer(add_completion=False, help="CPU-only deepfake / spoof / injection risk engine demo")


@app.command()
def fetch(external: bool = typer.Option(True, "--external/--no-external", help="Also fetch TalkingHeadBench fakes (P3)")):
    """Download model weights, RAVDESS clips (HTTP Range) and optionally TalkingHeadBench fakes."""
    from fdemo.data.fetch_models import fetch_models
    from fdemo.data.prepare import fetch_ravdess

    fetch_models()
    typer.echo(f"bona fide clips: {len(fetch_ravdess())}")
    if external:
        try:
            from fdemo.data.external import fetch_external

            typer.echo(f"external fakes: {len(fetch_external())}")
        except Exception as e:  # P3 is optional
            typer.echo(f"WARNING: TalkingHeadBench fetch failed, P3 will be skipped: {e}")


@app.command()
def generate(workers: int = config.WORKERS):
    """Generate attacks A1-A5 and write data/manifest.csv."""
    from fdemo.attacks.runner import build_manifest, generate_all, plan_jobs, recapture_bona

    bona = pd.read_csv(config.BONA_CSV)
    # Bona fide must go through the identical decode/re-encode + SDK-capture path every attack does; this is
    # a no-op (skips via the marker file) once it has already run once.
    recapture_bona(bona, workers=workers)
    rows = generate_all(plan_jobs(bona), workers=workers)
    ext = pd.read_csv(config.EXT_CSV) if config.EXT_CSV.exists() else None
    typer.echo(f"manifest rows: {len(build_manifest(bona, rows, ext))}")


@app.command()
def extract(workers: int = config.WORKERS):
    """Run all signals on every manifest video -> cache/features.csv."""
    from fdemo.features import extract_all

    typer.echo(f"feature rows: {len(extract_all(pd.read_csv(config.MANIFEST), workers=workers))}")


@app.command()
def bench():
    """Run protocols P1-P3, write reports/benchmark_report.md and the final model."""
    from fdemo.eval.bench import run_bench

    typer.echo(f"report: {run_bench()}")


@app.command()
def recommend():
    """Write reports/recommendations.md from the P2 risk-engine results."""
    from fdemo.eval.recommend import write_recommendations

    typer.echo(f"recommendations: {write_recommendations()}")


@app.command()
def scan(video: Path = typer.Argument(..., exists=True, dir_okay=False, help="Video file to score"),
         vendor: Path = typer.Option(None, exists=True, dir_okay=False,
                                     help="FaceTec session JSON (object or list of step responses)"),
         photo_id: bool = typer.Option(False, help="Session includes 3D:2D Photo ID matching"),
         expected_ref: str = typer.Option(None, "--expected-ref",
                                          help="Server-issued externalDatabaseRefID the session must carry on "
                                               "every response (binds the session to this user/transaction)")):
    """Score one video and print risk, decision, top signals and latency."""
    from fdemo.scan import scan_video

    if expected_ref and not vendor:
        raise SystemExit("--expected-ref requires --vendor")

    typer.echo(json.dumps(scan_video(video, vendor, photo_id, expected_ref), indent=2, default=str))


if __name__ == "__main__":
    app()

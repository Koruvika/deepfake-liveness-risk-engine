"""Paths and constants shared by every stage of the demo."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MODELS = ROOT / "models"
CACHE = ROOT / "cache"
REPORTS = ROOT / "reports"

BONA_RAW = DATA / "bona_raw"          # trimmed 25 fps clips before SDK capture (attack sources)
ATTACK_RAW = DATA / "attack_raw"      # persisted pre-capture attack frames (A2 feeds A4)
VIDEOS = DATA / "videos"              # final SDK-captured videos: videos/<attack|bona|EXT>/<vid>.mp4
EXT_SRC = DATA / "external_src"
SPLITS = DATA / "splits.json"
BONA_CSV = DATA / "bona.csv"
EXT_CSV = DATA / "external.csv"
MANIFEST = DATA / "manifest.csv"
FEATURES = CACHE / "features.csv"
RESULTS = CACHE / "results"
FIGURES = REPORTS / "figures"
BUNDLE = MODELS / "fusion_all.joblib"

FPS = 25
CLIP_SEC = 3.2
N_FRAMES = 80
WIDTH, HEIGHT = 640, 360
N_SAMPLED = 16
CAPTURE_CRF = 23
RAW_CRF = 12

ACTORS = list(range(1, 13))           # RAVDESS: odd = male, even = female
CLIPS_PER_ACTOR = 15
N_FOLDS = 4
N_ATTACK = 150                        # attacks per type (iBeta: 150 per PAI species)
SEED = 1337

ATTACKS = ["A1", "A2", "A3", "A4", "A5"]
ATTACK_GROUP = {"A1": "G1", "A2": "G2", "A3": "G3", "A4": "G2", "A5": "G4"}
ATTACK_INFO = {
    "A1": ("Face swap (classic, landmark warp + blending)", "Deepfake (digital)"),
    "A2": ("Face swap (neural, inswapper_128)", "Deepfake (digital)"),
    "A3": ("Screen replay (simulated recapture)", "≈ iBeta Level 1 PAI: screen replay"),
    "A4": ("Virtual-camera injection of A2 stream", "CEN/TS 18099 IAI: virtual camera"),
    "A5": ("Static photo / print (simulated)", "≈ iBeta Level 1 PAI: print"),
}

WORKERS = 3
THREADS = 4

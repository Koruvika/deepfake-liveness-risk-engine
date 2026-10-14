"""Download every model weight into models/ (idempotent)."""
import shutil
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

from fdemo import config

URLS = {
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "MiniFASNetV2.onnx": "https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV2.onnx",
    "MiniFASNetV1SE.onnx": "https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV1SE.onnx",
    "inswapper_128.onnx": "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
}
BUFFALO_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
SBI_GDRIVE_ID = "1X0-NYT8KPursLZZdxduRQju6E52hauV0"


def download(url, dest) -> Path:
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f, tqdm(
            total=int(r.headers.get("Content-Length", 0)), unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed for {dest.name}. Download it manually from {url} into {dest}") from e
    tmp.rename(dest)
    return dest


def fetch_models() -> None:
    for name, url in URLS.items():
        download(url, config.MODELS / name)

    r50 = config.MODELS / "buffalo_l" / "w600k_r50.onnx"
    if not r50.exists():
        z = download(BUFFALO_URL, config.MODELS / "buffalo_l.zip")
        with zipfile.ZipFile(z) as zf:
            member = next(n for n in zf.namelist() if n.endswith("w600k_r50.onnx"))
            r50.parent.mkdir(parents=True, exist_ok=True)
            r50_tmp = r50.with_suffix(r50.suffix + ".part")
            with zf.open(member) as src, open(r50_tmp, "wb") as dst:
                shutil.copyfileobj(src, dst)
            r50_tmp.rename(r50)
        z.unlink()

    sbi = config.MODELS / "sbi_FFc23.tar"
    if not sbi.exists():
        import gdown

        if gdown.download(id=SBI_GDRIVE_ID, output=str(sbi), quiet=False) is None:
            raise RuntimeError(
                f"SBI weights download failed. Download https://drive.google.com/file/d/{SBI_GDRIVE_ID}/view into {sbi}"
            )

    from fdemo import models

    models.detector()  # triggers the uniface RetinaFace weight download


if __name__ == "__main__":
    fetch_models()

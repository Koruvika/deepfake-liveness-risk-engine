"""Lazily-loaded, per-process model singletons."""
from functools import lru_cache

import numpy as np

from fdemo import config


def ort_session(path):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = config.THREADS
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


@lru_cache(maxsize=None)
def detector():
    import uniface

    uniface.set_cache_dir(str(config.MODELS / "uniface"))
    from fdemo.face.detector import FaceDetector

    fd = FaceDetector()
    fd._net.session = ort_session(fd._net._model_path)
    return fd


@lru_cache(maxsize=None)
def landmarker():
    from fdemo.face.landmarks import Landmarker

    return Landmarker(config.MODELS / "face_landmarker.task")


@lru_cache(maxsize=None)
def arcface():
    from fdemo.face.arcface import ArcFace

    return ArcFace(ort_session(config.MODELS / "buffalo_l" / "w600k_r50.onnx"))


@lru_cache(maxsize=None)
def minifas():
    return (
        (ort_session(config.MODELS / "MiniFASNetV2.onnx"), 2.7),
        (ort_session(config.MODELS / "MiniFASNetV1SE.onnx"), 4.0),
    )


@lru_cache(maxsize=None)
def sbi():
    import torch
    from efficientnet_pytorch import EfficientNet

    torch.set_num_threads(config.THREADS)
    ck = torch.load(config.MODELS / "sbi_FFc23.tar", map_location="cpu", weights_only=False)
    net = EfficientNet.from_name("efficientnet-b4", num_classes=2)
    net.load_state_dict({k.removeprefix("net."): v for k, v in ck["model"].items()}, strict=True)
    return net.eval()


@lru_cache(maxsize=None)
def inswapper():
    import onnx
    from onnx import numpy_helper

    path = config.MODELS / "inswapper_128.onnx"
    emap = numpy_helper.to_array(onnx.load(str(path)).graph.initializer[-1]).astype(np.float32)
    return ort_session(path), emap

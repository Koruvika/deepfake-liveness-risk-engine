import pytest


def pytest_collection_modifyitems(config, items):
    from fdemo import config as C

    models_ok = (C.MODELS / "inswapper_128.onnx").exists() and (C.MODELS / "sbi_FFc23.tar").exists()
    data_ok = C.BONA_CSV.exists()
    skip_models = pytest.mark.skip(reason="model weights not downloaded (run `fdemo fetch`)")
    skip_data = pytest.mark.skip(reason="RAVDESS clips not prepared (run `fdemo fetch`)")
    for item in items:
        if "models" in item.keywords and not models_ok:
            item.add_marker(skip_models)
        if "data" in item.keywords and not data_ok:
            item.add_marker(skip_data)

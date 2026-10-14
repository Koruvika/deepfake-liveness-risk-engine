from fdemo.signals import (
    blend_noise, deepfake_sbi, identity, pad_minifasnet, rppg, spectrum, temporal,
)

SIGNALS = {
    "pad": pad_minifasnet.score,
    "sbi": deepfake_sbi.score,
    "blend": blend_noise.score,
    "spectrum": spectrum.score,
    "temporal": temporal.score,
    "identity": identity.score,
    "rppg": rppg.score,
}

SIGNAL_PREFIX = {
    "pad": "pad_",
    "sbi": "sbi_",
    "blend": "bn_",
    "spectrum": "sp_",
    "temporal": "tm_",
    "identity": "id_",
    "rppg": "rp_",
}

from fdemo import config


def test_attack_groups_pair_a2_and_a4():
    assert config.ATTACK_GROUP["A2"] == config.ATTACK_GROUP["A4"]
    assert len(set(config.ATTACK_GROUP.values())) == 4


def test_clip_length_matches_frame_count():
    assert round(config.CLIP_SEC * config.FPS) == config.N_FRAMES

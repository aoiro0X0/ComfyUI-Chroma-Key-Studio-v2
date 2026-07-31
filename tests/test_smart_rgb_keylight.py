import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


engine = load_module("smart_rgb_engine", "core/engine.py")


def matte_for(pixel, key, **kwargs):
    image = torch.tensor(pixel, dtype=torch.float32).view(1, 3, 1, 1)
    key_rgb = torch.tensor(key, dtype=torch.float32).view(1, 3, 1, 1)
    matte, *_ = engine.compute_matte(
        image, key_rgb, tolerance=1.0, clip_black=-0.02,
        clip_white=0.30, **kwargs
    )
    return float(matte.item())


def test_all_three_primary_screens_key_out():
    for colour in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):
        assert matte_for(colour, colour) < 0.01


def test_black_subject_is_preserved_for_every_primary():
    for key in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):
        assert matte_for([0, 0, 0], key) > 0.99


def test_cyan_subject_is_preserved_on_red_screen():
    assert matte_for([0, 0.8, 0.8], [1, 0, 0]) > 0.99


def test_dark_coloured_screen_recovery_improves_matte():
    without = matte_for([0.30, 0, 0], [1, 0, 0], shadow_recovery=0.0)
    with_recovery = matte_for([0.30, 0, 0], [1, 0, 0], shadow_recovery=0.85)
    assert with_recovery < without * 0.35


def test_hybrid_despill_handles_whole_frame():
    work = torch.zeros((1, 3, 4, 5), dtype=torch.float32)
    work[:, 1] = 0.9
    work[:, 0] = 0.1
    work[:, 2] = 0.1
    key = torch.tensor([0, 1, 0], dtype=torch.float32).view(1, 3, 1, 1)
    key = engine.build_key_vector(key)
    proj = (work * key).sum(1, keepdim=True)
    dist = torch.linalg.norm(work - proj * key, dim=1, keepdim=True)
    matte = torch.zeros((1, 1, 4, 5), dtype=torch.float32)
    result = engine._spill_suppress(work, proj, dist, key, matte, 1.0)
    assert torch.all(result[:, 1] < work[:, 1])

import importlib.util
from pathlib import Path
import sys
import types
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]

V2_IDS = {
    "ChromaKeyStudioSmartBackgroundV2",
    "ChromaKeyStudioKeylightV2",
    "ChromaKeyStudioSpillArgsV2",
    "ChromaKeyStudioProtectHighlightsArgsV2",
    "ChromaKeyStudioEdgeArgsV2",
    "ChromaKeyStudioMatteMathArgsV2",
    "ChromaKeyStudioSamplerArgsV2",
}

LEGACY_IDS = {
    "AutoChromaSmartBackground",
    "KeylightSmartBackground",
    "KeylightCoreHubV3",
    "Key Spill/Algo Args (V2.3.6fixE2_clean)",
    "Key Protect Highlights Args (V2.3.6fixE2_clean)",
    "Key Edge Args (V2.3.6fixE2_clean)",
    "Key Matte Math Args (V2.3.6fixE2_clean)",
    "Key Sampler Args (V2.3.6fixE2_clean)",
}


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


engine = load_module("smart_rgb_engine", "core/engine.py")


def load_plugin():
    spec = importlib.util.spec_from_file_location(
        "chroma_key_studio_test_package",
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_arbitrary_hue_screens_and_shadows_key_out():
    colours = (
        [0, 1, 1], [1, 1, 0], [1, 0, 1], [0.75, 0, 1], [1, 0.5, 0],
    )
    for colour in colours:
        assert matte_for(colour, colour) < 0.01
        shadow = [value * 0.25 for value in colour]
        assert matte_for(shadow, colour) < 0.01


def test_neutral_black_gray_white_and_metal_are_preserved_on_intermediate_keys():
    keys = ([1, 0, 1], [0.75, 0, 1], [1, 1, 0], [0, 1, 1])
    neutrals = ([0, 0, 0], [0.15, 0.15, 0.15], [0.5, 0.5, 0.5], [1, 1, 1],
                [0.42, 0.46, 0.48])
    for key in keys:
        for pixel in neutrals:
            assert matte_for(pixel, key) > 0.98


def test_magenta_key_preserves_red_blue_and_cyan_subject_colours():
    for subject in ([1, 0, 0], [0, 0, 1], [0, 1, 1]):
        assert matte_for(subject, [1, 0, 1]) > 0.98


def test_full_vector_despill_does_not_change_neutral_gray():
    work = torch.full((1, 3, 4, 5), 0.35, dtype=torch.float32)
    key = torch.tensor([1, 0, 1], dtype=torch.float32).view(1, 3, 1, 1)
    key = engine.build_key_vector(key)
    matte = torch.full((1, 1, 4, 5), 0.4, dtype=torch.float32)
    result = engine._spill_suppress(work, None, None, key, matte, 2.0)
    assert torch.allclose(result, work, atol=1e-6)


def make_motion_blur_strip(key, foreground, alpha_values, size=64):
    height = width = int(size)
    key_rgb = torch.tensor(key, dtype=torch.float32).view(1, 3, 1, 1)
    foreground_rgb = torch.tensor(foreground, dtype=torch.float32).view(1, 3, 1, 1)
    alpha = torch.zeros((1, 1, height, width), dtype=torch.float32)
    start = width // 4
    for offset, value in enumerate(alpha_values):
        alpha[:, :, :, start + offset] = float(value)
    alpha[:, :, :, start + len(alpha_values):] = 1.0
    image = foreground_rgb * alpha + key_rgb * (1.0 - alpha)
    return image, key_rgb, alpha, start


def test_default_adaptive_cleanup_recovers_fast_motion_edge():
    values = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    image, key, _, start = make_motion_blur_strip(
        [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], values
    )
    _, matte_before, _, clean_before = engine.run(
        image, key, 1.0, -0.02, 0.30,
        edge_soft=0.05, defringe=0.07, adaptive_cleanup=False,
    )
    _, matte_after, _, clean_after = engine.run(
        image, key, 1.0, -0.02, 0.30,
        edge_soft=0.05, defringe=0.07,
    )

    row = 32
    half = start + values.index(0.5)
    three_quarters = start + values.index(0.75)
    assert matte_before[0, 0, row, three_quarters] > 0.98
    assert 0.60 < matte_after[0, 0, row, three_quarters] < 0.85
    assert 0.40 < matte_after[0, 0, row, half] < 0.65
    assert clean_before[0, 1, row, three_quarters] > 0.20
    assert clean_after[0, 1, row, three_quarters] < 0.08
    assert clean_after[0, 0, row, three_quarters] > 0.90


def test_adaptive_cleanup_is_not_green_specific():
    values = [0.0, 0.15, 0.35, 0.6, 0.8, 1.0]
    image, key, _, start = make_motion_blur_strip(
        [1.0, 0.0, 1.0], [0.0, 0.8, 0.9], values
    )
    _, matte, _, clean = engine.run(
        image, key, 1.0, -0.02, 0.30,
        edge_soft=0.05, defringe=0.07,
    )
    row = 32
    edge = start + values.index(0.8)
    assert 0.65 < matte[0, 0, row, edge] < 0.92
    assert clean[0, 0, row, edge] < 0.08
    assert clean[0, 1, row, edge] > 0.70
    assert clean[0, 2, row, edge] > 0.80


def test_adaptive_cleanup_preserves_solid_highlights_and_core_colour():
    values = [0.0, 0.2, 0.5, 0.8, 1.0]
    image, key, _, start = make_motion_blur_strip(
        [0.0, 1.0, 0.0], [1.0, 1.0, 1.0], values
    )
    _, matte, _, clean = engine.run(
        image, key, 1.0, -0.02, 0.30,
        edge_soft=0.05, defringe=0.07,
    )
    row = 32
    edge = start + values.index(0.8)
    core = start + len(values) + 8
    assert 0.65 < matte[0, 0, row, edge] < 0.92
    assert torch.all(clean[0, :, row, edge] > 0.92)
    assert matte[0, 0, row, core] > 0.99
    assert torch.allclose(clean[0, :, row, core], torch.ones(3), atol=1e-5)


def test_connected_highlight_args_cannot_block_adaptive_motion_cleanup():
    values = [0.0, 0.2, 0.5, 0.8, 1.0]
    image, key, _, start = make_motion_blur_strip(
        [0.0, 1.0, 0.0], [1.0, 1.0, 1.0], values
    )
    protect_highlights = {
        "thr": 0.75,
        "strength": 1.0,
        "soft_width": 0.10,
        "gamma": 1.0,
    }
    _, _, _, clean_without = engine.run(
        image, key, 1.0, -0.02, 0.30,
        edge_soft=0.02, defringe=0.0, ph=protect_highlights,
        adaptive_cleanup=False,
    )
    _, matte, _, clean_with = engine.run(
        image, key, 1.0, -0.02, 0.30,
        edge_soft=0.02, defringe=0.0, ph=protect_highlights,
    )
    row = 32
    edge = start + values.index(0.8)
    old_green_cast = clean_without[0, 1, row, edge] - clean_without[0, 0, row, edge]
    new_green_cast = clean_with[0, 1, row, edge] - clean_with[0, 0, row, edge]
    assert 0.68 < matte[0, 0, row, edge] < 0.90
    assert old_green_cast > 0.015
    assert new_green_cast < 0.01
    assert new_green_cast < old_green_cast * 0.40


def test_adaptive_cleanup_recovers_black_fast_motion_edge():
    values = [0.0, 0.2, 0.5, 0.8, 0.95, 1.0]
    image, key, _, start = make_motion_blur_strip(
        [0.0, 1.0, 0.0], [0.0, 0.0, 0.0], values
    )
    _, matte_before, _, _ = engine.run(
        image, key, 1.0, -0.02, 0.30,
        edge_soft=0.05, defringe=0.07, adaptive_cleanup=False,
    )
    _, matte_after, _, clean = engine.run(
        image, key, 1.0, -0.02, 0.30,
        edge_soft=0.05, defringe=0.07,
    )
    row = 32
    half = start + values.index(0.5)
    dark_edge = start + values.index(0.8)
    assert matte_before[0, 0, row, dark_edge] < 0.05
    assert 0.65 < matte_after[0, 0, row, dark_edge] < 0.88
    assert 0.35 < matte_after[0, 0, row, half] < 0.60
    assert torch.all(clean[0, :, row, dark_edge] < 0.03)


def test_adaptive_cleanup_falls_back_safely_without_foreground_seed():
    key = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32).view(1, 3, 1, 1)
    image = key.repeat(1, 1, 24, 24)
    image[:, :, :, 11] = torch.tensor([0.35, 0.65, 0.0]).view(1, 3, 1)
    image[:, :, :, 12] = torch.tensor([0.70, 0.30, 0.0]).view(1, 3, 1)
    without = engine.run(
        image, key, 1.0, -0.02, 0.30,
        edge_soft=0.0, defringe=0.0, adaptive_cleanup=False,
    )
    with_adaptive = engine.run(
        image, key, 1.0, -0.02, 0.30,
        edge_soft=0.0, defringe=0.0,
    )
    assert torch.allclose(with_adaptive[1], without[1], atol=1e-6)
    assert torch.allclose(with_adaptive[3], without[3], atol=1e-6)


def test_adaptive_cleanup_preserves_opaque_yellow_orange_rim():
    key = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32).view(1, 3, 1, 1)
    image = key.repeat(1, 1, 64, 64)
    image[:, :, :, 16] = torch.tensor([0.50, 0.50, 0.0]).view(1, 3, 1)
    image[:, :, :, 17] = torch.tensor([0.75, 0.25, 0.0]).view(1, 3, 1)
    image[:, :, :, 18] = torch.tensor([0.90, 0.10, 0.0]).view(1, 3, 1)
    image[:, :, :, 19:] = torch.tensor([1.0, 0.0, 0.0]).view(1, 3, 1, 1)
    without = engine.run(
        image, key, 1.0, -0.02, 0.30,
        edge_soft=0.0, defringe=0.0, adaptive_cleanup=False,
    )
    with_adaptive = engine.run(
        image, key, 1.0, -0.02, 0.30,
        edge_soft=0.0, defringe=0.0,
    )
    assert torch.allclose(
        with_adaptive[1][:, :, :, 16:19],
        without[1][:, :, :, 16:19],
        atol=1e-6,
    )
    assert torch.allclose(
        with_adaptive[3][:, :, :, 16:19],
        without[3][:, :, :, 16:19],
        atol=1e-6,
    )


def test_adaptive_cleanup_preserves_unrelated_orange_glow():
    key = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32).view(1, 3, 1, 1)
    orange = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32).view(1, 3, 1)
    red = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32).view(1, 3, 1, 1)
    image = key.repeat(1, 1, 64, 64)
    start = 16
    for offset, alpha in enumerate((0.15, 0.30, 0.50, 0.70)):
        image[:, :, :, start + offset] = (
            orange * alpha + key.view(1, 3, 1) * (1.0 - alpha)
        )
    image[:, :, :, start + 4:] = red
    without = engine.run(
        image, key, 1.0, -0.02, 0.30,
        edge_soft=0.0, defringe=0.0, adaptive_cleanup=False,
    )
    with_adaptive = engine.run(
        image, key, 1.0, -0.02, 0.30,
        edge_soft=0.0, defringe=0.0,
    )
    assert torch.allclose(
        with_adaptive[1][:, :, :, start:start + 4],
        without[1][:, :, :, start:start + 4],
        atol=1e-6,
    )
    assert torch.allclose(
        with_adaptive[3][:, :, :, start:start + 4],
        without[3][:, :, :, start:start + 4],
        atol=1e-6,
    )


def test_adaptive_cleanup_does_not_cross_contaminate_adjacent_colours():
    key = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32).view(1, 3, 1, 1)
    image = key.repeat(1, 1, 64, 64)
    values = [0.0, 0.2, 0.5, 0.8, 1.0]
    start = 16
    for rows, foreground in (
        (slice(0, 32), torch.tensor([1.0, 0.0, 0.0])),
        (slice(32, 64), torch.tensor([0.0, 0.0, 1.0])),
    ):
        for offset, alpha in enumerate(values):
            image[:, :, rows, start + offset] = (
                foreground.view(1, 3, 1) * alpha
                + key.view(1, 3, 1) * (1.0 - alpha)
            )
        image[:, :, rows, start + len(values):] = foreground.view(1, 3, 1, 1)
    _, _, _, clean = engine.run(
        image, key, 1.0, -0.02, 0.30,
        edge_soft=0.05, defringe=0.07,
    )
    edge = start + values.index(0.8)
    assert clean[0, 2, 31, edge] < 0.05
    assert clean[0, 0, 32, edge] < 0.05
    assert clean[0, 0, 16, edge] > 0.90
    assert clean[0, 2, 48, edge] > 0.90


def test_edge_soft_does_not_reduce_foreground_at_frame_border():
    key = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32).view(1, 3, 1, 1)
    image = torch.ones((1, 3, 24, 24), dtype=torch.float32)
    _, matte, _, _ = engine.run(
        image, key, 1.0, -0.02, 0.30,
        edge_soft=0.20, defringe=0.07,
    )
    assert torch.all(matte > 0.99)


def test_edge_soft_does_not_grow_alpha_into_clean_screen():
    values = [0.0, 0.2, 0.5, 0.8, 1.0]
    image, key, _, start = make_motion_blur_strip(
        [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], values
    )
    _, matte, _, _ = engine.run(
        image, key, 1.0, -0.02, 0.30,
        edge_soft=0.20, defringe=0.07,
    )
    assert matte[0, 0, 32, start] < 1e-4
    assert matte[0, 0, 32, start - 1] < 1e-4


def test_adaptive_motion_accuracy_across_keys_and_subjects():
    values = [0.0, 0.2, 0.5, 0.8, 1.0]
    cases = (
        ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0]),
        ([0.0, 1.0, 0.0], [1.0, 1.0, 1.0]),
        ([0.0, 1.0, 0.0], [0.0, 0.0, 0.0]),
        ([1.0, 0.0, 1.0], [0.0, 0.8, 0.9]),
        ([0.0, 0.0, 1.0], [1.0, 0.5, 0.0]),
    )
    expected_alpha = torch.tensor(values, dtype=torch.float32)
    for key_colour, foreground_colour in cases:
        image, key, _, start = make_motion_blur_strip(
            key_colour, foreground_colour, values
        )
        _, matte, _, clean = engine.run(
            image, key, 1.0, -0.02, 0.30,
            edge_soft=0.05, defringe=0.07,
        )
        predicted_alpha = matte[0, 0, 32, start:start + len(values)]
        alpha_mae = (predicted_alpha - expected_alpha).abs().mean()
        predicted_premultiplied = (
            clean[0, :, 32, start:start + len(values)]
            * predicted_alpha.unsqueeze(0)
        )
        expected_premultiplied = (
            torch.tensor(foreground_colour, dtype=torch.float32).view(3, 1)
            * expected_alpha.view(1, -1)
        )
        colour_mae = (
            predicted_premultiplied - expected_premultiplied
        ).abs().mean()
        assert alpha_mae < 0.04
        assert colour_mae < 0.03


def test_keylight_node_uses_adaptive_cleanup_without_args_nodes():
    plugin = load_plugin()
    node = plugin.NODE_CLASS_MAPPINGS["ChromaKeyStudioKeylightV2"]()
    values = [0.0, 0.2, 0.5, 0.8, 1.0]
    image, _, _, start = make_motion_blur_strip(
        [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], values
    )
    _, mask, _, rgba = node.apply(
        image.permute(0, 2, 3, 1),
        "manual", "#00FF00", "alpha", "#000000",
        1.0, -0.02, 0.30,
        shadow_recovery=0.85, edge_soft=0.05,
        defringe=0.07, shrink_expand=0.0,
    )
    edge = start + values.index(0.8)
    assert 0.68 < mask[0, 32, edge] < 0.90
    assert rgba[0, 32, edge, 0] > 0.90
    assert rgba[0, 32, edge, 1] < 0.08


def test_keylight_node_keeps_adaptive_cleanup_with_legacy_edge_and_ph_args():
    plugin = load_plugin()
    node = plugin.NODE_CLASS_MAPPINGS["ChromaKeyStudioKeylightV2"]()
    values = [0.0, 0.2, 0.5, 0.8, 1.0]
    image, _, _, start = make_motion_blur_strip(
        [0.0, 1.0, 0.0], [1.0, 1.0, 1.0], values
    )
    _, mask, _, rgba = node.apply(
        image.permute(0, 2, 3, 1),
        "manual", "#00FF00", "alpha", "#000000",
        1.0, -0.02, 0.30,
        edge_args={
            "shrink_expand": 0.0,
            "edge_soft": 0.02,
            "defringe": 0.0,
        },
        ph_args={
            "thr": 0.75,
            "strength": 1.0,
            "soft_width": 0.10,
            "gamma": 1.0,
        },
        shadow_recovery=0.85, edge_soft=0.05,
        defringe=0.07, shrink_expand=0.0,
    )
    edge = start + values.index(0.8)
    assert 0.68 < mask[0, 32, edge] < 0.90
    assert rgba[0, 32, edge, 1] - rgba[0, 32, edge, 0] < 0.01


def test_adaptive_cleanup_handles_wide_motion_blur_at_production_resolution():
    size = 1024
    values = torch.linspace(0.0, 1.0, 61)
    key = torch.tensor([0.0, 1.0, 0.0]).view(1, 3, 1, 1)
    red = torch.tensor([1.0, 0.0, 0.0]).view(1, 3, 1)
    blue = torch.tensor([0.0, 0.0, 1.0]).view(1, 3, 1)
    image = key.repeat(1, 1, size, size)
    start = 300

    # Three thin moving parts: the 10px core specifically guards the knife/ring
    # case that cannot contain a seed under a large symmetric exclusion radius.
    thin_parts = ((160, 10), (360, 30), (560, 60))
    for centre_y, core_width in thin_parts:
        rows = slice(centre_y - 32, centre_y + 32)
        for offset, alpha in enumerate(values):
            image[:, :, rows, start + offset] = (
                red * alpha + key.view(1, 3, 1) * (1.0 - alpha)
            )
        core_start = start + len(values)
        image[:, :, rows, core_start:core_start + core_width] = red.view(1, 3, 1, 1)
        for offset, alpha in enumerate(reversed(values)):
            image[:, :, rows, core_start + core_width + offset] = (
                red * alpha + key.view(1, 3, 1) * (1.0 - alpha)
            )

    # Red and blue parts meet directly; axis-aware sampling must not average the
    # two foreground colours into purple near their shared boundary.
    for rows, foreground in (
        (slice(720, 820), red),
        (slice(820, 920), blue),
    ):
        for offset, alpha in enumerate(values):
            image[:, :, rows, start + offset] = (
                foreground * alpha + key.view(1, 3, 1) * (1.0 - alpha)
            )
        image[:, :, rows, start + len(values):] = foreground.view(1, 3, 1, 1)

    _, matte, _, clean = engine.run(
        image, key, 1.0, -0.02, 0.30,
        edge_soft=0.05, defringe=0.07,
    )
    expected = values.to(dtype=torch.float32)
    three_quarters = start + int(round(0.75 * (len(values) - 1)))
    for centre_y, _ in thin_parts:
        predicted = matte[0, 0, centre_y, start:start + len(values)]
        assert (predicted - expected).abs().mean() < 0.03
        assert clean[0, 1, centre_y, three_quarters] < 0.05

    for row, foreground_channel, cross_channel in (
        (819, 0, 2),
        (820, 2, 0),
    ):
        predicted = matte[0, 0, row, start:start + len(values)]
        assert (predicted - expected).abs().mean() < 0.03
        assert clean[0, foreground_channel, row, three_quarters] > 0.90
        assert clean[0, cross_channel, row, three_quarters] < 0.05
    assert clean[0, 1, 819, three_quarters] < 0.05
    assert clean[0, 1, 820, three_quarters] < 0.05


def test_adaptive_cleanup_tolerates_small_generated_noise():
    values = torch.linspace(0.0, 1.0, 21).tolist()
    image, key, _, start = make_motion_blur_strip(
        [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], values, size=256
    )
    generator = torch.Generator().manual_seed(7)
    noise = torch.randn(image.shape, generator=generator) * 0.008
    image = (image + noise).clamp(0.0, 1.0)
    _, matte, _, clean = engine.run(
        image, key, 1.0, -0.02, 0.30,
        edge_soft=0.05, defringe=0.07,
    )
    predicted = matte[0, 0, 128, start:start + len(values)]
    expected = torch.tensor(values, dtype=torch.float32)
    assert (predicted - expected).abs().mean() < 0.04
    assert clean[0, 1, 128, start + 10] < 0.08


def test_adaptive_cleanup_chunking_matches_unchunked_batch():
    values = [0.0, 0.2, 0.5, 0.8, 1.0]
    frames = []
    keys = []
    for foreground in (
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.8, 0.9],
    ):
        image, key, _, _ = make_motion_blur_strip(
            [0.0, 1.0, 0.0], foreground, values
        )
        frames.append(image)
        keys.append(key)
    batch = torch.cat(frames, dim=0)
    key_batch = torch.cat(keys, dim=0)
    matte, _, _, _ = engine.compute_matte(
        batch, key_batch, 1.0, -0.02, 0.30, shadow_recovery=0.85
    )
    unchunked = engine._adaptive_motion_cleanup(
        batch, key_batch, matte, max_working_pixels=None
    )
    one_frame_chunks = engine._adaptive_motion_cleanup(
        batch, key_batch, matte, max_working_pixels=64 * 64
    )
    assert torch.allclose(one_frame_chunks[0], unchunked[0], atol=1e-6)
    assert torch.allclose(one_frame_chunks[1], unchunked[1], atol=1e-6)


def test_guided_sampling_supports_two_channel_hues_and_video_drift():
    plugin = load_plugin()
    node = plugin.NODE_CLASS_MAPPINGS["ChromaKeyStudioKeylightV2"]()
    frames = torch.zeros((3, 32, 32, 3), dtype=torch.float32)
    frames[0] = torch.tensor([0.80, 0.05, 1.00])
    frames[1] = torch.tensor([0.95, 0.03, 0.78])
    frames[2] = torch.tensor([0.70, 0.02, 0.98])
    frames[:, 8:24, 8:24] = torch.tensor([0.50, 0.50, 0.50])
    anchor = torch.tensor([0.75, 0.0, 1.0]).view(1, 3, 1, 1).repeat(3, 1, 1, 1)
    refined = node._guided_key_from_border(frames, anchor)
    assert refined.shape == (3, 3, 1, 1)
    output = node.apply(
        frames, "guided", "#BF00FF", "alpha", "#000000",
        1.0, -0.02, 0.30,
        shadow_recovery=0.85, edge_soft=0.0, defringe=0.0, shrink_expand=0.0,
    )
    mask = output[1]
    assert torch.all(mask[:, 0, 0] < 0.01)
    assert torch.all(mask[:, 16, 16] > 0.98)


def test_plugin_ids_are_v2_isolated_and_args_are_legacy_compatible():
    plugin = load_plugin()
    assert set(plugin.NODE_CLASS_MAPPINGS) == V2_IDS
    assert set(plugin.NODE_DISPLAY_NAME_MAPPINGS) == V2_IDS
    assert V2_IDS.isdisjoint(LEGACY_IDS)
    assert plugin.PYTHON_NAMESPACE == "ChromaKeyStudioV2"
    node_class = plugin.NODE_CLASS_MAPPINGS["ChromaKeyStudioKeylightV2"]
    schema = node_class.INPUT_TYPES()
    assert list(schema["required"])[:8] == [
        "image", "key_mode", "key_color", "background_mode", "bg_color",
        "tolerance", "clip_black", "clip_white",
    ]
    assert schema["required"]["key_color"][0] == "CHROMA_STUDIO_V2_COLOR"
    assert schema["required"]["bg_color"][0] == "CHROMA_STUDIO_V2_COLOR"
    assert list(schema["optional"]) == [
        "sampler_args", "edge_args", "spill_algo_args", "ph_args", "mm_args",
    ]
    assert [schema["optional"][name][0] for name in schema["optional"]] == [
        "KEY_SAMPLER_ARGS",
        "KEY_EDGE_ARGS",
        "KEY_SPILL_ALGO_ARGS",
        "KEY_PH_ARGS",
        "KEY_MM_ARGS",
    ]
    assert plugin.NODE_CLASS_MAPPINGS["ChromaKeyStudioSamplerArgsV2"].RETURN_TYPES == ("KEY_SAMPLER_ARGS",)
    assert plugin.NODE_CLASS_MAPPINGS["ChromaKeyStudioEdgeArgsV2"].RETURN_TYPES == ("KEY_EDGE_ARGS",)
    assert plugin.NODE_CLASS_MAPPINGS["ChromaKeyStudioSpillArgsV2"].RETURN_TYPES == ("KEY_SPILL_ALGO_ARGS",)
    assert plugin.NODE_CLASS_MAPPINGS["ChromaKeyStudioProtectHighlightsArgsV2"].RETURN_TYPES == ("KEY_PH_ARGS",)
    assert plugin.NODE_CLASS_MAPPINGS["ChromaKeyStudioMatteMathArgsV2"].RETURN_TYPES == ("KEY_MM_ARGS",)
    assert node_class.RETURN_TYPES == ("IMAGE", "MASK", "IMAGE", "IMAGE")
    assert all("V2" in name for name in plugin.NODE_DISPLAY_NAME_MAPPINGS.values())
    assert all(cls.CATEGORY == "Chroma Key Studio V2" for cls in plugin.NODE_CLASS_MAPPINGS.values())
    smart_class = plugin.NODE_CLASS_MAPPINGS["ChromaKeyStudioSmartBackgroundV2"]
    assert smart_class.RETURN_TYPES[1] == schema["required"]["key_color"][0]
    for value in ("#FF0000", "#00FF00", "#0000FF", "#00FFFF", "#FFFF00", "#FF00FF", "#BF00FF"):
        parsed = plugin.core_helpers.to_color3(value)
        assert parsed.shape == (1, 3, 1, 1)


def test_v2_load_preserves_legacy_private_modules():
    names = (
        "KeylightChromaKeyHub.core.engine",
        "KeylightChromaKeyHub.core.helpers",
        "KeylightChromaKeyHub.nodes.core_hub",
    )
    previous = {name: sys.modules.get(name) for name in names}
    sentinels = {name: types.ModuleType(name) for name in names}
    try:
        sys.modules.update(sentinels)
        load_plugin()
        for name, sentinel in sentinels.items():
            assert sys.modules[name] is sentinel
        assert "ChromaKeyStudioV2.core.engine" in sys.modules
        assert "ChromaKeyStudioV2.nodes.core_hub" in sys.modules
    finally:
        for name, original in previous.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def test_frontend_extension_and_widget_type_are_v2_isolated():
    source = (ROOT / "web" / "colorWidget.js").read_text(encoding="utf-8")
    assert 'name: "ChromaKeyStudioV2.colorWidget"' in source
    assert 'name: "AILab.colorWidget"' not in source
    assert "CHROMA_STUDIO_V2_COLOR" in source


def test_rgba_uses_clean_foreground_even_in_colour_background_mode():
    plugin = load_plugin()
    node = plugin.NODE_CLASS_MAPPINGS["ChromaKeyStudioKeylightV2"]()
    image = torch.zeros((1, 8, 8, 3), dtype=torch.float32)
    image[:] = torch.tensor([0.75, 0.0, 1.0])
    image[:, 2:6, 2:6] = torch.tensor([0.4, 0.4, 0.4])
    image_out, mask, _, rgba = node.apply(
        image, "manual", "#BF00FF", "color", "#FFFFFF",
        1.0, -0.02, 0.30,
        shadow_recovery=0.85, edge_soft=0.0, defringe=0.0, shrink_expand=0.0,
    )
    assert torch.all(image_out[:, 0, 0] > 0.99)
    assert mask[0, 0, 0] < 0.01
    assert rgba[0, 0, 0, 3] < 0.01
    assert not torch.all(rgba[0, 0, 0, :3] > 0.99)


class KeylightRegressionTests(unittest.TestCase):
    def test_all_three_primary_screens_key_out(self):
        test_all_three_primary_screens_key_out()

    def test_black_subject_is_preserved_for_every_primary(self):
        test_black_subject_is_preserved_for_every_primary()

    def test_cyan_subject_is_preserved_on_red_screen(self):
        test_cyan_subject_is_preserved_on_red_screen()

    def test_dark_coloured_screen_recovery_improves_matte(self):
        test_dark_coloured_screen_recovery_improves_matte()

    def test_hybrid_despill_handles_whole_frame(self):
        test_hybrid_despill_handles_whole_frame()

    def test_arbitrary_hue_screens_and_shadows_key_out(self):
        test_arbitrary_hue_screens_and_shadows_key_out()

    def test_neutral_black_gray_white_and_metal_are_preserved(self):
        test_neutral_black_gray_white_and_metal_are_preserved_on_intermediate_keys()

    def test_magenta_key_preserves_subject_colours(self):
        test_magenta_key_preserves_red_blue_and_cyan_subject_colours()

    def test_full_vector_despill_does_not_change_neutral_gray(self):
        test_full_vector_despill_does_not_change_neutral_gray()

    def test_default_adaptive_cleanup_recovers_fast_motion_edge(self):
        test_default_adaptive_cleanup_recovers_fast_motion_edge()

    def test_adaptive_cleanup_is_not_green_specific(self):
        test_adaptive_cleanup_is_not_green_specific()

    def test_adaptive_cleanup_preserves_highlights_and_core(self):
        test_adaptive_cleanup_preserves_solid_highlights_and_core_colour()

    def test_connected_highlight_args_cannot_block_adaptive_cleanup(self):
        test_connected_highlight_args_cannot_block_adaptive_motion_cleanup()

    def test_adaptive_cleanup_recovers_black_motion_edge(self):
        test_adaptive_cleanup_recovers_black_fast_motion_edge()

    def test_adaptive_cleanup_has_safe_no_seed_fallback(self):
        test_adaptive_cleanup_falls_back_safely_without_foreground_seed()

    def test_adaptive_cleanup_preserves_opaque_colour_rim(self):
        test_adaptive_cleanup_preserves_opaque_yellow_orange_rim()

    def test_adaptive_cleanup_preserves_unrelated_glow(self):
        test_adaptive_cleanup_preserves_unrelated_orange_glow()

    def test_adaptive_cleanup_keeps_adjacent_colours_separate(self):
        test_adaptive_cleanup_does_not_cross_contaminate_adjacent_colours()

    def test_edge_soft_preserves_foreground_at_frame_border(self):
        test_edge_soft_does_not_reduce_foreground_at_frame_border()

    def test_edge_soft_does_not_grow_alpha_into_clean_screen(self):
        test_edge_soft_does_not_grow_alpha_into_clean_screen()

    def test_adaptive_motion_accuracy_across_keys_and_subjects(self):
        test_adaptive_motion_accuracy_across_keys_and_subjects()

    def test_keylight_node_defaults_to_adaptive_cleanup(self):
        test_keylight_node_uses_adaptive_cleanup_without_args_nodes()

    def test_keylight_node_accepts_legacy_edge_and_ph_args(self):
        test_keylight_node_keeps_adaptive_cleanup_with_legacy_edge_and_ph_args()

    def test_adaptive_cleanup_handles_wide_production_motion_blur(self):
        test_adaptive_cleanup_handles_wide_motion_blur_at_production_resolution()

    def test_adaptive_cleanup_handles_generated_noise(self):
        test_adaptive_cleanup_tolerates_small_generated_noise()

    def test_adaptive_cleanup_batch_chunking_is_exact(self):
        test_adaptive_cleanup_chunking_matches_unchunked_batch()

    def test_guided_sampling_supports_video_drift(self):
        test_guided_sampling_supports_two_channel_hues_and_video_drift()

    def test_plugin_ids_are_v2_isolated_and_args_are_legacy_compatible(self):
        test_plugin_ids_are_v2_isolated_and_args_are_legacy_compatible()

    def test_v2_load_preserves_legacy_private_modules(self):
        test_v2_load_preserves_legacy_private_modules()

    def test_frontend_extension_and_widget_type_are_v2_isolated(self):
        test_frontend_extension_and_widget_type_are_v2_isolated()

    def test_rgba_uses_clean_foreground_in_colour_mode(self):
        test_rgba_uses_clean_foreground_even_in_colour_background_mode()


if __name__ == "__main__":
    unittest.main()

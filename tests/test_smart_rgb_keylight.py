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


def test_plugin_mappings_and_socket_types_are_v2_isolated():
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
        "CHROMA_STUDIO_V2_SAMPLER_ARGS",
        "CHROMA_STUDIO_V2_EDGE_ARGS",
        "CHROMA_STUDIO_V2_SPILL_ALGO_ARGS",
        "CHROMA_STUDIO_V2_PH_ARGS",
        "CHROMA_STUDIO_V2_MM_ARGS",
    ]
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

    def test_guided_sampling_supports_video_drift(self):
        test_guided_sampling_supports_two_channel_hues_and_video_drift()

    def test_plugin_mappings_and_socket_types_are_v2_isolated(self):
        test_plugin_mappings_and_socket_types_are_v2_isolated()

    def test_v2_load_preserves_legacy_private_modules(self):
        test_v2_load_preserves_legacy_private_modules()

    def test_frontend_extension_and_widget_type_are_v2_isolated(self):
        test_frontend_extension_and_widget_type_are_v2_isolated()

    def test_rgba_uses_clean_foreground_in_colour_mode(self):
        test_rgba_uses_clean_foreground_even_in_colour_background_mode()


if __name__ == "__main__":
    unittest.main()

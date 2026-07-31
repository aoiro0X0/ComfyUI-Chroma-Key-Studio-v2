import importlib.util
from pathlib import Path
import re
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("chroma_studio_smart_background", ROOT / "smart_background.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def black_canvas(colour, size=64):
    image = np.zeros((1, size, size, 3), dtype=np.float32)
    image[:, size // 4: 3 * size // 4, size // 4: 3 * size // 4] = colour
    return torch.from_numpy(image)


def complex_subject():
    image = np.zeros((1, 128, 128, 3), dtype=np.float32)
    image[:, 20:112, 18:110] = 0.25
    image[:, 35:70, 25:105] = [0.05, 0.55, 0.80]  # cyan/blue glass
    image[:, 72:100, 30:100] = [0.05, 0.35, 0.25]  # green smoke
    image[:, 25:31, 55:75] = [1.00, 0.05, 0.03]    # small red lamp
    image[:, 100:104, 45:85] = [0.80, 0.65, 0.05] # yellow accent
    return torch.from_numpy(image)


class SmartBackgroundTests(unittest.TestCase):
    def setUp(self):
        self.node = MODULE.AutoChromaSmartBackground()

    def test_primary_choices_for_simple_subjects(self):
        cases = [
            ([0.0, 0.8, 0.8], "#FF0000"),
            ([0.0, 0.8, 0.0], "#0000FF"),
            ([0.0, 0.0, 0.8], "#00FF00"),
            ([0.8, 0.0, 0.0], "#00FF00"),
            ([0.35, 0.35, 0.35], "#00FF00"),
        ]
        for colour, expected in cases:
            with self.subTest(colour=colour):
                _, selected, _ = self.node.process(black_canvas(colour))
                self.assertEqual(selected, expected)

    def test_complex_subject_uses_purple_fallback(self):
        background, selected, report = self.node.process(
            complex_subject(), disable_yellow_bg=True, disable_cyan_bg=True
        )
        self.assertEqual(selected, "#BF00FF")
        self.assertIn("间色安全兜底", report)
        self.assertTrue(torch.allclose(background[0, 0, 0], torch.tensor([0.75, 0.0, 1.0]), atol=0.002))

    def test_small_salient_red_lamp_prevents_red_key(self):
        _, selected, _ = self.node.process(complex_subject())
        self.assertNotEqual(selected, "#FF0000")

    def test_sparse_rgb_noise_does_not_trigger_fallback(self):
        image = np.zeros((1, 128, 128, 3), dtype=np.float32)
        image[:, 16:112, 16:112] = 0.35
        for index in range(10):
            image[0, 20 + index * 7, 24 + index * 5] = [1.0, 0.0, 0.0]
            image[0, 22 + index * 7, 26 + index * 5] = [0.0, 1.0, 0.0]
            image[0, 24 + index * 7, 28 + index * 5] = [0.0, 0.0, 1.0]
        _, selected, report = self.node.process(torch.from_numpy(image))
        self.assertEqual(selected, "#00FF00")
        self.assertIn("三原色优先", report)

    def test_disabling_all_primaries_uses_a_fallback_instead_of_ignoring_flags(self):
        _, selected, report = self.node.process(
            black_canvas([0.35, 0.35, 0.35]),
            disable_red_bg=True,
            disable_green_bg=True,
            disable_blue_bg=True,
        )
        self.assertNotIn(selected, {"#FF0000", "#00FF00", "#0000FF"})
        self.assertIn("间色安全兜底", report)

    def test_batch_returns_one_hex_and_one_shared_key_colour(self):
        first = black_canvas([0.0, 0.9, 0.9])
        second = black_canvas([0.9, 0.0, 0.0])
        batch = torch.cat([first, second], dim=0)
        background, selected, _ = self.node.process(batch)
        self.assertRegex(selected, r"^#[0-9A-F]{6}$")
        self.assertTrue(torch.equal(background[0], background[1]))

    def test_edge_connected_black_is_removed_but_interior_black_is_retained(self):
        image = np.full((32, 32, 3), 0.3, dtype=np.float32)
        image[:4] = 0.0
        image[-4:] = 0.0
        image[:, :4] = 0.0
        image[:, -4:] = 0.0
        image[12:20, 12:20] = 0.0
        background = MODULE.edge_connected_black_background(image, 0.08)
        self.assertTrue(background[0, 0])
        self.assertFalse(background[16, 16])

    def test_rgba_alpha_is_never_counted_as_colour(self):
        image = np.zeros((1, 48, 48, 4), dtype=np.float32)
        image[:, 12:36, 12:36, :3] = [0.0, 0.8, 0.8]
        image[:, 12:36, 12:36, 3] = 1.0
        _, selected, _ = self.node.process(torch.from_numpy(image))
        self.assertEqual(selected, "#FF0000")

    def test_legacy_widget_prefix_and_outputs_are_stable(self):
        schema = self.node.INPUT_TYPES()
        optional = list(schema["optional"])
        self.assertEqual(
            optional[:10],
            [
                "mask", "saturation_threshold", "value_threshold",
                "presence_threshold", "vivid_threshold", "disable_red_bg",
                "disable_yellow_bg", "disable_green_bg", "disable_cyan_bg",
                "disable_blue_bg",
            ],
        )
        self.assertEqual(self.node.RETURN_TYPES, ("IMAGE", "STRING", "STRING"))
        self.assertEqual(self.node.RETURN_NAMES, ("background_image", "color_hex", "analysis_info"))

    def test_standalone_alias_keeps_its_historical_widget_order(self):
        standalone = MODULE.KeylightSmartBackground()
        self.assertEqual(
            list(standalone.INPUT_TYPES()["optional"]),
            [
                "black_background_threshold", "saturation_threshold",
                "value_threshold", "hue_conflict_radius", "disable_green_bg",
                "disable_blue_bg", "disable_red_bg",
            ],
        )
        _, selected, _ = standalone.process(
            black_canvas([0.0, 0.8, 0.8]), 0.08, 0.08, 0.08, 75.0,
            False, False, False,
        )
        self.assertEqual(selected, "#FF0000")


if __name__ == "__main__":
    unittest.main()

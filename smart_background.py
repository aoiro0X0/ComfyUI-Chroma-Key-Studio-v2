from collections import deque
import colorsys
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


PRIMARY_HUES: Tuple[int, ...] = (120, 240, 0)  # green, blue, red tie priority
CANDIDATE_HUES: Tuple[int, ...] = tuple(range(0, 360, 15))
NAMED_HUES = {
    0: "红",
    30: "橙",
    60: "黄",
    90: "黄绿",
    120: "绿",
    150: "春绿",
    180: "青",
    210: "天蓝",
    240: "蓝",
    270: "紫",
    285: "蓝紫",
    300: "品红",
    330: "玫红",
}
DISABLE_CENTERS = {
    "red": 0,
    "yellow": 60,
    "green": 120,
    "cyan": 180,
    "blue": 240,
}


def rgb_to_hsv(rgb: np.ndarray):
    """Convert an [..., 3] RGB array in [0, 1] to HSV arrays."""
    rgb = np.asarray(rgb, dtype=np.float32)[..., :3]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    delta = maxc - minc

    saturation = np.zeros_like(maxc)
    nonzero_value = maxc > 1e-6
    saturation[nonzero_value] = delta[nonzero_value] / maxc[nonzero_value]

    hue = np.zeros_like(maxc)
    chromatic = delta > 1e-6
    red_max = chromatic & (maxc == r)
    green_max = chromatic & (maxc == g)
    blue_max = chromatic & (maxc == b)
    hue[red_max] = ((g[red_max] - b[red_max]) / (delta[red_max] + 1e-6)) % 6.0
    hue[green_max] = (b[green_max] - r[green_max]) / (delta[green_max] + 1e-6) + 2.0
    hue[blue_max] = (r[blue_max] - g[blue_max]) / (delta[blue_max] + 1e-6) + 4.0
    return (hue * 60.0) % 360.0, saturation, maxc


def circular_distance(values: np.ndarray, hue: float) -> np.ndarray:
    distance = np.abs(values - float(hue))
    return np.minimum(distance, 360.0 - distance)


def hue_to_rgb(hue: int) -> Tuple[int, int, int]:
    rgb = colorsys.hsv_to_rgb((int(hue) % 360) / 360.0, 1.0, 1.0)
    return tuple(int(round(channel * 255.0)) for channel in rgb)


def rgb_to_hex(rgb: Sequence[int]) -> str:
    return f"#{int(rgb[0]):02X}{int(rgb[1]):02X}{int(rgb[2]):02X}"


def edge_connected_black_background(rgb: np.ndarray, threshold: float) -> np.ndarray:
    """Return near-black pixels connected to an image edge.

    Interior black areas remain part of the subject. A perfectly black subject fused
    into a perfectly black background is inherently ambiguous and cannot be recovered
    from RGB values alone.
    """
    near_black = np.max(rgb[..., :3], axis=-1) <= float(threshold)
    height, width = near_black.shape
    background = np.zeros((height, width), dtype=bool)
    queue = deque()

    def enqueue(y: int, x: int):
        if near_black[y, x] and not background[y, x]:
            background[y, x] = True
            queue.append((y, x))

    for x in range(width):
        enqueue(0, x)
        if height > 1:
            enqueue(height - 1, x)
    for y in range(1, height - 1):
        enqueue(y, 0)
        if width > 1:
            enqueue(y, width - 1)

    while queue:
        y, x = queue.popleft()
        if y > 0:
            enqueue(y - 1, x)
        if y + 1 < height:
            enqueue(y + 1, x)
        if x > 0:
            enqueue(y, x - 1)
        if x + 1 < width:
            enqueue(y, x + 1)
    return background


def _top_mean(values: np.ndarray, fraction: float) -> float:
    if values.size == 0:
        return 0.0
    count = max(1, int(math.ceil(values.size * float(fraction))))
    if count >= values.size:
        return float(np.mean(values))
    return float(np.mean(np.partition(values, values.size - count)[-count:]))


def calculate_frame_risks(
    rgb: np.ndarray,
    subject_region: np.ndarray,
    saturation_threshold: float,
    value_threshold: float,
    presence_threshold: float,
    vivid_threshold: float,
    candidate_hues: Iterable[int] = CANDIDATE_HUES,
    hue_radius: float = 75.0,
):
    """Measure global, local and salient-accent conflicts for every key hue."""
    hue, saturation, value = rgb_to_hsv(rgb)
    valid = (
        subject_region
        & (saturation >= float(saturation_threshold))
        & (value >= float(value_threshold))
    )
    valid_count = int(np.sum(valid))
    hues = tuple(int(item) for item in candidate_hues)
    empty = {item: 0.0 for item in hues}
    subject_pixels = int(np.sum(subject_region))
    # Ignore isolated colour noise. Evidence must be both absolutely meaningful
    # and non-trivial relative to the retained subject region.
    minimum_evidence = max(32, int(math.ceil(subject_pixels * 0.0005)))
    if valid_count < minimum_evidence:
        return empty.copy(), empty.copy(), empty.copy(), empty.copy(), valid_count

    hue_valid = hue[valid]
    saturation_valid = saturation[valid]
    value_valid = value[valid]
    global_weights = np.power(saturation_valid, 1.5) * np.sqrt(value_valid)
    total_weight = float(np.sum(global_weights))
    if total_weight <= 1e-8:
        return empty.copy(), empty.copy(), empty.copy(), empty.copy(), valid_count

    vivid_mask = saturation_valid >= float(vivid_threshold)
    accent_weights = np.power(saturation_valid, 2.5) * np.power(value_valid, 1.2)
    accent_fraction = max(0.005, min(0.05, float(presence_threshold)))

    global_scores: Dict[int, float] = {}
    accent_scores: Dict[int, float] = {}
    for candidate in hues:
        proximity = np.clip(
            1.0 - circular_distance(hue_valid, candidate) / float(hue_radius),
            0.0,
            1.0,
        )
        global_scores[candidate] = float(
            np.sum(proximity * global_weights) / total_weight
        )
        accent_values = proximity * accent_weights * vivid_mask
        accent_scores[candidate] = _top_mean(accent_values, accent_fraction)

    # A 16x16 spatial grid catches a small but important lamp, logo or highlight
    # that global pixel coverage would otherwise hide.
    height, width = subject_region.shape
    y_edges = np.linspace(0, height, min(16, height) + 1, dtype=np.int32)
    x_edges = np.linspace(0, width, min(16, width) + 1, dtype=np.int32)
    minimum_tile_mass = total_weight * max(
        0.0005, min(0.01, float(presence_threshold) * 0.10)
    )
    minimum_tile_pixels = max(2, int(math.ceil(subject_pixels * 0.00002)))
    tile_scores: Dict[int, List[float]] = {item: [] for item in hues}
    for yi in range(len(y_edges) - 1):
        y0, y1 = int(y_edges[yi]), int(y_edges[yi + 1])
        for xi in range(len(x_edges) - 1):
            x0, x1 = int(x_edges[xi]), int(x_edges[xi + 1])
            tile_valid = valid[y0:y1, x0:x1]
            if int(np.sum(tile_valid)) < minimum_tile_pixels:
                continue
            tile_hue = hue[y0:y1, x0:x1][tile_valid]
            tile_saturation = saturation[y0:y1, x0:x1][tile_valid]
            tile_value = value[y0:y1, x0:x1][tile_valid]
            tile_weight = np.power(tile_saturation, 1.5) * np.sqrt(tile_value)
            tile_mass = float(np.sum(tile_weight))
            if tile_mass < minimum_tile_mass:
                continue
            for candidate in hues:
                proximity = np.clip(
                    1.0 - circular_distance(tile_hue, candidate) / float(hue_radius),
                    0.0,
                    1.0,
                )
                tile_scores[candidate].append(
                    float(np.sum(proximity * tile_weight) / max(tile_mass, 1e-8))
                )

    local_scores = {
        candidate: _top_mean(np.asarray(tile_scores[candidate], dtype=np.float32), 0.02)
        for candidate in hues
    }
    combined_scores = {
        candidate: (
            0.48 * global_scores[candidate]
            + 0.32 * local_scores[candidate]
            + 0.20 * accent_scores[candidate]
        )
        for candidate in hues
    }
    return global_scores, local_scores, accent_scores, combined_scores, valid_count


def aggregate_frame_scores(frame_scores: List[Dict[int, float]]) -> Dict[int, float]:
    """Use one stable key hue for the complete image/video batch."""
    if not frame_scores:
        return {hue: 0.0 for hue in CANDIDATE_HUES}
    result = {}
    for hue in CANDIDATE_HUES:
        values = np.asarray([scores[hue] for scores in frame_scores], dtype=np.float32)
        result[hue] = float(0.70 * np.mean(values) + 0.30 * np.max(values))
    return result


def _is_disabled(hue: int, disabled_centers: Sequence[int]) -> bool:
    return any(
        min(abs(int(hue) - center), 360 - abs(int(hue) - center)) <= 15
        for center in disabled_centers
    )


def select_key_hue(
    risks: Dict[int, float],
    disabled_centers: Sequence[int] = (),
    primary_safe_threshold: float = 0.12,
    fallback_improvement: float = 0.03,
):
    """Prefer RGB primaries; use a 15-degree fallback only when materially safer."""
    primary = [hue for hue in PRIMARY_HUES if not _is_disabled(hue, disabled_centers)]
    fallback = [
        hue
        for hue in CANDIDATE_HUES
        if hue not in PRIMARY_HUES and not _is_disabled(hue, disabled_centers)
    ]
    if not primary:
        if fallback:
            return min(fallback, key=lambda hue: risks.get(hue, 0.0)), "fallback"
        primary = list(PRIMARY_HUES)  # only possible if every hue was disabled
    best_primary = min(primary, key=lambda hue: risks.get(hue, 0.0))
    best_primary_risk = float(risks.get(best_primary, 0.0))
    if not fallback or best_primary_risk <= float(primary_safe_threshold):
        return best_primary, "primary"
    best_fallback = min(fallback, key=lambda hue: risks.get(hue, 0.0))
    if float(risks.get(best_fallback, 0.0)) + float(fallback_improvement) < best_primary_risk:
        return best_fallback, "fallback"
    return best_primary, "primary"


def _mask_for_frame(mask_array: Optional[np.ndarray], index: int, shape: Tuple[int, int]):
    if mask_array is None:
        return None
    item = mask_array[index if mask_array.shape[0] > 1 else 0]
    item = np.squeeze(item)
    if item.shape != shape:
        raise ValueError(f"Mask shape {item.shape} does not match image shape {shape}")
    return item > 0.5


class AutoChromaSmartBackground:
    """Select a Keylight-safe solid background while retaining legacy workflow IDs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image": ("IMAGE",)},
            "optional": {
                # Optional keeps the old socket name/link working while allowing a plain
                # black-background IMAGE to be connected by itself.
                "mask": (
                    "MASK",
                    {"tooltip": "可选；不连接时自动排除画面边缘连通的黑背景"},
                ),
                # Keep these nine legacy widgets in their exact historical order.
                "saturation_threshold": (
                    "FLOAT",
                    {"default": 0.08, "min": 0.01, "max": 0.50, "step": 0.01},
                ),
                "value_threshold": (
                    "FLOAT",
                    {"default": 0.08, "min": 0.01, "max": 0.50, "step": 0.01},
                ),
                "presence_threshold": (
                    "FLOAT",
                    {"default": 0.005, "min": 0.001, "max": 0.10, "step": 0.001},
                ),
                "vivid_threshold": (
                    "FLOAT",
                    {"default": 0.50, "min": 0.10, "max": 1.00, "step": 0.05},
                ),
                "disable_red_bg": ("BOOLEAN", {"default": False}),
                "disable_yellow_bg": ("BOOLEAN", {"default": False}),
                "disable_green_bg": ("BOOLEAN", {"default": False}),
                "disable_cyan_bg": ("BOOLEAN", {"default": False}),
                "disable_blue_bg": ("BOOLEAN", {"default": False}),
                # New controls are append-only so existing widgets_values remain aligned.
                "black_background_threshold": (
                    "FLOAT",
                    {
                        "default": 0.08,
                        "min": 0.0,
                        "max": 0.50,
                        "step": 0.01,
                        "tooltip": "无 Mask 时，从边缘排除该阈值以下的黑色/近黑背景",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("background_image", "color_hex", "analysis_info")
    FUNCTION = "process"
    CATEGORY = "Chroma Key Studio"

    def process(
        self,
        image,
        mask=None,
        saturation_threshold: float = 0.08,
        value_threshold: float = 0.08,
        presence_threshold: float = 0.005,
        vivid_threshold: float = 0.50,
        disable_red_bg: bool = False,
        disable_yellow_bg: bool = False,
        disable_green_bg: bool = False,
        disable_cyan_bg: bool = False,
        disable_blue_bg: bool = False,
        black_background_threshold: float = 0.08,
        _hue_conflict_radius: float = 75.0,
    ):
        if not torch.is_tensor(image):
            image = torch.from_numpy(np.asarray(image))
        if image.ndim != 4 or image.shape[-1] < 3:
            raise ValueError(f"Expected IMAGE shape [B,H,W,C>=3], got {tuple(image.shape)}")
        if mask is not None and not torch.is_tensor(mask):
            mask = torch.from_numpy(np.asarray(mask))

        image_array = image.detach().cpu().numpy().astype(np.float32)
        mask_array = None if mask is None else mask.detach().cpu().numpy().astype(np.float32)
        if mask_array is not None and mask_array.ndim == 2:
            mask_array = mask_array[None, ...]

        disabled_names = []
        for name, flag in (
            ("red", disable_red_bg),
            ("yellow", disable_yellow_bg),
            ("green", disable_green_bg),
            ("cyan", disable_cyan_bg),
            ("blue", disable_blue_bg),
        ):
            if flag:
                disabled_names.append(name)
        disabled_centers = [DISABLE_CENTERS[name] for name in disabled_names]

        global_frames = []
        local_frames = []
        accent_frames = []
        combined_frames = []
        frame_details = []
        for index in range(image_array.shape[0]):
            item = image_array[index]
            rgb = item[..., :3]
            explicit = _mask_for_frame(mask_array, index, rgb.shape[:2])
            if explicit is not None and np.any(explicit):
                subject_region = explicit
                selection_mode = "mask"
                excluded_count = int(subject_region.size - np.sum(subject_region))
            elif item.shape[-1] >= 4 and np.any(item[..., 3] > 0.5):
                subject_region = item[..., 3] > 0.5
                selection_mode = "alpha"
                excluded_count = int(subject_region.size - np.sum(subject_region))
            else:
                black_background = edge_connected_black_background(
                    rgb, float(black_background_threshold)
                )
                subject_region = ~black_background
                selection_mode = "black-edge"
                excluded_count = int(np.sum(black_background))

            global_risk, local_risk, accent_risk, combined_risk, valid_count = (
                calculate_frame_risks(
                    rgb,
                    subject_region,
                    float(saturation_threshold),
                    float(value_threshold),
                    float(presence_threshold),
                    float(vivid_threshold),
                    hue_radius=float(_hue_conflict_radius),
                )
            )
            global_frames.append(global_risk)
            local_frames.append(local_risk)
            accent_frames.append(accent_risk)
            combined_frames.append(combined_risk)
            frame_details.append(
                (selection_mode, int(np.sum(subject_region)), excluded_count, valid_count)
            )

        global_scores = aggregate_frame_scores(global_frames)
        local_scores = aggregate_frame_scores(local_frames)
        accent_scores = aggregate_frame_scores(accent_frames)
        combined_scores = aggregate_frame_scores(combined_frames)
        selected_hue, strategy = select_key_hue(combined_scores, disabled_centers)
        selected_rgb = hue_to_rgb(selected_hue)
        selected_hex = rgb_to_hex(selected_rgb)

        batch, height, width = image_array.shape[:3]
        background = np.empty((batch, height, width, 3), dtype=np.float32)
        background[...] = np.asarray(selected_rgb, dtype=np.float32) / 255.0
        background_tensor = torch.from_numpy(background).to(
            device=image.device, dtype=torch.float32
        )

        detail_lines = []
        for index, (mode, subject_count, excluded_count, valid_count) in enumerate(frame_details):
            detail_lines.append(
                f"  帧{index + 1}: 模式={mode}, 主体={subject_count:,}, "
                f"排除背景={excluded_count:,}, 有效彩色={valid_count:,}"
            )
        selected_name = NAMED_HUES.get(selected_hue, f"色相{selected_hue}°")
        disabled_text = ", ".join(disabled_names) if disabled_names else "无"
        report = "\n".join(
            [
                "===== Chroma Key Studio 智能背景 =====",
                f"批次帧数: {batch}（整批统一键色）",
                *detail_lines,
                f"禁用色相: {disabled_text}",
                "三原色风险（全局/局部/显著色/综合）:",
                (
                    f"  绿: {global_scores[120]:.4f}/{local_scores[120]:.4f}/"
                    f"{accent_scores[120]:.4f}/{combined_scores[120]:.4f}"
                ),
                (
                    f"  蓝: {global_scores[240]:.4f}/{local_scores[240]:.4f}/"
                    f"{accent_scores[240]:.4f}/{combined_scores[240]:.4f}"
                ),
                (
                    f"  红: {global_scores[0]:.4f}/{local_scores[0]:.4f}/"
                    f"{accent_scores[0]:.4f}/{combined_scores[0]:.4f}"
                ),
                f"选择策略: {'三原色优先' if strategy == 'primary' else '间色安全兜底'}",
                f"选择背景: {selected_name}，色相 {selected_hue}°，{selected_hex}",
            ]
        )
        return background_tensor, selected_hex, report


class KeylightSmartBackground(AutoChromaSmartBackground):
    """Schema-compatible wrapper for the earlier standalone smart-background node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image": ("IMAGE",)},
            "optional": {
                # Keep the standalone repository's exact historical widget order.
                "black_background_threshold": (
                    "FLOAT",
                    {"default": 0.08, "min": 0.0, "max": 0.50, "step": 0.01},
                ),
                "saturation_threshold": (
                    "FLOAT",
                    {"default": 0.08, "min": 0.01, "max": 0.50, "step": 0.01},
                ),
                "value_threshold": (
                    "FLOAT",
                    {"default": 0.08, "min": 0.01, "max": 0.50, "step": 0.01},
                ),
                "hue_conflict_radius": (
                    "FLOAT",
                    {"default": 75.0, "min": 30.0, "max": 120.0, "step": 5.0},
                ),
                "disable_green_bg": ("BOOLEAN", {"default": False}),
                "disable_blue_bg": ("BOOLEAN", {"default": False}),
                "disable_red_bg": ("BOOLEAN", {"default": False}),
            },
        }

    def process(
        self,
        image,
        black_background_threshold=0.08,
        saturation_threshold=0.08,
        value_threshold=0.08,
        hue_conflict_radius=75.0,
        disable_green_bg=False,
        disable_blue_bg=False,
        disable_red_bg=False,
    ):
        return super().process(
            image=image,
            mask=None,
            saturation_threshold=saturation_threshold,
            value_threshold=value_threshold,
            presence_threshold=0.005,
            vivid_threshold=0.50,
            disable_red_bg=disable_red_bg,
            disable_yellow_bg=False,
            disable_green_bg=disable_green_bg,
            disable_cyan_bg=False,
            disable_blue_bg=disable_blue_bg,
            black_background_threshold=black_background_threshold,
            _hue_conflict_radius=hue_conflict_radius,
        )


NODE_CLASS_MAPPINGS = {
    "AutoChromaSmartBackground": AutoChromaSmartBackground,
    "KeylightSmartBackground": KeylightSmartBackground,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoChromaSmartBackground": "Smart Chroma Background (智能抠像背景 V2.0)",
    "KeylightSmartBackground": "Smart Chroma Background (兼容别名)",
}

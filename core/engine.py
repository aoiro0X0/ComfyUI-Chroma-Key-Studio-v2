import math

import torch
import torch.nn.functional as F


def srgb_to_linear(x):
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    return torch.where(
        x <= 0.0031308,
        x * 12.92,
        1.055 * torch.clamp(x, min=0) ** (1.0 / 2.4) - 0.055,
    )


def _to_nchw(img):
    if img.ndim == 4 and img.shape[1] in (1, 3, 4):
        return img
    if img.ndim == 4 and img.shape[-1] in (1, 3, 4):
        return img.permute(0, 3, 1, 2)
    raise ValueError("Unsupported image tensor shape.")


def _normalize(v, eps=1e-6):
    return v / (torch.linalg.norm(v, dim=1, keepdim=True) + eps)


def _smoothstep(edge0, edge1, x):
    t = ((x - edge0) / (edge1 - edge0 + 1e-6)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _chroma_features(x):
    """Return opponent hue direction, HSV-like saturation and value."""
    maximum = x.max(dim=1, keepdim=True).values
    minimum = x.min(dim=1, keepdim=True).values
    saturation = (maximum - minimum) / (maximum + 1e-6)
    chroma = x - x.mean(dim=1, keepdim=True)
    direction = _normalize(chroma)
    return direction, saturation, maximum, chroma


def _morph_shrink_expand(matte, amount):
    if amount == 0:
        return matte
    radius = int(abs(amount))
    if radius == 0:
        return matte
    if amount > 0:
        return F.max_pool2d(matte, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return -F.max_pool2d(-matte, kernel_size=2 * radius + 1, stride=1, padding=radius)


def build_key_vector(key_rgb):
    """Build a neutral-free full key vector for arbitrary-hue despill."""
    key_linear = srgb_to_linear(key_rgb.clamp(0, 1))
    opponent = key_linear - key_linear.mean(dim=1, keepdim=True)
    return _normalize(opponent)


def compute_matte(
    img_srgb,
    key_rgb,
    tolerance,
    clip_black,
    clip_white,
    use_linear=True,
    shadow_recovery=0.85,
):
    """Compute a hue/chroma-gated matte for RGB primaries and intermediate hues.

    Neutral pixels have zero chroma and therefore cannot be mistaken for a magenta,
    cyan, yellow or purple screen. A value gate independently protects true black.
    """
    x = _to_nchw(img_srgb).float().clamp(0, 1)
    key_srgb = _to_nchw(key_rgb).float().clamp(0, 1)
    if key_srgb.shape[0] == 1 and x.shape[0] > 1:
        key_srgb = key_srgb.repeat(x.shape[0], 1, 1, 1)

    x_hue, x_saturation, x_value, _ = _chroma_features(x)
    key_hue, key_saturation, _, _ = _chroma_features(key_srgb)
    hue_cosine = (x_hue * key_hue).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)

    # Keep the historical direction: larger tolerance protects more foreground.
    strictness = max(0.0, min(1.0, float(tolerance) / 2.0))
    outer_degrees = 55.0 - 35.0 * strictness
    inner_degrees = outer_degrees * 0.35
    hue_match = _smoothstep(
        math.cos(math.radians(outer_degrees)),
        math.cos(math.radians(inner_degrees)),
        hue_cosine,
    )

    saturation_low = (0.12 * key_saturation).clamp(0.06, 0.16)
    saturation_high = (0.35 * key_saturation).clamp(0.18, 0.45)
    saturation_gate = _smoothstep(saturation_low, saturation_high, x_saturation)
    value_gate = _smoothstep(0.015, 0.060, x_value)

    x_work = srgb_to_linear(x) if use_linear else x
    key_work = srgb_to_linear(key_srgb) if use_linear else key_srgb
    ray_key = _normalize(key_work + 1e-6)
    projection = (x_work * ray_key).sum(dim=1, keepdim=True)
    distance = torch.linalg.norm(x_work - projection * ray_key, dim=1, keepdim=True)
    ray_score = projection - float(tolerance) * distance
    clip_width = max(1e-6, float(clip_white) - float(clip_black))
    ray_background = _smoothstep(
        float(clip_black), float(clip_black) + clip_width, ray_score
    )

    recovery = max(0.0, min(1.0, float(shadow_recovery)))
    colour_evidence = (
        hue_match
        * saturation_gate
        * value_gate
        * ((1.0 - recovery) * ray_background + recovery)
    )
    background_probability = _smoothstep(0.08, 0.78, colour_evidence)
    matte = (1.0 - background_probability).clamp(0.0, 1.0)
    return matte, projection, distance, build_key_vector(key_srgb)


def _restore_luminance(before, after):
    weights = before.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
    before_luma = (before * weights).sum(dim=1, keepdim=True)
    after_luma = (after * weights).sum(dim=1, keepdim=True)
    return (after + before_luma - after_luma).clamp(0.0, 1.0)


def _spill_suppress(
    work,
    proj,
    dist,
    k,
    matte,
    desp,
    balance=1.2,
    extra_lowalpha=0.2,
    ph_mask=None,
    ph_strength=0.0,
    mode="hybrid",
    gain=1.0,
):
    """Suppress spill along the complete neutral-free key vector."""
    del proj, dist  # retained in the signature for Python/API compatibility
    chroma = work - work.mean(dim=1, keepdim=True)
    key_projection = (chroma * k).sum(dim=1, keepdim=True)
    orthogonal = torch.linalg.norm(chroma - key_projection * k, dim=1, keepdim=True)
    geometric = (float(gain) * key_projection - float(balance) * orthogonal).clamp(min=0.0)
    screen = (float(gain) * key_projection).clamp(min=0.0)

    if mode == "geometric":
        spill = geometric
    elif mode == "screen":
        spill = screen
    else:
        spill = 0.70 * geometric + 0.30 * screen

    background_weight = (1.0 - matte).clamp(0.0, 1.0)
    edge_weight = (4.0 * matte * (1.0 - matte)).clamp(0.0, 1.0)
    suppress = (
        float(desp) * (background_weight + 0.35 * edge_weight)
        + float(extra_lowalpha) * background_weight.square()
    ).clamp(min=0.0)
    if ph_mask is not None and ph_strength > 0.0:
        suppress = suppress * (1.0 - ph_mask * ph_strength * 0.8).clamp(0.25, 1.0)

    corrected = work - spill * suppress * k
    return _restore_luminance(work, corrected)


def _defringe(work, key_vector, matte, strength):
    if float(strength) <= 0.0:
        return work
    dilated = F.max_pool2d(matte, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-matte, kernel_size=3, stride=1, padding=1)
    edge_band = (dilated - eroded).clamp(0.0, 1.0)
    chroma = work - work.mean(dim=1, keepdim=True)
    projection = (chroma * key_vector).sum(dim=1, keepdim=True).clamp(min=0.0)
    corrected = work - float(strength) * edge_band * projection * key_vector
    return _restore_luminance(work, corrected)


def composite(rgb_srgb, matte, mode="alpha", bg_color=None):
    x = _to_nchw(rgb_srgb).clamp(0, 1)
    alpha = matte.clamp(0, 1)
    if mode == "alpha":
        return x, alpha
    if bg_color is None:
        bg_color = torch.tensor(
            [0.0, 0.0, 0.0], device=x.device, dtype=x.dtype
        ).view(1, 3, 1, 1)
    else:
        bg_color = bg_color.view(1, 3, 1, 1).to(x.device, x.dtype)
    if mode == "color":
        return x * alpha + bg_color * (1.0 - alpha), alpha
    if mode == "soft_color":
        kernel = torch.tensor(
            [[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]],
            device=alpha.device,
            dtype=alpha.dtype,
        ) / 16.0
        alpha_soft = F.conv2d(alpha, kernel, padding=1)
        return x * alpha_soft + bg_color * (1.0 - alpha_soft), alpha_soft
    return x, alpha


def run(
    rgb_srgb,
    key_rgb,
    tolerance,
    clip_black,
    clip_white,
    edge_soft=0.0,
    shrink_expand=0.0,
    defringe=0.0,
    spill_algo=None,
    ph=None,
    matte_math=None,
    background_mode="alpha",
    bg_color=None,
    use_linear=True,
    verbose=False,
    shadow_recovery=0.85,
):
    x = _to_nchw(rgb_srgb).float().clamp(0, 1)
    if verbose:
        print(f"[Chroma Key Studio] batch={x.shape[0]}, mode={background_mode}")
    key = key_rgb.view(x.shape[0], 3, 1, 1) if key_rgb.ndim == 2 else key_rgb
    matte, projection, distance, key_vector = compute_matte(
        x,
        key,
        tolerance,
        clip_black,
        clip_white,
        use_linear=use_linear,
        shadow_recovery=shadow_recovery,
    )

    if edge_soft > 0.0:
        radius = int(max(1, round(float(edge_soft) * 10)))
        size = 2 * radius + 1
        kernel = torch.ones(
            (1, 1, size, size), device=matte.device, dtype=matte.dtype
        ) / float(size * size)
        matte = F.conv2d(matte, kernel, padding=radius)
    if abs(shrink_expand) > 0.0:
        matte = _morph_shrink_expand(matte, shrink_expand)

    # Matte Math must finish before colour cleanup so every stage uses final alpha.
    if matte_math:
        extra = float(matte_math.get("extra_shrink_expand", 0.0))
        feather = float(matte_math.get("feather", 0.0))
        gamma = float(matte_math.get("gamma", 1.0))
        if abs(extra) > 0.0:
            matte = _morph_shrink_expand(matte, extra)
        if feather > 0.0:
            radius = int(max(1, round(feather * 10)))
            size = 2 * radius + 1
            kernel = torch.ones(
                (1, 1, size, size), device=matte.device, dtype=matte.dtype
            ) / float(size * size)
            matte = F.conv2d(matte, kernel, padding=radius)
        if gamma != 1.0:
            matte = matte.clamp(0.0, 1.0) ** gamma
    matte = matte.clamp(0.0, 1.0)

    spill = spill_algo or {}
    diff_gain = float(spill.get("diff_gain", 1.0))
    diff_balance = float(spill.get("diff_balance", 1.0))
    despill = float(spill.get("despill", 2.0))
    extra_lowalpha = float(spill.get("extra_lowalpha", 1.5))
    final_strength = float(spill.get("final_despill_strength", 0.6))
    despill_mode = str(spill.get("despill_mode", "hybrid"))
    # Preserve the historical Args-node authority of algo when it is present.
    legacy_algo = str(spill.get("algo", ""))
    if legacy_algo == "diff":
        despill_mode = "geometric"
    elif legacy_algo == "proj":
        despill_mode = "screen"
    elif legacy_algo == "blend":
        despill_mode = "hybrid"

    ph_mask = None
    ph_strength = 0.0
    if ph:
        ph_threshold = float(ph.get("thr", 0.75))
        ph_width = float(ph.get("soft_width", 0.2))
        ph_gamma = float(ph.get("gamma", 1.0))
        ph_strength = float(ph.get("strength", 0.7))
        luminance = (
            0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3]
        ).clamp(0, 1)
        low = ph_threshold - ph_width * 0.5
        high = ph_threshold + ph_width * 0.5
        ph_mask = _smoothstep(low, high, luminance) ** ph_gamma

    rgb_linear = srgb_to_linear(x)
    despilled = _spill_suppress(
        rgb_linear,
        projection,
        distance,
        key_vector,
        matte,
        despill,
        balance=diff_balance,
        extra_lowalpha=extra_lowalpha,
        ph_mask=ph_mask,
        ph_strength=ph_strength,
        mode=despill_mode,
        gain=diff_gain,
    )
    strength = max(0.0, min(1.5, final_strength))
    out_linear = (rgb_linear + strength * (despilled - rgb_linear)).clamp(0.0, 1.0)
    out_linear = _defringe(out_linear, key_vector, matte, defringe)
    clean_foreground = linear_to_srgb(out_linear).clamp(0.0, 1.0)

    comp, alpha = composite(
        clean_foreground, matte, mode=background_mode, bg_color=bg_color
    )
    mask_image = alpha.repeat(1, 3, 1, 1)
    return comp, alpha, mask_image, clean_foreground

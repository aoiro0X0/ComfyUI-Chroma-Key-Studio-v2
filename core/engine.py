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


def _channel_norm3(v):
    """Fast Euclidean norm for the engine's three-channel colour tensors."""
    return v.square().sum(dim=1, keepdim=True).sqrt()


def _normalize(v, eps=1e-6):
    return v / (_channel_norm3(v) + eps)


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


def _separable_average(x, radius):
    """Fast box average with cost linear in radius instead of kernel area."""
    radius = int(radius)
    if radius <= 0:
        return x
    size = 2 * radius + 1
    horizontal = F.avg_pool2d(
        x, kernel_size=(1, size), stride=1, padding=(0, radius)
    )
    return F.avg_pool2d(
        horizontal, kernel_size=(size, 1), stride=1, padding=(radius, 0)
    )


def _separable_average_replicate(x, radius):
    """Box average that does not invent transparent pixels at frame borders."""
    radius = int(radius)
    if radius <= 0:
        return x
    size = 2 * radius + 1
    horizontal = F.avg_pool2d(
        F.pad(x, (radius, radius, 0, 0), mode="replicate"),
        kernel_size=(1, size),
        stride=1,
    )
    return F.avg_pool2d(
        F.pad(horizontal, (0, 0, radius, radius), mode="replicate"),
        kernel_size=(size, 1),
        stride=1,
    )


def _horizontal_average(x, radius):
    radius = int(radius)
    if radius <= 0:
        return x
    size = 2 * radius + 1
    return F.avg_pool2d(
        x, kernel_size=(1, size), stride=1, padding=(0, radius)
    )


def _vertical_average(x, radius):
    radius = int(radius)
    if radius <= 0:
        return x
    size = 2 * radius + 1
    return F.avg_pool2d(
        x, kernel_size=(size, 1), stride=1, padding=(radius, 0)
    )


def _separable_maximum(x, radius):
    radius = int(radius)
    if radius <= 0:
        return x
    return _vertical_maximum(_horizontal_maximum(x, radius), radius)


def _horizontal_maximum(x, radius):
    radius = int(radius)
    if radius <= 0:
        return x
    size = 2 * radius + 1
    padded = F.pad(
        x, (radius, radius, 0, 0), mode="constant", value=float("-inf")
    )
    # unfold is a stride-only view; amax reduces it without materialising a
    # [N,C,H,W,K] tensor.  This is equivalent to the old 1D max_pool2d call,
    # but avoids its very slow CPU kernel for large motion-search radii.
    return padded.unfold(3, size, 1).amax(dim=-1)


def _vertical_maximum(x, radius):
    radius = int(radius)
    if radius <= 0:
        return x
    size = 2 * radius + 1
    padded = F.pad(
        x, (0, 0, radius, radius), mode="constant", value=float("-inf")
    )
    return padded.unfold(2, size, 1).amax(dim=-1)


def _edge_aware_smooth(matte, radius):
    """Smooth only an existing transition, never grow alpha into clean screen."""
    blurred = _separable_average_replicate(matte, radius)
    transition = (matte > 1e-4) & (matte < 1.0 - 1e-4)
    return torch.where(transition, blurred, matte)


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
        # Highlight protection is for confident foreground, not a licence to
        # preserve a bright key-coloured motion edge.  Strong measured spill
        # progressively disables the protection even when an old PH Args node
        # supplies strength=1.
        foreground_confidence = _smoothstep(0.80, 0.98, matte)
        spill_confidence = spill / (spill + 0.04)
        protection = (
            ph_mask
            * float(ph_strength)
            * 0.8
            * foreground_confidence
            * (1.0 - spill_confidence)
        )
        suppress = suppress * (1.0 - protection).clamp(0.25, 1.0)

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


def _adaptive_motion_cleanup(
    rgb_srgb, key_srgb, matte, max_working_pixels=1_000_000
):
    """Recover key-contaminated motion blur without a per-shot preset.

    A fast edge is a physical mixture of nearby foreground and screen colour.
    Hue-only keyers can mistake that rotated mixture for opaque foreground.  This
    routine finds clean foreground just inside the boundary, tests whether an
    edge pixel lies on the foreground/screen mixing line, and only then repairs
    both its alpha and unpremultiplied RGB.  Unrelated glow, texture and solid
    highlights fail the mixing-line test and are left alone.
    """
    x = _to_nchw(rgb_srgb).float().clamp(0.0, 1.0)
    key = _to_nchw(key_srgb).float().clamp(0.0, 1.0)
    if key.shape[0] == 1 and x.shape[0] > 1:
        key = key.repeat(x.shape[0], 1, 1, 1)
    if min(x.shape[-2:]) < 8:
        return x, matte

    # Video IMAGE batches can contain dozens of 1K frames.  Process a bounded
    # number at a time so temporary colour fields do not scale with clip length.
    frame_pixels = int(x.shape[-2] * x.shape[-1])
    if max_working_pixels is not None:
        frames_per_chunk = max(1, int(max_working_pixels) // frame_pixels)
        if x.shape[0] > frames_per_chunk:
            corrected_rgb = torch.empty_like(x)
            corrected_matte = torch.empty_like(matte)
            for start in range(0, x.shape[0], frames_per_chunk):
                end = min(x.shape[0], start + frames_per_chunk)
                rgb_chunk, matte_chunk = _adaptive_motion_cleanup(
                    x[start:end],
                    key[start:end],
                    matte[start:end],
                    max_working_pixels=None,
                )
                corrected_rgb[start:end] = rgb_chunk
                corrected_matte[start:end] = matte_chunk
            return corrected_rgb, corrected_matte

    # Scale the search with resolution.  Fast generated motion can span dozens
    # of pixels at 1K, so the candidate boundary is wider than a normal antialias.
    boundary_radius = max(2, min(64, int(round(min(x.shape[-2:]) * 0.05))))
    sample_radius = min(64, boundary_radius * 3)

    background_nearby = _separable_maximum(
        (1.0 - matte).clamp(0.0, 1.0), boundary_radius
    )
    # Seed validation is deliberately much narrower than the candidate band.
    # Otherwise a 1K knife blade or ring would need to be over 100px thick before
    # it contained any usable foreground.  A local colour plateau plus the key
    # direction gate rejects the falsely opaque inner half of a wide blur.
    seed_radius = max(1, min(4, int(round(min(x.shape[-2:]) * 0.002))))
    seed_background_nearby = _separable_maximum(
        (1.0 - matte).clamp(0.0, 1.0), seed_radius
    )
    # Use a wider, resolution-scaled plateau check than the matte-isolation
    # radius.  At 1K a 40-60px linear motion ramp can look nearly constant in a
    # 3px window and masquerade as opaque foreground; a 9px window rejects that
    # ramp while still leaving a valid centre seed in a 10px solid core.
    plateau_radius = max(
        1, min(8, int(round(min(x.shape[-2:]) * 0.004)))
    )
    square_plateau = (
        _separable_maximum(x, plateau_radius)
        + _separable_maximum(-x, plateau_radius)
    ).amax(dim=1, keepdim=True) <= 0.06
    horizontal_plateau = (
        _horizontal_maximum(x, plateau_radius)
        + _horizontal_maximum(-x, plateau_radius)
    ).amax(dim=1, keepdim=True) <= 0.06
    vertical_plateau = (
        _vertical_maximum(x, plateau_radius)
        + _vertical_maximum(-x, plateau_radius)
    ).amax(dim=1, keepdim=True) <= 0.06
    foreground_hue, foreground_saturation, _, _ = _chroma_features(x)
    key_hue, _, _, _ = _chroma_features(key)
    key_alignment = (foreground_hue * key_hue).sum(dim=1, keepdim=True)
    seed_colour_is_safe = (
        (foreground_saturation <= 0.04) | (key_alignment <= 0.20)
    )
    foreground_seed_base = (
        (matte >= 0.985)
        & (seed_background_nearby <= 0.02)
        & seed_colour_is_safe
    )
    foreground_seeds = {
        "square": (foreground_seed_base & square_plateau).to(dtype=x.dtype),
        "horizontal": (
            foreground_seed_base & horizontal_plateau
        ).to(dtype=x.dtype),
        "vertical": (
            foreground_seed_base & vertical_plateau
        ).to(dtype=x.dtype),
    }
    observed_delta = x - key
    boundary_confidence = _smoothstep(0.03, 0.35, background_nearby)

    # Try several neighbourhood sizes and keep the locally best physical fit.
    # Small estimates avoid averaging adjacent red/blue parts into purple;
    # larger estimates can still reach through a wide motion-blur band.
    candidate_specs = (
        ("square", max(2, seed_radius * 2)),
        ("horizontal", sample_radius),
        ("vertical", sample_radius),
        ("square", sample_radius),
    )
    best_score = torch.full_like(matte, float("inf"))
    fitted_alpha = torch.zeros_like(matte)
    fit_error = torch.ones_like(matte)
    support_confidence = torch.zeros_like(matte)
    profile_confidence = torch.zeros_like(matte)
    seen_candidates = set()
    for orientation, radius in candidate_specs:
        identity = (orientation, int(radius))
        if identity in seen_candidates:
            continue
        seen_candidates.add(identity)
        if orientation == "horizontal":
            average = _horizontal_average
        elif orientation == "vertical":
            average = _vertical_average
        else:
            average = _separable_average
        foreground_seed = foreground_seeds[orientation]
        candidate_support = average(foreground_seed, radius)
        candidate_foreground = average(
            x * foreground_seed, radius
        ) / (candidate_support + 1e-6)
        foreground_delta = candidate_foreground - key
        denominator = foreground_delta.square().sum(dim=1, keepdim=True) + 1e-6
        candidate_alpha = (
            (observed_delta * foreground_delta).sum(dim=1, keepdim=True)
            / denominator
        ).clamp(0.0, 1.0)
        candidate_rgb = key + candidate_alpha * foreground_delta
        candidate_error = _channel_norm3(x - candidate_rgb) / (
            _channel_norm3(foreground_delta) + 0.02
        )
        candidate_support_confidence = _smoothstep(
            0.002, 0.03, candidate_support
        )
        candidate_fit_confidence = _smoothstep(0.14, 0.025, candidate_error)
        candidate_profile_evidence = (
            (candidate_fit_confidence >= 0.60)
            & (candidate_support_confidence >= 0.40)
            & (boundary_confidence >= 0.10)
        )
        candidate_low_mix = (
            candidate_profile_evidence
            & (candidate_alpha >= 0.08)
            & (candidate_alpha <= 0.34)
        ).to(dtype=x.dtype)
        candidate_mid_mix = (
            candidate_profile_evidence
            & (candidate_alpha >= 0.38)
            & (candidate_alpha <= 0.62)
        ).to(dtype=x.dtype)
        candidate_high_mix = (
            candidate_profile_evidence
            & (candidate_alpha >= 0.68)
            & (candidate_alpha <= 0.92)
        ).to(dtype=x.dtype)
        profile_radius = max(3, boundary_radius)
        if orientation == "horizontal":
            maximum = _horizontal_maximum
        elif orientation == "vertical":
            maximum = _vertical_maximum
        else:
            maximum = _separable_maximum
        candidate_profile_confidence = (
            maximum(candidate_low_mix, profile_radius)
            * maximum(candidate_mid_mix, profile_radius)
            * maximum(candidate_high_mix, profile_radius)
        )
        candidate_score = (
            candidate_error + 0.25 * (1.0 - candidate_support_confidence)
            + 0.50 * (1.0 - candidate_profile_confidence)
        )
        better = candidate_score < best_score
        best_score = torch.where(better, candidate_score, best_score)
        fitted_alpha = torch.where(better, candidate_alpha, fitted_alpha)
        fit_error = torch.where(better, candidate_error, fit_error)
        support_confidence = torch.where(
            better, candidate_support_confidence, support_confidence
        )
        profile_confidence = torch.where(
            better, candidate_profile_confidence, profile_confidence
        )

    fit_confidence = _smoothstep(0.14, 0.025, fit_error)
    # A single opaque yellow/orange rim can lie on the same mathematical line
    # as a red/green mixture.  The selected candidate must carry its own
    # direction-consistent low/mid/high alpha progression; evidence from a
    # perpendicular edge or neighbouring coloured part cannot activate it.
    alpha_error = (matte - fitted_alpha).abs().clamp(0.0, 1.0)
    correction_confidence = _smoothstep(0.02, 0.12, alpha_error)
    base_confidence = (
        fit_confidence
        * support_confidence
        * boundary_confidence
        * profile_confidence
    )
    alpha_confidence = (
        base_confidence * correction_confidence
    ).clamp(0.0, 0.95)

    corrected_matte = (
        matte * (1.0 - alpha_confidence)
        + fitted_alpha * alpha_confidence
    ).clamp(0.0, 1.0)

    # Recover unpremultiplied foreground colour from the fitted physical alpha.
    # The local colour estimate is used for the fit; direct unmixing retains more
    # texture than simply copying that local average over the moving edge.
    safe_alpha = fitted_alpha.clamp(min=0.06)
    unmixed = ((x - (1.0 - fitted_alpha) * key) / safe_alpha).clamp(0.0, 1.0)
    mixture_confidence = _smoothstep(
        0.02, 0.18, (1.0 - fitted_alpha).clamp(0.0, 1.0)
    )
    rgb_confidence = (
        base_confidence * mixture_confidence
    ).clamp(0.0, 0.95)
    corrected_rgb = (
        x * (1.0 - rgb_confidence) + unmixed * rgb_confidence
    ).clamp(0.0, 1.0)
    return corrected_rgb, corrected_matte


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
    adaptive_cleanup=True,
):
    x = _to_nchw(rgb_srgb).float().clamp(0, 1)
    if verbose:
        print(f"[Chroma Key Studio V2] batch={x.shape[0]}, mode={background_mode}")
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

    # Batch production uses this path by default.  It establishes a physically
    # plausible baseline first; legacy Edge/Matte Args below remain authoritative
    # optional finishing controls instead of being silently undone afterwards.
    cleaned_source = x
    if adaptive_cleanup:
        cleaned_source, matte = _adaptive_motion_cleanup(x, key, matte)

    if edge_soft > 0.0:
        radius = int(max(1, round(float(edge_soft) * 10)))
        matte = _edge_aware_smooth(matte, radius)
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
            matte = _separable_average_replicate(matte, radius)
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

    rgb_linear = srgb_to_linear(cleaned_source)
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


import math

import torch
from ..core.engine import run as engine_run
from ..core.helpers import to_color3


_ACCELERATOR_MIN_PIXELS = 256 * 256
_ACCELERATOR_CHUNK_PIXELS = 1_000_000
_FAILED_ACCELERATOR_DEVICES = set()


def _comfy_execution_device():
    """Return ComfyUI's selected accelerator without making it a hard import."""
    try:
        from comfy import model_management
        device = model_management.get_torch_device()
    except (ImportError, AttributeError, RuntimeError):
        return None
    if getattr(device, "type", None) == "cpu":
        return None
    if str(device) in _FAILED_ACCELERATOR_DEVICES:
        return None
    return device


def _engine_call(x, key, bg, options):
    return engine_run(x, key, bg_color=bg, **options)


def _run_engine_accelerated(x, key, bg, device, options):
    """Run complete Keylight chunks on one device and return to the input device."""
    frame_pixels = int(x.shape[-2] * x.shape[-1])
    frames_per_chunk = max(1, _ACCELERATOR_CHUNK_PIXELS // frame_pixels)
    alpha_background = str(options.get("background_mode", "alpha")).lower() == "alpha"
    original_device = x.device
    comp_chunks = []
    alpha_chunks = []
    clean_chunks = []

    with torch.inference_mode():
        key_device = key.to(device=device, dtype=x.dtype)
        bg_device = bg.to(device=device, dtype=x.dtype)
        for start in range(0, x.shape[0], frames_per_chunk):
            end = min(x.shape[0], start + frames_per_chunk)
            x_chunk = x[start:end].to(device=device, dtype=x.dtype)
            key_chunk = (
                key_device
                if key_device.shape[0] == 1
                else key_device[start:end]
            )
            comp, alpha, _, clean = _engine_call(
                x_chunk, key_chunk, bg_device, options
            )
            # One packed transfer avoids returning full-resolution results
            # separately.  In alpha mode comp and clean are identical, and
            # mask_image is always an exact alpha repeat, so neither duplicate
            # crosses the bus.
            if alpha_background:
                packed = torch.cat((alpha, clean), dim=1).to(original_device)
                alpha_chunk, clean_chunk = packed.split((1, 3), dim=1)
                comp_chunk = clean_chunk
            else:
                packed = torch.cat((comp, alpha, clean), dim=1).to(original_device)
                comp_chunk, alpha_chunk, clean_chunk = packed.split(
                    (3, 1, 3), dim=1
                )
            comp_chunks.append(comp_chunk)
            alpha_chunks.append(alpha_chunk)
            clean_chunks.append(clean_chunk)

    comp = torch.cat(comp_chunks, dim=0)
    alpha = torch.cat(alpha_chunks, dim=0)
    clean = torch.cat(clean_chunks, dim=0)
    return comp, alpha, alpha.repeat(1, 3, 1, 1), clean


def _release_accelerator_cache(device):
    device_type = getattr(device, "type", None)
    try:
        if device_type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif device_type == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
    except RuntimeError:
        pass


def _run_engine_device_aware(x, key, bg, options):
    """Use ComfyUI's accelerator automatically, with a transparent CPU fallback."""
    total_pixels = int(x.shape[0] * x.shape[-2] * x.shape[-1])
    if x.device.type != "cpu" or total_pixels < _ACCELERATOR_MIN_PIXELS:
        return _engine_call(x, key, bg, options)

    device = _comfy_execution_device()
    if device is None:
        return _engine_call(x, key, bg, options)
    try:
        return _run_engine_accelerated(x, key, bg, device, options)
    except (RuntimeError, NotImplementedError) as error:
        # Unsupported backends and tight VRAM configurations must remain usable.
        # Disable only the failed device for later calls in this process.
        device_name = str(device)
        failure_message = str(error).replace("\n", " ")[:240]
        _FAILED_ACCELERATOR_DEVICES.add(device_name)

    # Leave the exception scope first so its traceback releases any live device
    # tensors before clearing the cache or starting the CPU recomputation.
    _release_accelerator_cache(device)
    print(
        "[Chroma Key Studio V2] Keylight accelerator "
        f"{device_name} unavailable; falling back to CPU ({failure_message})."
    )
    return _engine_call(x, key, bg, options)


class KeylightCoreHubV3:
    CATEGORY = "Chroma Key Studio V2"
    FUNCTION = "apply"
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "mask", "mask_image", "image_rgba")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                "image": ("IMAGE",),
                "key_mode": (["guided","manual","auto"], {"default":"guided"}),
                "key_color": ("COLORCODE", {"default": "#00FF00"}),
                "background_mode": (["alpha","color","soft_color"], {"default":"alpha"}),
                "bg_color": ("COLORCODE", {"default": "#000000"}),
                "tolerance": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "clip_black": ("FLOAT", {"default": -0.02, "min": -1.0, "max": 1.0, "step": 0.001}),
                "clip_white": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 2.0, "step": 0.001}),
                "shadow_recovery": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01}),
                "edge_soft": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "defringe": ("FLOAT", {"default": 0.07, "min": 0.0, "max": 1.0, "step": 0.01}),
                "shrink_expand": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 1.0}),
            },
            "optional": {
                # Keep the legacy Args socket contracts: their dictionary
                # schemas are compatible and this lets existing workflows
                # reconnect without replacing every parameter node.
                "sampler_args": ("KEY_SAMPLER_ARGS",),
                "edge_args": ("KEY_EDGE_ARGS",),
                "spill_algo_args": ("KEY_SPILL_ALGO_ARGS",),
                "ph_args": ("KEY_PH_ARGS",),
                "mm_args": ("KEY_MM_ARGS",),
            }
        }

    def _ensure_bchw(self, img):
        if img.ndim == 4 and img.shape[1] in (1,3,4):
            return img
        if img.ndim == 4 and img.shape[-1] in (1,3,4):
            return img.permute(0,3,1,2)
        raise ValueError("Unsupported image tensor shape. Expect [N,C,H,W] or [N,H,W,C].")

    def _border_pixels(self, bhwc, frac=0.08):
        N,H,W,_ = bhwc.shape
        g = max(1, int(round(min(H,W) * max(0.0, min(0.45, float(frac))))))
        if g*2 >= min(H,W):
            g = max(1, min(H,W)//4)
        top    = bhwc[:, :g, :, :]
        bottom = bhwc[:, -g:, :, :]
        left   = bhwc[:, :, :g, :]
        right  = bhwc[:, :, -g:, :]
        border = torch.cat([top.reshape(N,-1,3), bottom.reshape(N,-1,3),
                            left.reshape(N,-1,3), right.reshape(N,-1,3)], dim=1)
        return border

    def _colour_features(self, values):
        maximum = values.max(dim=2, keepdim=True).values
        minimum = values.min(dim=2, keepdim=True).values
        saturation = (maximum - minimum) / (maximum + 1e-6)
        chroma = values - values.mean(dim=2, keepdim=True)
        direction = chroma / (torch.linalg.norm(chroma, dim=2, keepdim=True) + 1e-6)
        return direction, saturation.squeeze(2), maximum.squeeze(2)

    def _robust_observed_colour(self, samples):
        value = samples.max(dim=1).values
        normalized = samples / (value.unsqueeze(1) + 1e-6)
        shape = normalized.median(dim=0).values
        shape = shape / (shape.max() + 1e-6)
        return (shape * value.median()).clamp(0.0, 1.0)

    def _auto_key_from_border(self, bhwc, frac=0.08):
        """Estimate the dominant saturated border hue without averaging to gray."""
        border = self._border_pixels(bhwc, frac=frac)
        direction, saturation, value = self._colour_features(border)
        refined = []
        for index in range(border.shape[0]):
            valid = (saturation[index] >= 0.08) & (value[index] >= 0.03)
            samples = border[index][valid]
            sample_direction = direction[index][valid]
            minimum = max(16, border.shape[1] // 200)
            if samples.shape[0] < minimum:
                refined.append(border[index].mean(dim=0))
                continue
            weight = (saturation[index][valid] * value[index][valid]).unsqueeze(1)
            anchor_direction = (sample_direction * weight).sum(dim=0)
            anchor_direction = anchor_direction / (torch.linalg.norm(anchor_direction) + 1e-6)
            compatible = (sample_direction * anchor_direction).sum(dim=1) >= math.cos(math.radians(40.0))
            compatible_samples = samples[compatible]
            if compatible_samples.shape[0] < minimum:
                compatible_samples = samples
            refined.append(self._robust_observed_colour(compatible_samples))
        return torch.stack(refined, dim=0).view(border.shape[0], 3, 1, 1)

    def _auto_key_from_rect(self, bhwc, rect):
        # rect: [x,y,w,h] in [0..1], center-based
        N,H,W,_ = bhwc.shape
        x,y,w,h = rect
        cx = int(round(float(x) * W))
        cy = int(round(float(y) * H))
        rw = max(1, int(round(float(w) * W)))
        rh = max(1, int(round(float(h) * H)))
        x0 = max(0, cx - rw//2); x1 = min(W, cx + rw//2)
        y0 = max(0, cy - rh//2); y1 = min(H, cy + rh//2)
        patch = bhwc[:, y0:y1, x0:x1, :]
        if patch.numel() == 0:
            patch = bhwc  # fallback to whole image
        mean = patch.reshape(N,-1,3).mean(dim=1)
        return mean.view(N,3,1,1)

    def _guided_key_from_border(self, bhwc, anchor, frac=0.08, similarity=0.86):
        """Refine a supplied RGB key using only compatible border pixels.

        The supplied key remains the semantic anchor. Full opponent-hue matching
        supports primaries, cyan/yellow/magenta and arbitrary intermediate hues.
        """
        border = self._border_pixels(bhwc, frac=frac)
        N = border.shape[0]
        anchor_rgb = anchor.view(N,1,3).clamp(0,1)
        border_direction, border_saturation, border_value = self._colour_features(border)
        anchor_direction, anchor_saturation, _ = self._colour_features(anchor_rgb)
        cosine = (border_direction * anchor_direction).sum(2)
        saturation_floor = torch.maximum(
            anchor_saturation * 0.15,
            anchor_saturation.new_full(anchor_saturation.shape, 0.08),
        )
        cosine_floor = max(float(similarity), math.cos(math.radians(35.0)))
        valid = (
            (cosine >= cosine_floor)
            & (border_saturation >= saturation_floor)
            & (border_value >= 0.03)
        )

        refined = []
        for i in range(N):
            samples = border[i][valid[i]]
            minimum = max(16, border.shape[1] // 200)
            if samples.shape[0] < minimum:
                refined.append(anchor_rgb[i,0])
            else:
                # Hue-normalized median rejects foreground, gray edges and particles.
                observed = self._robust_observed_colour(samples)
                refined.append(0.15 * anchor_rgb[i,0] + 0.85 * observed)
        frame_estimates = torch.stack(refined, dim=0)
        # A batch median damps frame-to-frame key drift in generated video.
        batch_estimate = frame_estimates.median(dim=0).values.view(1,3)
        stable = 0.70 * frame_estimates + 0.30 * batch_estimate
        return stable.clamp(0.0, 1.0).view(N,3,1,1)

    def apply(self, image, key_mode, key_color, background_mode, bg_color,
              tolerance, clip_black, clip_white,
              sampler_args=None, edge_args=None, spill_algo_args=None, ph_args=None, mm_args=None,
              shadow_recovery=0.85, edge_soft=0.05, defringe=0.07, shrink_expand=0.0):
        # Normalize image to BCHW [N,3,H,W]
        x_bchw = self._ensure_bchw(image).float().clamp(0,1)
        if x_bchw.shape[1] == 1:
            x_bchw = x_bchw.repeat(1,3,1,1)
        if x_bchw.shape[1] == 4:
            x_bchw = x_bchw[:,:3,:,:]

        # Prepare sampler-based auto key if requested
        bhwc = x_bchw.permute(0,2,3,1).contiguous()
        mode_name = str(key_mode)
        if mode_name == "auto":
            s = sampler_args or {}
            mode = str(s.get("mode","auto_border"))
            if mode == "auto_border":
                frac = float(s.get("auto_border_frac", 0.08))
                key_rgb = self._auto_key_from_border(bhwc, frac=frac)
            else:
                rect = s.get("rect", [0.45,0.45,0.1,0.1])
                key_rgb = self._auto_key_from_rect(bhwc, rect=rect)
        else:
            # Manual and guided modes both start from the upstream colour.
            N = x_bchw.shape[0]
            k1 = to_color3(key_color, device=x_bchw.device, dtype=x_bchw.dtype)  # [1,3,1,1]
            key_rgb = k1.repeat(N,1,1,1)  # [N,3,1,1]
            if mode_name == "guided":
                s = sampler_args or {}
                frac = float(s.get("auto_border_frac", 0.08))
                key_rgb = self._guided_key_from_border(bhwc, key_rgb, frac=frac)

        # Background color tensor (single)
        bg_rgb = to_color3(bg_color, device=x_bchw.device, dtype=x_bchw.dtype)

        # Unpack optional args safely
        edge_soft       = float((edge_args or {}).get("edge_soft", edge_soft))
        shrink_expand   = float((edge_args or {}).get("shrink_expand", shrink_expand))
        defringe        = float((edge_args or {}).get("defringe", defringe))
        spill_cfg       = spill_algo_args or None
        ph_cfg          = ph_args or None
        mm_cfg          = mm_args or None

        # Keep matte, motion cleanup, despill and defringe together on ComfyUI's
        # selected execution device.  Large video batches are chunked by pixels;
        # unsupported accelerators transparently fall back to the optimized CPU.
        engine_options = {
            "tolerance": float(tolerance),
            "clip_black": float(clip_black),
            "clip_white": float(clip_white),
            "edge_soft": edge_soft,
            "shrink_expand": shrink_expand,
            "defringe": defringe,
            "spill_algo": spill_cfg,
            "ph": ph_cfg,
            "matte_math": mm_cfg,
            "background_mode": str(background_mode),
            "use_linear": True,
            "verbose": False,
            "shadow_recovery": float(shadow_recovery),
        }
        comp, a, mask_img, clean_foreground = _run_engine_device_aware(
            x_bchw, key_rgb, bg_rgb, engine_options
        )

        # Build normalized outputs
        comp_bhwc = comp.permute(0,2,3,1).contiguous()
        clean_bhwc = clean_foreground.permute(0,2,3,1).contiguous()
        mask_hw   = a[:,0,:,:] if a.dim()==4 and a.shape[1]==1 else a.squeeze(1)
        if mask_hw.dim()==2:
            mask_hw = mask_hw.unsqueeze(0)
        alpha     = mask_hw.unsqueeze(-1)  # [N,H,W,1]

        # Always provide RGBA
        try:
            image_rgba = torch.cat([clean_bhwc, alpha], dim=-1).clamp(0.0, 1.0)
        except Exception:
            ones = clean_bhwc.new_ones((*clean_bhwc.shape[:3],1))
            image_rgba = torch.cat([clean_bhwc[...,:3], ones], dim=-1)

        # Primary 'image' output auto-switch
        bgm = str(background_mode).lower() if isinstance(background_mode, str) else "alpha"
        image_out = image_rgba if bgm == "alpha" else comp_bhwc

        mask_image_bhwc = mask_img.permute(0,2,3,1).contiguous()
        return (image_out, mask_hw, mask_image_bhwc, image_rgba)

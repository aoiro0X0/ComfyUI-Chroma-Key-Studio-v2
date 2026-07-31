
import torch
from ..core.engine import run as engine_run, _to_nchw
from ..core.helpers import to_color3

class KeylightCoreHubV3:
    CATEGORY = "KeylightChromaKeyHub"
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

    def _auto_key_from_border(self, bhwc, frac=0.08):
        # bhwc: [N,H,W,3]
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
        mean = border.mean(dim=1)  # [N,3]
        return mean.view(N,3,1,1)  # -> [N,3,1,1]

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

        The supplied pure R/G/B key remains the semantic anchor. Per-frame
        border sampling only compensates for hue drift introduced by video
        generation, compression and lighting.
        """
        N,H,W,_ = bhwc.shape
        g = max(1, int(round(min(H,W) * max(0.0, min(0.45, float(frac))))))
        if g*2 >= min(H,W):
            g = max(1, min(H,W)//4)
        border = torch.cat([
            bhwc[:, :g].reshape(N,-1,3), bhwc[:, -g:].reshape(N,-1,3),
            bhwc[:, :, :g].reshape(N,-1,3), bhwc[:, :, -g:].reshape(N,-1,3)
        ], dim=1)

        # Match in linear RGB direction so exposure changes do not prevent a
        # valid screen sample. Very dark pixels are excluded to protect black.
        border_lin = torch.where(
            border <= 0.04045, border/12.92,
            ((border+0.055)/1.055).clamp(min=0)**2.4
        )
        anchor_rgb = anchor.view(N,1,3).clamp(0,1)
        anchor_lin = torch.where(
            anchor_rgb <= 0.04045, anchor_rgb/12.92,
            ((anchor_rgb+0.055)/1.055).clamp(min=0)**2.4
        )
        border_mag = torch.linalg.norm(border_lin, dim=2, keepdim=True)
        anchor_dir = anchor_lin / (torch.linalg.norm(anchor_lin, dim=2, keepdim=True) + 1e-6)
        border_dir = border_lin / (border_mag + 1e-6)
        cosine = (border_dir * anchor_dir).sum(2)
        anchor_channel = torch.argmax(anchor_rgb, dim=2)
        border_channel = torch.argmax(border, dim=2)
        valid = ((cosine >= float(similarity)) &
                 (border_mag.squeeze(2) >= 0.008) &
                 (border_channel == anchor_channel))

        refined = []
        for i in range(N):
            samples = border[i][valid[i]]
            minimum = max(16, border.shape[1] // 200)
            if samples.shape[0] < minimum:
                refined.append(anchor_rgb[i,0])
            else:
                # Median rejects foreground, particles and isolated highlights.
                observed = samples.median(dim=0).values
                refined.append(0.15 * anchor_rgb[i,0] + 0.85 * observed)
        return torch.stack(refined, dim=0).view(N,3,1,1)

    def apply(self, image, key_mode, key_color, background_mode, bg_color,
              tolerance, clip_black, clip_white, shadow_recovery=0.85,
              edge_soft=0.05, defringe=0.07, shrink_expand=0.0,
              sampler_args=None, edge_args=None, spill_algo_args=None, ph_args=None, mm_args=None):
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

        # Call engine
        comp, a, mask_img, _ = engine_run(
            x_bchw, key_rgb, float(tolerance), float(clip_black), float(clip_white),
            edge_soft=edge_soft, shrink_expand=shrink_expand, defringe=defringe,
            spill_algo=spill_cfg, ph=ph_cfg, matte_math=mm_cfg,
            background_mode=str(background_mode), bg_color=bg_rgb,
            use_linear=True, verbose=False, shadow_recovery=float(shadow_recovery)
        )

        # Build normalized outputs
        comp_bhwc = comp.permute(0,2,3,1).contiguous()
        mask_hw   = a[:,0,:,:] if a.dim()==4 and a.shape[1]==1 else a.squeeze(1)
        if mask_hw.dim()==2:
            mask_hw = mask_hw.unsqueeze(0)
        alpha     = mask_hw.unsqueeze(-1)  # [N,H,W,1]

        # Always provide RGBA
        try:
            image_rgba = torch.cat([comp_bhwc, alpha], dim=-1).clamp(0.0, 1.0)
        except Exception:
            ones = comp_bhwc.new_ones((*comp_bhwc.shape[:3],1))
            image_rgba = torch.cat([comp_bhwc[...,:3], ones], dim=-1)

        # Primary 'image' output auto-switch
        bgm = str(background_mode).lower() if isinstance(background_mode, str) else "alpha"
        image_out = image_rgba if bgm == "alpha" else comp_bhwc

        mask_image_bhwc = mask_img.permute(0,2,3,1).contiguous()
        return (image_out, mask_hw, mask_image_bhwc, image_rgba)

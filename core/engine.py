
import torch, torch.nn.functional as F

def srgb_to_linear(x):
    return torch.where(x <= 0.04045, x/12.92, ((x+0.055)/(1.055))**2.4)

def linear_to_srgb(x):
    return torch.where(x <= 0.0031308, x*12.92, 1.055*torch.clamp(x,min=0)**(1.0/2.4)-0.055)

def _to_nchw(img):
    if img.ndim == 4 and img.shape[1] in (1,3,4):
        return img
    if img.ndim == 4 and img.shape[-1] in (1,3,4):
        return img.permute(0,3,1,2)
    raise ValueError("Unsupported image tensor shape.")

def _normalize(v, eps=1e-6):
    n = torch.linalg.norm(v, dim=1, keepdim=True) + eps
    return v / n

def _primary_channel_from_k(k):
    return torch.argmax(k, dim=1)

def _morph_shrink_expand(matte, amount):
    if amount == 0: return matte
    r = int(abs(amount))
    if r == 0: return matte
    if amount > 0:
        return torch.nn.functional.max_pool2d(matte, kernel_size=2*r+1, stride=1, padding=r)
    else:
        return -torch.nn.functional.max_pool2d(-matte, kernel_size=2*r+1, stride=1, padding=r)

def build_key_vector(key_rgb):
    k_lin = srgb_to_linear(key_rgb.clamp(0,1))
    return _normalize(k_lin + 1e-6)

def compute_matte(img_srgb, key_rgb, tolerance, clip_black, clip_white,
                  use_linear=True, shadow_recovery=0.85):
    x = _to_nchw(img_srgb).float().clamp(0,1)
    k = build_key_vector(key_rgb)
    xx = srgb_to_linear(x) if use_linear else x
    proj = (xx * k).sum(1, keepdim=True)
    dist = torch.linalg.norm(xx - proj*k, dim=1, keepdim=True)
    score = proj - tolerance*dist
    m = (score - clip_black) / max(1e-6, (clip_white - clip_black))
    matte = 1.0 - m.clamp(0,1)

    # Projection-based keys become weak when the screen is darkened by a video
    # model. Recover coloured shadows from their direction while protecting
    # genuinely black foreground pixels with a magnitude gate.
    recovery = max(0.0, min(1.0, float(shadow_recovery)))
    if recovery > 0.0:
        mag = torch.linalg.norm(xx, dim=1, keepdim=True)
        direction = xx / (mag + 1e-6)
        similarity = (direction * k).sum(1, keepdim=True).clamp(0, 1)
        similarity_floor = max(0.55, min(0.92, 0.82 - 0.10 * float(tolerance)))
        chroma_match = ((similarity - similarity_floor) /
                        max(1e-6, 1.0 - similarity_floor)).clamp(0, 1)
        # Linear RGB magnitude below this range remains foreground. This is
        # what keeps a black subject intact even when its hue is undefined.
        visible_colour = ((mag - 0.008) / 0.045).clamp(0, 1)
        chroma_background = chroma_match * visible_colour
        matte = matte * (1.0 - recovery * chroma_background)
        black_protect = (1.0 - mag / 0.008).clamp(0, 1)
        matte = torch.maximum(matte, black_protect)
    return matte.clamp(0,1), proj, dist, k

def _spill_suppress(work, proj, dist, k, matte, desp, balance=1.2, extra_lowalpha=0.2, ph_mask=None, ph_strength=0.0, mode='hybrid'):
    geometric=(proj - float(balance)*dist).clamp(min=0.0)
    r,g,b = work[:,0:1], work[:,1:2], work[:,2:3]
    idx = _primary_channel_from_k(k)
    other_max = torch.stack([torch.maximum(g,b), torch.maximum(r,b), torch.maximum(r,g)], dim=1)
    key_ch = torch.stack([r,g,b], dim=1)
    gather_idx = idx.view(-1,1,1,1,1).repeat(1,1,1,work.shape[2],work.shape[3])
    key_sel = torch.gather(key_ch, 1, gather_idx).squeeze(1)
    oth_sel = torch.gather(other_max,1, gather_idx).squeeze(1)
    channel_excess = (key_sel - oth_sel).clamp(min=0.0)
    others = work.clone()
    scatter_idx = idx.view(-1,1,1,1).expand(-1,1,work.shape[2],work.shape[3])
    others.scatter_(1, scatter_idx, -1e6)
    other_only_max = others.max(1, keepdim=True)[0]
    screen = (work * k).sum(1, keepdim=True)
    screen_excess = (screen - other_only_max).clamp(min=0.0)

    if mode == 'geometric':
        spill = geometric
    elif mode == 'screen':
        spill = screen_excess
    else:
        spill = 0.5*geometric + 0.5*channel_excess + 0.5*screen_excess

    w = (1.0 - matte).clamp(0.0, 1.0)
    suppress = (float(desp) + float(extra_lowalpha)*w).clamp(min=0.0)
    if ph_mask is not None and ph_strength>0.0:
        atten = (1.0 - ph_mask*ph_strength*0.8).clamp(0.25, 1.0)
        suppress = suppress * atten

    return (work - spill*suppress*k).clamp(0,1)

def composite(rgb_srgb, matte, mode='alpha', bg_color=None):
    x = _to_nchw(rgb_srgb).clamp(0,1)
    a = matte.clamp(0,1)
    if mode == 'alpha':
        return x, a
    if bg_color is None:
        bg_color = torch.tensor([0.0,0.0,0.0], device=x.device, dtype=x.dtype).view(1,3,1,1)
    else:
        bg_color = bg_color.view(1,3,1,1).to(x.device, x.dtype)
    if mode == 'color':
        comp = x * a + bg_color * (1.0 - a)
        return comp, a
    if mode == 'soft_color':
        k = torch.tensor([[[[1,2,1],[2,4,2],[1,2,1]]]], device=a.device, dtype=a.dtype)/16.0
        a_soft = torch.nn.functional.conv2d(a, k, padding=1)
        comp = x * a_soft + bg_color * (1.0 - a_soft)
        return comp, a_soft
    return x, a

def run(rgb_srgb, key_rgb, tolerance, clip_black, clip_white,
        edge_soft=0.0, shrink_expand=0.0, defringe=0.0,
        spill_algo=None, ph=None, matte_math=None,
        background_mode='alpha', bg_color=None, use_linear=True, verbose=False,
        shadow_recovery=0.85):
    x = _to_nchw(rgb_srgb).float().clamp(0,1)
    B = x.shape[0]
    if verbose:
        print(f"[Keylight] batch={B}, mode={background_mode}")
    if key_rgb.ndim == 2:
        k = key_rgb.view(B,3,1,1)
    else:
        k = key_rgb
    matte, proj, dist, key_vec = compute_matte(
        x, k, tolerance, clip_black, clip_white,
        use_linear=use_linear, shadow_recovery=shadow_recovery
    )

    if edge_soft>0.0:
        g = int(max(1, round(edge_soft*10)))
        kernel = torch.ones((1,1,2*g+1,2*g+1), device=matte.device, dtype=matte.dtype)/(2*g+1)**2
        matte = torch.nn.functional.conv2d(matte, kernel, padding=g)
    if abs(shrink_expand)>0.0:
        matte = _morph_shrink_expand(matte, shrink_expand)

    algo = (spill_algo or {}).get("algo","blend")
    diff_gain = float((spill_algo or {}).get("diff_gain",1.0))
    diff_balance = float((spill_algo or {}).get("diff_balance",1.0))
    despill = float((spill_algo or {}).get("despill",2.0))
    extra_lowalpha = float((spill_algo or {}).get("extra_lowalpha",1.5))
    final_despill_strength = float((spill_algo or {}).get("final_despill_strength",0.6))
    despill_mode = str((spill_algo or {}).get("despill_mode","hybrid"))
    _algo = str((spill_algo or {}).get("algo",""))
    if _algo in ("diff","proj","blend"):
        # prefer legacy 'algo' toggle for mode selection
        if _algo == "diff":
            despill_mode = "geometric"
        elif _algo == "proj":
            despill_mode = "screen"
        else:
            despill_mode = "hybrid"


    ph_mask = None
    ph_strength = 0.0
    if ph:
        ph_thr = float(ph.get("thr",0.75))
        ph_width = float(ph.get("soft_width",0.2))
        ph_gamma = float(ph.get("gamma",1.0))
        ph_strength = float(ph.get("strength",0.7))
        lum = (0.2126*x[:,0:1] + 0.7152*x[:,1:2] + 0.0722*x[:,2:3]).clamp(0,1)
        t0, t1 = ph_thr - ph_width*0.5, ph_thr + ph_width*0.5
        m = ((lum - t0)/(t1 - t0 + 1e-6)).clamp(0,1)
        ph_mask = m**ph_gamma

    proj_p = proj * diff_gain
    dist_p = dist / max(1e-6, diff_balance)

    rgb_lin = srgb_to_linear(x)
    out_lin = _spill_suppress(rgb_lin, proj_p, dist_p, key_vec, matte, despill, diff_balance, extra_lowalpha, ph_mask, ph_strength, mode=despill_mode)
    if final_despill_strength>0.0:
        out_lin = out_lin.lerp(rgb_lin, 1.0 - final_despill_strength).clamp(0,1)
    out = linear_to_srgb(out_lin).clamp(0,1)
    # simple defringe: attenuate key channel in background to reduce color halos
    if defringe>0.0:
        idx = _primary_channel_from_k(key_vec)
        w = (1.0 - matte).clamp(0.0,1.0)
        # gather key channel
        gather_idx = idx.view(-1,1,1,1).repeat(1,1,out.shape[2],out.shape[3])
        key_ch = torch.gather(out, 1, gather_idx)
        key_ch = (key_ch * (1.0 - defringe*w)).clamp(0,1)
        out = out.scatter(1, gather_idx, key_ch)

    if matte_math:
        fea = float(matte_math.get("feather",0.0))
        extra = float(matte_math.get("extra_shrink_expand",0.0))
        gamma = float(matte_math.get("gamma",1.0))
        if abs(extra)>0.0:
            matte = _morph_shrink_expand(matte, extra)
        if fea>0.0:
            g = int(max(1, round(fea*10)))
            kernel = torch.ones((1,1,2*g+1,2*g+1), device=matte.device, dtype=matte.dtype)/(2*g+1)**2
            matte = torch.nn.functional.conv2d(matte, kernel, padding=g)
        if gamma!=1.0:
            matte = matte.clamp(0,1) ** gamma

    comp, a = composite(out, matte, mode=background_mode, bg_color=bg_color)
    mask_img = a.repeat(1,3,1,1)
    return comp, a, mask_img, out

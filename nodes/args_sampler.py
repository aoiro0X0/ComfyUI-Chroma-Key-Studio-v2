
class KeySamplerArgsV2_3_6:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mode": (["manual","auto_border"], {"default":"auto_border"}),
            "auto_border_frac": ("FLOAT", {"default":0.08, "min":0.0, "max":0.45, "step":0.001}),
            "rect_x": ("FLOAT", {"default":0.45, "min":0.0, "max":1.0, "step":0.001}),
            "rect_y": ("FLOAT", {"default":0.45, "min":0.0, "max":1.0, "step":0.001}),
            "rect_w": ("FLOAT", {"default":0.1, "min":0.0, "max":1.0, "step":0.001}),
            "rect_h": ("FLOAT", {"default":0.1, "min":0.0, "max":1.0, "step":0.001}),
        }}
    RETURN_TYPES = ("KEY_SAMPLER_ARGS",)
    RETURN_NAMES = ("sampler_args",)
    FUNCTION = "build"
    CATEGORY = "Chroma Key Studio V2"
    def build(self, mode, auto_border_frac, rect_x, rect_y, rect_w, rect_h):
        return ({ "mode": str(mode), "auto_border_frac": float(auto_border_frac),
                  "rect": [float(rect_x), float(rect_y), float(rect_w), float(rect_h)] },)

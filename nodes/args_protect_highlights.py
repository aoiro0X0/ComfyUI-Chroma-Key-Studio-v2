
class KeyProtectHighlightsArgsV2_3_6:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "thr": ("FLOAT", {"default":0.75, "min":0.0, "max":1.0, "step":0.001}),
            "strength": ("FLOAT", {"default":0.7, "min":0.0, "max":1.0, "step":0.01}),
            "soft_width": ("FLOAT", {"default":0.2, "min":0.0, "max":1.0, "step":0.01}),
            "gamma": ("FLOAT", {"default":1.1, "min":0.1, "max":3.0, "step":0.01}),
        }}
    RETURN_TYPES = ("CHROMA_STUDIO_V2_PH_ARGS",)
    RETURN_NAMES = ("ph_args",)
    FUNCTION = "build"
    CATEGORY = "Chroma Key Studio V2"
    def build(self, thr, strength, soft_width, gamma):
        return ({ "thr": float(thr), "strength": float(strength),
                  "soft_width": float(soft_width), "gamma": float(gamma) },)

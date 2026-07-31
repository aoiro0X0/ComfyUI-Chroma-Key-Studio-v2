
class KeyMatteMathArgsV2_3_6:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "extra_shrink_expand": ("FLOAT", {"default":0.0, "min":-3.0, "max":3.0, "step":1.0}),
            "feather": ("FLOAT", {"default":0.15, "min":0.0, "max":2.0, "step":0.01}),
            "gamma": ("FLOAT", {"default":1.0, "min":0.2, "max":4.0, "step":0.01}),
        }}
    RETURN_TYPES = ("CHROMA_STUDIO_V2_MM_ARGS",)
    RETURN_NAMES = ("mm_args",)
    FUNCTION = "build"
    CATEGORY = "Chroma Key Studio V2"
    def build(self, extra_shrink_expand, feather, gamma):
        return ({ "extra_shrink_expand": float(extra_shrink_expand),
                  "feather": float(feather), "gamma": float(gamma) },)

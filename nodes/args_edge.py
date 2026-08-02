
class KeyEdgeArgsV2_3_6:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "shrink_expand": ("FLOAT", {"default":0.0, "min":-5.0, "max":5.0, "step":1.0}),
            "edge_soft": ("FLOAT", {"default":0.08, "min":0.0, "max":1.0, "step":0.01}),
            "defringe": ("FLOAT", {"default":0.07, "min":0.0, "max":1.0, "step":0.01}),
        }}
    RETURN_TYPES = ("KEY_EDGE_ARGS",)
    RETURN_NAMES = ("edge_args",)
    FUNCTION = "build"
    CATEGORY = "Chroma Key Studio V2"
    def build(self, shrink_expand, edge_soft, defringe):
        return ({ "shrink_expand": float(shrink_expand),
                  "edge_soft": float(edge_soft), "defringe": float(defringe) },)

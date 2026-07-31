
class KeySpillAlgoArgsV2_3_6:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "despill_mode": (["hybrid","screen","geometric"], {"default":"hybrid"}),
            "algo": (["blend","diff","proj"], {"default":"blend"}),
            "diff_gain": ("FLOAT", {"default":1.0, "min":0.0, "max":5.0, "step":0.01}),
            "diff_balance": ("FLOAT", {"default":1.0, "min":0.0, "max":3.0, "step":0.01}),
            "despill": ("FLOAT", {"default":2.0, "min":0.0, "max":8.0, "step":0.01}),
            "extra_lowalpha": ("FLOAT", {"default":1.5, "min":0.0, "max":8.0, "step":0.01}),
            "final_despill_strength": ("FLOAT", {"default":0.6, "min":0.0, "max":1.5, "step":0.01}),
        }}
    RETURN_TYPES = ("CHROMA_STUDIO_V2_SPILL_ALGO_ARGS",)
    RETURN_NAMES = ("spill_algo_args",)
    FUNCTION = "build"
    CATEGORY = "Chroma Key Studio V2"
    def build(self, despill_mode, algo, diff_gain, diff_balance, despill, extra_lowalpha, final_despill_strength):
        return ({ "despill_mode": str(despill_mode), "algo": str(algo),
                 "diff_gain": float(diff_gain), "diff_balance": float(diff_balance),
                 "despill": float(despill), "extra_lowalpha": float(extra_lowalpha),
                 "final_despill_strength": float(final_despill_strength) },)

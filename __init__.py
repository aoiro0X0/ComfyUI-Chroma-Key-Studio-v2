
import os, sys, importlib.util

HERE = os.path.dirname(__file__)

# Tell ComfyUI where to find our frontend extension (registers COLORCODE widget)
WEB_DIRECTORY = "./web"

def _load_module(name, relpath):
    path = os.path.join(HERE, relpath)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # Set __package__ to support relative imports
    if "." in name:
        mod.__package__ = ".".join(name.split(".")[:-1])
    spec.loader.exec_module(mod)
    sys.modules[name] = mod
    return mod

# Load core modules first to support relative imports in nodes
core_engine = _load_module("KeylightChromaKeyHub.core.engine", os.path.join("core","engine.py"))
core_helpers = _load_module("KeylightChromaKeyHub.core.helpers", os.path.join("core","helpers.py"))

# Create a fake parent module for relative imports to work
if "KeylightChromaKeyHub.core" not in sys.modules:
    import types
    core_module = types.ModuleType("KeylightChromaKeyHub.core")
    sys.modules["KeylightChromaKeyHub.core"] = core_module
sys.modules["KeylightChromaKeyHub.core"].engine = core_engine
sys.modules["KeylightChromaKeyHub.core"].helpers = core_helpers

# Create a fake parent module for nodes
if "KeylightChromaKeyHub.nodes" not in sys.modules:
    import types
    nodes_module = types.ModuleType("KeylightChromaKeyHub.nodes")
    sys.modules["KeylightChromaKeyHub.nodes"] = nodes_module

core_hub   = _load_module("KeylightChromaKeyHub.nodes.core_hub", os.path.join("nodes","core_hub.py"))
args_spill = _load_module("KeylightChromaKeyHub.nodes.args_spill_algo", os.path.join("nodes","args_spill_algo.py"))
args_ph    = _load_module("KeylightChromaKeyHub.nodes.args_protect_highlights", os.path.join("nodes","args_protect_highlights.py"))
args_edge  = _load_module("KeylightChromaKeyHub.nodes.args_edge", os.path.join("nodes","args_edge.py"))
args_mm    = _load_module("KeylightChromaKeyHub.nodes.args_matte_math", os.path.join("nodes","args_matte_math.py"))
args_smp   = _load_module("KeylightChromaKeyHub.nodes.args_sampler", os.path.join("nodes","args_sampler.py"))
smart_bg   = _load_module("KeylightChromaKeyHub.smart_background", "smart_background.py")

KeylightCoreHubV3              = core_hub.KeylightCoreHubV3
KeySpillAlgoArgsV2_3_6         = args_spill.KeySpillAlgoArgsV2_3_6
KeyProtectHighlightsArgsV2_3_6 = args_ph.KeyProtectHighlightsArgsV2_3_6
KeyEdgeArgsV2_3_6              = args_edge.KeyEdgeArgsV2_3_6
KeyMatteMathArgsV2_3_6         = args_mm.KeyMatteMathArgsV2_3_6
KeySamplerArgsV2_3_6           = args_smp.KeySamplerArgsV2_3_6
AutoChromaSmartBackground      = smart_bg.AutoChromaSmartBackground
KeylightSmartBackground        = smart_bg.KeylightSmartBackground

# Keep mapping KEYS identical to previous version to avoid breaking old workflows
NODE_CLASS_MAPPINGS = {
    "AutoChromaSmartBackground": AutoChromaSmartBackground,
    "KeylightSmartBackground": KeylightSmartBackground,
    "KeylightCoreHubV3": KeylightCoreHubV3,
    "Key Spill/Algo Args (V2.3.6fixE2_clean)": KeySpillAlgoArgsV2_3_6,
    "Key Protect Highlights Args (V2.3.6fixE2_clean)": KeyProtectHighlightsArgsV2_3_6,
    "Key Edge Args (V2.3.6fixE2_clean)": KeyEdgeArgsV2_3_6,
    "Key Matte Math Args (V2.3.6fixE2_clean)": KeyMatteMathArgsV2_3_6,
    "Key Sampler Args (V2.3.6fixE2_clean)": KeySamplerArgsV2_3_6,
}

# Keep mapping keys stable while exposing the new version in the UI.
NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoChromaSmartBackground": "Smart Chroma Background (智能抠像背景 V2.0)",
    "KeylightSmartBackground": "Smart Chroma Background (兼容别名)",
    "KeylightCoreHubV3": "Adaptive Chroma Keylight (V3.1.0)",
    "Key Spill/Algo Args (V2.3.6fixE2_clean)": "Key Spill/Algo Args (V3.1.0)",
    "Key Protect Highlights Args (V2.3.6fixE2_clean)": "Key Protect Highlights Args (V3.1.0)",
    "Key Edge Args (V2.3.6fixE2_clean)": "Key Edge Args (V3.1.0)",
    "Key Matte Math Args (V2.3.6fixE2_clean)": "Key Matte Math Args (V3.1.0)",
    "Key Sampler Args (V2.3.6fixE2_clean)": "Key Sampler Args (V3.1.0)",
}

__all__ = ["NODE_CLASS_MAPPINGS","NODE_DISPLAY_NAME_MAPPINGS"]


import os, sys, importlib.util

HERE = os.path.dirname(__file__)

# Tell ComfyUI where to find the V2-only colour widget extension.
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

# Use a V2-only Python namespace so the legacy Keylight plugin can be loaded in
# the same ComfyUI process without either package replacing the other's modules.
PYTHON_NAMESPACE = "ChromaKeyStudioV2"

# Load core modules first to support relative imports in nodes
core_engine = _load_module(f"{PYTHON_NAMESPACE}.core.engine", os.path.join("core","engine.py"))
core_helpers = _load_module(f"{PYTHON_NAMESPACE}.core.helpers", os.path.join("core","helpers.py"))

# Create a fake parent module for relative imports to work
if f"{PYTHON_NAMESPACE}.core" not in sys.modules:
    import types
    core_module = types.ModuleType(f"{PYTHON_NAMESPACE}.core")
    sys.modules[f"{PYTHON_NAMESPACE}.core"] = core_module
sys.modules[f"{PYTHON_NAMESPACE}.core"].engine = core_engine
sys.modules[f"{PYTHON_NAMESPACE}.core"].helpers = core_helpers

# Create a fake parent module for nodes
if f"{PYTHON_NAMESPACE}.nodes" not in sys.modules:
    import types
    nodes_module = types.ModuleType(f"{PYTHON_NAMESPACE}.nodes")
    sys.modules[f"{PYTHON_NAMESPACE}.nodes"] = nodes_module

core_hub   = _load_module(f"{PYTHON_NAMESPACE}.nodes.core_hub", os.path.join("nodes","core_hub.py"))
args_spill = _load_module(f"{PYTHON_NAMESPACE}.nodes.args_spill_algo", os.path.join("nodes","args_spill_algo.py"))
args_ph    = _load_module(f"{PYTHON_NAMESPACE}.nodes.args_protect_highlights", os.path.join("nodes","args_protect_highlights.py"))
args_edge  = _load_module(f"{PYTHON_NAMESPACE}.nodes.args_edge", os.path.join("nodes","args_edge.py"))
args_mm    = _load_module(f"{PYTHON_NAMESPACE}.nodes.args_matte_math", os.path.join("nodes","args_matte_math.py"))
args_smp   = _load_module(f"{PYTHON_NAMESPACE}.nodes.args_sampler", os.path.join("nodes","args_sampler.py"))
smart_bg   = _load_module(f"{PYTHON_NAMESPACE}.smart_background", "smart_background.py")

ChromaKeyStudioKeylightV2          = core_hub.KeylightCoreHubV3
ChromaKeyStudioSpillArgsV2         = args_spill.KeySpillAlgoArgsV2_3_6
ChromaKeyStudioProtectArgsV2       = args_ph.KeyProtectHighlightsArgsV2_3_6
ChromaKeyStudioEdgeArgsV2          = args_edge.KeyEdgeArgsV2_3_6
ChromaKeyStudioMatteMathArgsV2     = args_mm.KeyMatteMathArgsV2_3_6
ChromaKeyStudioSamplerArgsV2       = args_smp.KeySamplerArgsV2_3_6
ChromaKeyStudioSmartBackgroundV2   = smart_bg.AutoChromaSmartBackground

# Every mapping key is V2-specific so all legacy repositories can remain installed.
NODE_CLASS_MAPPINGS = {
    "ChromaKeyStudioSmartBackgroundV2": ChromaKeyStudioSmartBackgroundV2,
    "ChromaKeyStudioKeylightV2": ChromaKeyStudioKeylightV2,
    "ChromaKeyStudioSpillArgsV2": ChromaKeyStudioSpillArgsV2,
    "ChromaKeyStudioProtectHighlightsArgsV2": ChromaKeyStudioProtectArgsV2,
    "ChromaKeyStudioEdgeArgsV2": ChromaKeyStudioEdgeArgsV2,
    "ChromaKeyStudioMatteMathArgsV2": ChromaKeyStudioMatteMathArgsV2,
    "ChromaKeyStudioSamplerArgsV2": ChromaKeyStudioSamplerArgsV2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChromaKeyStudioSmartBackgroundV2": "Smart Chroma Background (Studio V2)",
    "ChromaKeyStudioKeylightV2": "Adaptive Chroma Keylight (Studio V2)",
    "ChromaKeyStudioSpillArgsV2": "Key Spill/Algo Args (Studio V2)",
    "ChromaKeyStudioProtectHighlightsArgsV2": "Key Protect Highlights Args (Studio V2)",
    "ChromaKeyStudioEdgeArgsV2": "Key Edge Args (Studio V2)",
    "ChromaKeyStudioMatteMathArgsV2": "Key Matte Math Args (Studio V2)",
    "ChromaKeyStudioSamplerArgsV2": "Key Sampler Args (Studio V2)",
}

__all__ = ["NODE_CLASS_MAPPINGS","NODE_DISPLAY_NAME_MAPPINGS"]

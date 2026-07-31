# Third-party notices

ComfyUI Chroma Key Studio V2 integrates and extends node behavior from:

- `muriellee1x/ComfyUI-Mysterious-node1`
- `muriellee1x/ComfyUI-Mysterious-node2`

V2 uses independent node IDs, Python and frontend namespaces, colour types, and parameter socket types. It does not register the legacy mapping IDs, so the original plugins can remain installed.

The repository retains the downstream GitHub fork relationship for the Keylight source. The smart-background implementation was independently rewritten and extended with black-edge exclusion, primary-first adaptive hue selection, batch-stable output, and salient-accent protection.

`web/colorWidget.js` is based on the AILab colour widget and retains its GPL-3.0 notice and attribution in the source file.

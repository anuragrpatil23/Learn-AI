"""Which part of a drawing each module class is.

Every module class in a model must be named here, or the run stops and names it. Guessing would put a
wrong shape in the drawing without anyone noticing; a stop is noticed at once, and fixing it is one line.

A class is one of:
  an op kind      one step of the drawing (the kinds are the format's fixed list, in OP_KINDS)
  "container"     only holds other modules; look inside it
  "block"         one repeated block of the stream (norm, attention or memory, add, norm, MLP, add)
  "part"          only ever inside an op; the op's own reader handles it
  "skip"          does nothing to the numbers at inference (dropout), or only feeds another op
                  (rotary positions feed attention, and attention's note says so)
"""

OP_KINDS = ("input", "embed", "patchify", "position", "norm", "linear", "attention", "memory", "mlp", "act",
            "add", "join", "unembed", "output")

CLASS_KINDS = {
    # torch
    "ModuleList": "container",
    "Embedding": "embed",          # a position table is told apart by its name, see POSITION_NAMES
    "LayerNorm": "norm",
    "Linear": "linear",            # the last one, onto the vocabulary, is the unembedding, see UNEMBED_NAMES
    "Dropout": "skip",
    "Conv1d": "part",
    "Conv3d": "part",
    "GELU": "act",
    "SiLU": "act",
    # transformers' own activations and layers
    "NewGELUActivation": "act",
    "GELUActivation": "act",
    "GELUTanh": "act",
    "SiLUActivation": "act",
    "Conv1D": "part",              # GPT-2's linear layer, weights stored (in, out)
    # GPT-2
    "GPT2LMHeadModel": "container",
    "GPT2Model": "container",
    "GPT2Block": "block",
    "GPT2Attention": "attention",
    "GPT2MLP": "mlp",
    # Qwen3.5 (the language model and the vision tower)
    "Qwen3_5ForConditionalGeneration": "container",
    "Qwen3_5ForCausalLM": "container",
    "Qwen3_5Model": "container",
    "Qwen3_5TextModel": "container",
    "Qwen3_5VisionModel": "container",
    "Qwen3_5VisionPatchMerger": "container",   # norm, four patches joined, linear, GELU, linear
    "Qwen3_5VisionPatchEmbed": "patchify",
    "Qwen3_5VisionRotaryEmbedding": "skip",
    "Qwen3_5TextRotaryEmbedding": "skip",
    "Qwen3_5VisionBlock": "block",
    "Qwen3_5VisionAttention": "attention",
    "Qwen3_5VisionMLP": "mlp",
    "Qwen3_5DecoderLayer": "block",
    "Qwen3_5Attention": "attention",
    "Qwen3_5GatedDeltaNet": "memory",
    "Qwen3_5RMSNorm": "norm",
    "Qwen3_5RMSNormGated": "part",
    "Qwen3_5MLP": "mlp",
}

# An Embedding with one of these attribute names adds places, it does not look up tokens.
POSITION_NAMES = {"wpe", "pos_embed", "position_embeddings", "embed_positions"}
# A Linear with one of these names, at the top of the model, scores every token of the vocabulary.
UNEMBED_NAMES = {"lm_head"}
# Plain names for the bend in an MLP, as the drawing writes them.
ACT_NAMES = {"NewGELUActivation": "GELU", "GELUActivation": "GELU", "GELUTanh": "GELU", "GELU": "GELU",
             "SiLUActivation": "SiLU", "SiLU": "SiLU"}


class UnknownModuleClass(Exception):
    pass


def role_of(cls_name, attr_name="", top_level=False):
    """The role of a module, from its class (and, for two ambiguous classes, its attribute name)."""
    if cls_name not in CLASS_KINDS:
        raise UnknownModuleClass(
            f"The module class {cls_name} (at {attr_name or 'the top'}) is not in modeldiagram/kinds.py. "
            f"Say what it is there (an op kind, container, block, part or skip) and run again.")
    role = CLASS_KINDS[cls_name]
    if role == "embed" and attr_name in POSITION_NAMES:
        return "position"
    if role == "linear" and attr_name in UNEMBED_NAMES and top_level:
        return "unembed"
    return role

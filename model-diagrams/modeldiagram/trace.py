"""One forward pass with hooks, kept as a tree of calls.

The same pass serves both commands: on the meta device (no weights, every tensor a shape only) it gives the
spec; on real weights it gives the numbers, and because the same code names the tensors, the names agree.

Two things happen inside an op without a module of their own, so the functions are wrapped where the model's
own file calls them: the attention function (to keep the queries and keys after any rotation, and the
attention weights), and the running-memory function (to keep the state it ends with).
"""
import sys
from dataclasses import dataclass, field

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_map

from .kinds import role_of


@dataclass
class Call:
    path: str
    cls: str
    module: object
    inp: object = None       # the first tensor going in
    out: object = None       # the first tensor coming out
    kids: list = field(default_factory=list)
    caught: list = field(default_factory=list)   # (what, dict) from wrapped functions
    roots: bool = False

    @property
    def attr(self):
        return self.path.rsplit(".", 1)[-1]


def first_tensor(x):
    if torch.is_tensor(x):
        return x
    if isinstance(x, dict):
        x = list(x.values())
    if isinstance(x, (tuple, list)):
        for y in x:
            t = first_tensor(y)
            if t is not None:
                return t
    if hasattr(x, "last_hidden_state"):
        return x.last_hidden_state
    return None


class MetaCoerce(TorchDispatchMode):
    """On the meta device, a model's small helper tensors (indices worked out from the picture's grid) are
    still made on the CPU. When one op mixes the two, move the CPU ones to meta."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        flat, _ = tree_flatten((args, kwargs))
        if any(torch.is_tensor(t) and t.is_meta for t in flat):
            mv = lambda t: t.to("meta") if torch.is_tensor(t) and not t.is_meta and t.dim() > 0 else t
            args, kwargs = tree_map(mv, args), tree_map(mv, kwargs)
        return func(*args, **kwargs)


class Tracer:
    def __init__(self, model):
        self.model = model
        self.stack = []
        self.roots = []
        self.handles = []
        self.restore = []
        names = dict(model.named_modules())
        for path, mod in names.items():
            role_of(type(mod).__name__, path.rsplit(".", 1)[-1], top_level=path.count(".") == 0)  # stops on an unknown class
            self.handles.append(mod.register_forward_pre_hook(self._pre(path, mod), with_kwargs=True))
            self.handles.append(mod.register_forward_hook(self._post(), with_kwargs=True))
        for modname in {type(m).__module__ for m in names.values()}:
            self._wrap(sys.modules[modname])

    def _pre(self, path, mod):
        def hook(m, args, kwargs):
            c = Call(path, type(m).__name__, m, inp=first_tensor(args) if first_tensor(args) is not None else first_tensor(kwargs))
            (self.stack[-1].kids if self.stack else self.roots).append(c)
            self.stack.append(c)
        return hook

    def _post(self):
        def hook(m, args, kwargs, out):
            c = self.stack.pop()
            c.out = first_tensor(out)
        return hook

    def _catch(self, what, d):
        if self.stack:
            self.stack[-1].caught.append((what, d))

    def _wrap(self, mod):
        if hasattr(mod, "eager_attention_forward"):
            orig = mod.eager_attention_forward

            def eager(module, query, key, value, attention_mask, *a, _o=orig, **k):
                out, w = _o(module, query, key, value, attention_mask, *a, **k)
                scaling = k.get("scaling", a[0] if a else None)
                self._catch("attention", dict(query=query, key=key, weights=w, scaling=scaling))
                return out, w
            mod.eager_attention_forward = eager
            self.restore.append((mod, "eager_attention_forward", orig))
        for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
            if hasattr(mod, name):
                orig = getattr(mod, name)

                def delta(query, key, value, *a, _o=orig, **k):
                    keep = k.get("output_final_state", False)
                    k["output_final_state"] = True
                    out, state = _o(query, key, value, *a, **k)
                    self._catch("memory", dict(query=query, key=key, value=value, state=state, out=out))
                    return out, (state if keep else None)
                setattr(mod, name, delta)
                self.restore.append((mod, name, orig))

    def close(self):
        for h in self.handles:
            h.remove()
        for mod, name, orig in self.restore:
            setattr(mod, name, orig)


def load(hf_id, meta=True, revision=None):
    """The model with its config, on the meta device or with real weights on the CPU, and its kind."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText
    cfg = AutoConfig.from_pretrained(hf_id, revision=revision)
    kind = "vlm" if getattr(cfg, "vision_config", None) is not None else "llm"
    cls = AutoModelForImageTextToText if kind == "vlm" else AutoModelForCausalLM
    if meta:
        with torch.device("meta"):
            model = cls.from_config(cfg, attn_implementation="eager")
    else:
        model = cls.from_pretrained(hf_id, revision=revision, attn_implementation="eager", torch_dtype=torch.float32)
    return cfg, kind, model.eval()


def run_llm(model, ids):
    """One pass of a language model; ids is (batch, tokens)."""
    tr = Tracer(model)
    try:
        with torch.no_grad():
            model(input_ids=ids)
    finally:
        tr.close()
    return tr.roots


def run_vlm_meta(model, cfg, n_tokens, grid_thw, image_at):
    """One pass of a vision-language model on the meta device: the vision tower on a picture of grid_thw
    patches, the token table on n_tokens tokens, and the language model on the joined rows. The picture's
    rows go in at image_at. The join itself is plain indexing, so it is added by the builder, not traced."""
    vc = cfg.vision_config
    t, h, w = grid_thw
    n_patches = t * h * w
    patch_pixels = vc.in_channels * vc.temporal_patch_size * vc.patch_size ** 2
    inner = model.model
    tr = Tracer(model)
    try:
        with torch.no_grad(), MetaCoerce():
            inner.visual(torch.zeros(n_patches, patch_pixels, device="meta"), grid_thw=torch.tensor([grid_thw]))
            emb = inner.language_model.embed_tokens(torch.zeros(1, n_tokens, dtype=torch.long, device="meta"))
            pos = torch.arange(n_tokens).view(1, 1, n_tokens).expand(4 if getattr(cfg.text_config, "rope_parameters", None) else 3, 1, n_tokens)
            hid = inner.language_model(inputs_embeds=emb, position_ids=pos.to("meta")).last_hidden_state
            model.lm_head(hid)
    finally:
        tr.close()
    return tr.roots, dict(n_patches=n_patches, patch_pixels=patch_pixels, image_at=image_at)

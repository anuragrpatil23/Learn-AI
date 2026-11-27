"""From a traced pass to the format: axes, tensors, ops, blocks, the stream and the steps.

Every size comes from a tensor of the pass. Each op's reader below names the axes of what it makes; when one
axis name is given two sizes, the run stops, since then the reader is wrong about the model.

Naming, which the engine relies on:
  an op's output                 <op>.out
  a part inside an op            <op>.<part>          (attn.q, attn.scores, attn.weights, attn.mixed, mlp.act, ...)
  a weight                       <op>[.<part>].weight and .bias   (a weight is stored (out, in): row j is neuron j)
"""
import re

import torch

from .kinds import ACT_NAMES, role_of
from . import texts


class SpecError(Exception):
    pass


class Builder:
    def __init__(self, kind, hf_id):
        self.kind = kind
        self.hf_id = hf_id
        self.axes = {}
        self.tensors = {}       # id -> dict(id, axes, role, note)
        self.data = {}          # id -> torch tensor (meta on a spec pass, real on a values pass)
        self.ops = []
        self.blocks = []
        self.streams = {}       # tower -> [tensor ids]
        self.tower = ""
        self.cur = None         # the stream's tensor right now
        self.fill = []          # tensors made by plain arithmetic, filled from the next op's input
        self.notes = {}         # op id -> extra facts (act name, gated) for the yaml
        self.facts = {}         # numbers for the step sentences
        self.ids_used = set()

    # ---------- axes and tensors ----------
    def axis(self, name, size):
        size = int(size)
        if name in self.axes and self.axes[name] != size:
            raise SpecError(f"The axis {name} is {self.axes[name]} in one place and {size} in another; "
                            f"the reader that named it is wrong about this model.")
        self.axes[name] = size
        return name

    def tensor(self, tid, t, axes, role="activation", note=None, data=None):
        if t is None:
            raise SpecError(f"No tensor was caught for {tid}.")
        shape = list(t.shape)
        if len(axes) != len(shape):
            raise SpecError(f"{tid} has shape {shape} but {len(axes)} axis names {axes}.")
        named = []
        for a, n in zip(axes, shape):
            named.append(self.axis(a, n) if isinstance(a, str) else int(n))
            if not isinstance(a, str) and int(a) != n:
                raise SpecError(f"{tid}: size {n} where {a} was expected.")
        if tid in self.tensors:
            raise SpecError(f"The tensor id {tid} is made twice.")
        d = dict(id=tid, axes=named, role=role)
        if note:
            d["note"] = note
        self.tensors[tid] = d
        self.data[tid] = t.detach() if data is None else data.detach()
        return tid

    def weight(self, tid, w, axes, note=None):
        return self.tensor(tid, w, axes, role="weight", note=note)

    def op(self, oid, kind, ins, outs, weights=(), block=None, note=None, **extra):
        if oid in self.ids_used:
            raise SpecError(f"The op id {oid} is used twice.")
        self.ids_used.add(oid)
        o = dict(id=oid, kind=kind, **{"in": list(ins)}, out=list(outs), weights=list(weights))
        if block:
            o["block"] = block
        if note:
            o["note"] = note
        o.update(extra)
        self.ops.append(o)
        return o

    def oid(self, base):
        """An op id for an op outside the blocks: the tower's prefix, then the kind, numbered if repeated."""
        stem = self.prefix + base
        if stem not in self.ids_used:
            return stem
        n = 2
        while f"{stem}{n}" in self.ids_used:
            n += 1
        return f"{stem}{n}"

    @property
    def prefix(self):
        return {"vision": "vis_"}.get(self.tower, "")

    def ax(self, role):
        """The axis name for a role in the current tower: the vision tower has its own width and heads."""
        return ("v_" + role) if self.tower == "vision" else role

    def rows(self):
        """The leading axes of an activation: (batch, tokens) in a language model, patches in the vision tower."""
        return ["patches"] if self.tower == "vision" else ["batch", "tokens"]

    def stream_to(self, tid):
        self.cur = tid
        self.streams.setdefault(self.tower, []).append(tid)

    def filled_by(self, call):
        """A tensor made by plain arithmetic (an add, a join) gets its numbers from the next module's input."""
        for tid in self.fill:
            if call.inp is not None and list(call.inp.shape) == list(self.data[tid].shape):
                self.data[tid] = call.inp.detach()
        self.fill = []

    def act_tensor(self, tid, t, last, note=None):
        return self.tensor(tid, t, self.rows() + [last], note=note)

    # ---------- walking the tree ----------
    def walk(self, calls):
        calls = list(calls)
        i = 0
        while i < len(calls):
            c = calls[i]
            i += 1
            role = role_of(c.cls, c.attr, top_level=bool(c.path) and c.path.count(".") == 0)
            if role == "block":
                # a ModuleList is never called itself: its blocks are a run of calls side by side
                run = [c]
                while i < len(calls) and role_of(calls[i].cls) == "block":
                    run.append(calls[i])
                    i += 1
                self.block_list(run)
                continue
            if role == "skip":
                if "Rotary" in c.cls:
                    self.facts.setdefault("rotary", set()).add(self.tower)
                continue
            if role == "container":
                self.walk(c.kids)
                continue
            if role == "part":
                raise SpecError(f"{c.cls} at {c.path} is a part, but it was called outside any op.")
            self.filled_by(c)
            self.gap(c)
            getattr(self, "do_" + role)(c)

    def gap(self, c):
        """When the next module's input is the stream re-cut into fewer, wider rows, that is a join."""
        if self.cur is None or c.inp is None:
            return
        have = self.data[self.cur]
        if list(have.shape) == list(c.inp.shape):
            return
        if have.numel() == c.inp.numel() and have.shape[-1] < c.inp.shape[-1]:
            k = c.inp.shape[-1] // have.shape[-1]
            oid = self.oid("join")
            out = self.tensor(f"{oid}.out", c.inp, [self.ax("merged"), self.ax("joined_width")],
                              note=f"each group of {k} neighbouring rows put side by side as one row")
            self.op(oid, "join", [self.cur], [out], note=f"{k} neighbouring patches become one row")
            self.facts[oid] = dict(k=k)
            self.cur = out

    # ---------- the ops outside the blocks ----------
    def do_embed(self, c):
        oid = self.oid("embed")
        w = self.weight(f"{oid}.weight", c.module.weight, ["vocab", "width"], note="one row per token of the vocabulary")
        self.op(oid, "embed", [self.cur or "ids"], [self.tensor(f"{oid}.out", c.out, ["batch", "tokens", "width"])], [w])
        self.embed_weight = c.module.weight
        self.embed_id = w
        self.stream_to(f"{oid}.out")

    def do_position(self, c):
        oid = self.oid("position")
        w = self.weight(f"{oid}.weight", c.module.weight, ["positions" if self.tower != "vision" else "v_positions", self.ax("width")],
                        note="one learnt row per place" + (", blended from the four nearest for each patch" if self.tower == "vision" else ""))
        src = self.data[self.cur]
        out = self.tensor(f"{oid}.out", src, self.rows() + [self.ax("width")], note="the rows with their place added")
        self.fill.append(out)
        blend = next((d["weights"] for what, d in c.caught if what == "blend"), None)
        if blend is not None and c.out.dim() == 3:
            # a patch's place row is the four nearest table rows mixed, exactly as the model adds it
            rows = (c.out * blend[:, :, None].to(c.out.device, c.out.dtype)).sum(1).to(src.dtype)
            self.tensor(f"{oid}.rows", rows, self.rows() + [self.ax("width")], note="each patch's place row, blended from the four nearest")
        elif c.out.dim() == src.dim() and list(c.out.shape[1:]) == list(src.shape[1:]):
            # the place rows on their own, since the output is already the sum
            self.tensor(f"{oid}.rows", c.out.expand_as(src), self.rows() + [self.ax("width")], note="each token's row of the place table")
        self.op(oid, "position", [self.cur], [out], [w])
        # the stream starts once the places are in: the rows before are the table's alone
        if self.streams.get(self.tower) and self.streams[self.tower][-1] == self.cur:
            self.streams[self.tower].pop()
        self.stream_to(out)

    def do_patchify(self, c):
        oid = self.oid("patchify")
        conv = next(k for k in c.module.modules() if k is not c.module and hasattr(k, "weight"))
        wshape = list(conv.weight.shape)
        w = self.weight(f"{oid}.weight", conv.weight, [self.ax("width")] + wshape[1:],
                        note=f"one filter per output number, over {'x'.join(map(str, wshape[1:]))} pixels")
        ws = [w]
        if getattr(conv, "bias", None) is not None:
            ws.append(self.weight(f"{oid}.bias", conv.bias, [self.ax("width")]))
        self.tensor("pixels", c.inp, ["patches", "patch_pixels"], note="each patch's pixels in one row")
        self.op(self.oid("picture_in"), "input", [], ["pixels"], note="the picture, cut into patches")
        self.op(oid, "patchify", ["pixels"], [self.act_tensor(f"{oid}.out", c.out, self.ax("width"))], ws)
        self.stream_to(f"{oid}.out")

    def do_norm(self, c, oid=None, block=None):
        oid = oid or self.oid("norm")
        ws = [self.weight(f"{oid}.weight", c.module.weight, [self.ax("width")])]
        if getattr(c.module, "bias", None) is not None:
            ws.append(self.weight(f"{oid}.bias", c.module.bias, [self.ax("width")]))
        src = self.cur
        out = self.tensor(f"{oid}.out", c.out, self.rows() + [self.ax("width")])
        rms = "RMS" in c.cls
        self.op(oid, "norm", [src], [out], ws, block=block,
                note="scaled by the root of its mean square" if rms else "centred, then scaled to spread one")
        self.facts[oid] = dict(rms=rms)
        if block is None:
            self.cur = out
        return out

    def do_linear(self, c, oid=None):
        oid = oid or self.oid("linear")
        n_out = c.out.shape[-1]
        if self.tower == "vision":
            # inside the hand-over: the joined width, then the language model's width (the join checks it)
            name = "v_joined_width" if n_out == self.axes.get("v_joined_width") else "width"
        else:
            name = "width" if n_out == self.axes.get("width") else "mlp_width"
        w = self.weight(f"{oid}.weight", c.module.weight, [name, self.tensors[self.cur]["axes"][-1]])
        ws = [w] + ([self.weight(f"{oid}.bias", c.module.bias, [name])] if c.module.bias is not None else [])
        out = self.tensor(f"{oid}.out", c.out, self.tensors[self.cur]["axes"][:-1] + [name])
        self.op(oid, "linear", [self.cur], [out], ws)
        self.cur = out

    def do_act(self, c):
        oid = self.oid("act")
        out = self.tensor(f"{oid}.out", c.out, self.tensors[self.cur]["axes"])
        self.op(oid, "act", [self.cur], [out], act=ACT_NAMES.get(c.cls, c.cls))
        self.cur = out

    def do_unembed(self, c):
        oid = "unembed"
        tied = getattr(self, "embed_weight", None) is not None and (c.module.weight is self.embed_weight)
        if tied:
            ws = [self.embed_id]
        else:
            ws = [self.weight(f"{oid}.weight", c.module.weight, ["vocab", "width"], note="one row per token of the vocabulary")]
        out = self.tensor(f"{oid}.out", c.out, ["batch", "tokens", "vocab"], note="a score for every token")
        self.op(oid, "unembed", [self.cur], [out], ws,
                note="the same table as the token table, used the other way" if tied else "its own table, not the token table")
        probs = self.tensor("output.out", torch.softmax(c.out.float(), -1) if not c.out.is_meta else c.out, ["batch", "tokens", "vocab"],
                            note="the scores turned into chances that add up to one")
        self.op("output", "output", [out], [probs], note="softmax")
        # what the drawing shows of the scores and chances: every row's value for the last row's top tokens
        self.tensor("unembed.top", c.out, ["batch", "tokens", "vocab"],
                    note="the scores, at the columns of the last row's top 8 tokens; values.labels names them")
        self.tensor("output.top", self.data[probs], ["batch", "tokens", "vocab"],
                    note="the chances, at the same 8 columns")
        self.tensor("unembed.top.weight", c.module.weight, ["vocab", "width"], role="weight",
                    note="the rows of the table for those 8 tokens")
        self.facts["unembed"] = dict(tied=tied)

    # ---------- the blocks ----------
    def block_list(self, run):
        sig = lambda b: tuple(k.cls for k in b.kids)
        kinds = []
        for b in run:
            if sig(b) not in kinds:
                kinds.append(sig(b))
        n = len(run)
        where = {s: [i + 1 for i, b in enumerate(run) if sig(b) == s] for s in kinds}
        names = {}
        for s in kinds:
            mixer = next((role_of(k) for k in s if role_of(k) in ("attention", "memory")), "mlp")
            base = {"vision": "vis"}.get(self.tower, "")
            names[s] = (base if len(kinds) == 1 else (base + "_" if base else "") + {"attention": "att", "memory": "mem"}[mixer])
        drawn = set()
        last_drawn = None
        for i, b in enumerate(run):
            s = sig(b)
            if s in drawn:
                continue
            if last_drawn is not None and last_drawn != i - 1:
                # the stream has gone through blocks that are not drawn; it enters this one as its own tensor
                tid = self.tensor(f"{names[s] or 'block'}_in", b.inp, self.rows() + [self.ax("width")],
                                  note=f"the stream entering block {i + 1}")
                self.stream_to(tid)
            self.filled_by(b)
            bid = (names[s] or "block")
            pat = texts.pattern(where[s], n) if len(kinds) > 1 else None
            blk = dict(id=bid, ops=[], repeat=len(where[s]))
            if pat:
                blk["pattern"] = pat
            self.blocks.append(blk)
            self.one_block(b, bid, names[s])
            self.facts[bid] = dict(first=i + 1, count=len(where[s]), total=n, mixer=next((role_of(k) for k in s if role_of(k) in ("attention", "memory")), None))
            drawn.add(s)
            last_drawn = i
        if last_drawn != n - 1:
            tid = self.tensor(f"{self.prefix}after_blocks", run[-1].out, self.rows() + [self.ax("width")],
                              note=f"the stream after all {n} blocks")
            self.stream_to(tid)

    def one_block(self, b, bid, pre):
        p = (pre + "_") if pre else ""
        x = self.cur
        norms = adds = 0
        start = len(self.ops)
        h = None
        for k in b.kids:
            role = role_of(k.cls, k.attr)
            if role == "skip":
                continue
            self.filled_by(k)
            if role == "norm":
                norms += 1
                h = self.do_norm(k, oid=f"{p}norm{norms}", block=bid)
            elif role in ("attention", "memory", "mlp"):
                if h is None:
                    raise SpecError(f"In {b.path}, {k.cls} does not follow a norm: only pre-norm blocks are read so far.")
                name = {"attention": "attn", "memory": "memory", "mlp": "mlp"}[role]
                o = getattr(self, "read_" + role)(k, f"{p}{name}", h, bid)
                adds += 1
                aid = f"{p}add{adds}"
                out = self.tensor(f"{aid}.out", self.data[x], self.tensors[x]["axes"],
                                  note="the stream with what " + {"attn": "attention", "memory": "the memory", "mlp": "the MLP"}[name] + " found added")
                self.fill.append(out)
                self.op(aid, "add", [x, o], [out], block=bid)
                self.stream_to(out)
                x, h = out, None
            else:
                raise SpecError(f"In {b.path}, {k.cls} is a {role}; a block is read as norm, attention or memory, add, norm, MLP, add.")
        if b.out is not None:
            self.data[x] = b.out.detach()
            self.fill = [t for t in self.fill if t != x]
        self.blocks[-1]["ops"] = [o["id"] for o in self.ops[start:]]

    # ---------- readers for the ops inside a block ----------
    def lin_w(self, tid, w, b, rows, out_axis, in_axis, note=None):
        """A weight given (out, in), and its bias, for one part of a layer (rows picks the part's neurons)."""
        ws = [self.weight(f"{tid}.weight", w[rows] if rows is not None else w, [out_axis, in_axis], note=note)]
        if b is not None:
            ws.append(self.weight(f"{tid}.bias", b[rows] if rows is not None else b, [out_axis]))
        return ws

    def kid(self, c, attr):
        for k in c.kids:
            if k.attr == attr:
                return k
        raise SpecError(f"{c.cls} at {c.path} has no call to {attr}; its reader in build.py needs updating.")

    def read_attention(self, c, oid, h, bid):
        cap = [d for w, d in c.caught if w == "attention"]
        if len(cap) != 1:
            raise SpecError(f"{c.path}: expected one call of the attention function, caught {len(cap)}.")
        cap = cap[0]
        q4, k4, wts = cap["query"], cap["key"], cap["weights"]     # (batch, heads, tokens, head_width)
        if q4.dim() == 3:
            q4, k4, wts = q4[None], k4[None], wts if wts.dim() == 4 else wts[None]
        H, KV, hw = q4.shape[1], k4.shape[1], q4.shape[-1]
        W, R = self.ax("width"), self.rows()
        self.axis(self.ax("heads"), H)
        self.axis(self.ax("head_width"), hw)
        if KV != H:
            self.axis("kv_heads", KV)
        mod = c.module
        ws, parts = [], {}
        C = self.axes[W]
        if c.cls == "GPT2Attention":
            qkv = self.kid(c, "c_attn")
            Wt, b = qkv.module.weight.T, qkv.module.bias          # Conv1D keeps (in, out)
            for i, nm in enumerate("qkv"):
                parts[nm] = qkv.out[..., i * C:(i + 1) * C]
                ws += self.lin_w(f"{oid}.{nm}", Wt, b, slice(i * C, (i + 1) * C), W, W, note=f"columns {i * C} to {(i + 1) * C - 1} of c_attn")
            proj = self.kid(c, "c_proj")
            pw, pb = proj.module.weight.T, proj.module.bias
        elif c.cls == "Qwen3_5VisionAttention":
            qkv = self.kid(c, "qkv")
            for i, nm in enumerate("qkv"):
                parts[nm] = qkv.out[..., i * C:(i + 1) * C]
                ws += self.lin_w(f"{oid}.{nm}", qkv.module.weight, qkv.module.bias, slice(i * C, (i + 1) * C), W, W, note=f"rows {i * C} to {(i + 1) * C - 1} of qkv")
            proj = self.kid(c, "proj")
            pw, pb = proj.module.weight, proj.module.bias
        elif c.cls == "Qwen3_5Attention":
            qp = self.kid(c, "q_proj")
            both = qp.out.view(*qp.out.shape[:-1], H, 2 * hw)
            parts["q"] = both[..., :hw].reshape(*qp.out.shape[:-1], H * hw)
            parts["gate"] = both[..., hw:].reshape(*qp.out.shape[:-1], H * hw)
            qrows = torch.cat([torch.arange(j * 2 * hw, j * 2 * hw + hw) for j in range(H)])
            grows = qrows + hw
            ws += self.lin_w(f"{oid}.q", qp.module.weight, qp.module.bias, qrows, "q_width", W, note="the query half of q_proj (each head's first 256 rows)")
            ws += self.lin_w(f"{oid}.gate", qp.module.weight, qp.module.bias, grows, "q_width", W, note="the gate half of q_proj (each head's second 256 rows)")
            for nm in "kv":
                kp = self.kid(c, f"{nm}_proj")
                parts[nm] = kp.out
                ws += self.lin_w(f"{oid}.{nm}", kp.module.weight, kp.module.bias, None, "kv_width", W)
            for nm in "qk":
                nk = self.kid(c, f"{nm}_norm")
                ws.append(self.weight(f"{oid}.{nm}_norm.weight", nk.module.weight, ["head_width"], note=f"each head's {'query' if nm == 'q' else 'key'} is scaled by its root mean square, then by these"))
            proj = self.kid(c, "o_proj")
            pw, pb = proj.module.weight, proj.module.bias
        else:
            raise SpecError(f"There is no reader for the attention class {c.cls} in build.py.")
        split = c.cls == "Qwen3_5Attention"   # queries and keys have widths of their own
        widths = {"q": "q_width" if split else W, "gate": "q_width", "k": "kv_width" if split else W, "v": "kv_width" if split else W}
        for nm, t in parts.items():
            self.tensor(f"{oid}.{nm}", t, R + [widths[nm]],
                        note={"q": "the queries: what each token is looking for", "k": "the keys: what each token offers",
                              "v": "the values: what each token passes on", "gate": "how much of each head's result to let through"}[nm])
        # the similarity of every query with every key, and the weights made from it
        scale = cap["scaling"] if cap["scaling"] is not None else hw ** -0.5
        kk = k4.repeat_interleave(H // KV, dim=1) if KV != H else k4
        scores = (q4.float() @ kk.float().transpose(-1, -2)) * scale
        causal = bool(getattr(mod, "is_causal", False))
        if causal and not scores.is_meta:
            T = scores.shape[-1]
            scores = scores.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool, device=scores.device), 1), float("-inf"))
        sq = (["batch"] if self.tower != "vision" else [1]) + [self.ax("heads"), R[-1], R[-1]]
        self.tensor(f"{oid}.scores", scores, sq, note="each query against each key" + (", the later keys masked" if causal else ""))
        self.tensor(f"{oid}.weights", wts, sq, note="each row of scores turned into weights that add up to one")
        mixed = proj.inp
        self.tensor(f"{oid}.mixed", mixed, R + [widths["q"]],
                    note="each token's mix of values, the heads side by side" + (", after the gate" if "gate" in parts else ""))
        ws += self.lin_w(f"{oid}.out", pw, pb, None, W, widths["q"])
        out = self.tensor(f"{oid}.out", c.out, R + [W], note="what attention adds to the stream")
        extra = dict(mask="causal" if causal else "none", heads=H)
        if KV != H:
            extra["kv_heads"] = KV
        if self.tower in self.facts.get("rotary", ()):
            extra["note"] = "queries and keys are turned by their place (rotary) before they are compared"
        self.op(oid, "attention", [h], [out], ws, block=bid, **extra)
        self.facts[oid] = dict(heads=H, kv_heads=KV, head_width=hw, causal=causal, gated="gate" in parts)
        return out

    def read_memory(self, c, oid, h, bid):
        if c.cls != "Qwen3_5GatedDeltaNet":
            raise SpecError(f"There is no reader for the memory class {c.cls} in build.py.")
        cap = [d for w, d in c.caught if w == "memory"]
        if len(cap) != 1:
            raise SpecError(f"{c.path}: expected one call of the memory function, caught {len(cap)}.")
        cap = cap[0]
        m, R, W = c.module, self.rows(), "width"
        Hk, dk = cap["query"].shape[2], cap["query"].shape[3]
        Hv, dv = cap["value"].shape[2], cap["value"].shape[3]
        self.axis("mem_key_heads", Hk); self.axis("mem_heads", Hv)
        self.axis("mem_key", dk); self.axis("mem_value", dv)
        qkv, z, bb, aa = (self.kid(c, n) for n in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"))
        KD, VD = Hk * dk, Hv * dv
        flat = lambda t: t.reshape(*t.shape[:2], -1)
        ws = []
        for nm, t, rows, ax in (("q", cap["query"], slice(0, KD), "mem_key_width"), ("k", cap["key"], slice(KD, 2 * KD), "mem_key_width"),
                                ("v", cap["value"], slice(2 * KD, 2 * KD + VD), "mem_value_width")):
            self.tensor(f"{oid}.{nm}", flat(t), R + [ax], note={"q": "what each token reads from the memory",
                        "k": "where in the memory each token writes", "v": "what each token writes"}[nm] + ", after a short blur over the last few tokens")
            ws += self.lin_w(f"{oid}.{nm}", qkv.module.weight, None, rows, ax, W, note=f"rows {rows.start} to {rows.stop - 1} of in_proj_qkv")
        ws.append(self.weight(f"{oid}.conv.weight", m.conv1d.weight, ["mem_conv_width", 1, m.conv1d.weight.shape[-1]],
                              note=f"a blur over the last {m.conv1d.weight.shape[-1]} tokens, one per channel of q, k and v"))
        self.tensor(f"{oid}.gate", z.out, R + ["mem_value_width"], note="how much of each head's reading to let through")
        ws += self.lin_w(f"{oid}.gate", z.module.weight, None, None, "mem_value_width", W)
        self.tensor(f"{oid}.write", bb.out, R + ["mem_heads"], note="how strongly each token writes, per head (before the sigmoid)")
        ws += self.lin_w(f"{oid}.write", bb.module.weight, None, None, "mem_heads", W)
        self.tensor(f"{oid}.fade", aa.out, R + ["mem_heads"], note="how much the memory fades at each token, per head (before the softplus)")
        ws += self.lin_w(f"{oid}.fade", aa.module.weight, None, None, "mem_heads", W)
        ws.append(self.weight(f"{oid}.fade_scale", m.A_log, ["mem_heads"], note="A_log: each head's fading rate, as a log"))
        ws.append(self.weight(f"{oid}.fade_bias", m.dt_bias, ["mem_heads"]))
        ws.append(self.weight(f"{oid}.norm.weight", m.norm.weight, ["mem_value"], note="each head's reading is scaled by its root mean square, then by these"))
        self.tensor(f"{oid}.state", cap["state"], ["batch", "mem_heads", "mem_key", "mem_value"],
                    note="the running memory after the last token: one square of numbers per head")
        op_ = self.kid(c, "out_proj")
        self.tensor(f"{oid}.mixed", op_.inp, R + ["mem_value_width"], note="each token's reading from the memory, normed and gated")
        ws += self.lin_w(f"{oid}.out", op_.module.weight, None, None, W, "mem_value_width")
        out = self.tensor(f"{oid}.out", c.out, R + [W], note="what the memory adds to the stream")
        self.axis("mem_conv_width", m.conv1d.weight.shape[0])
        self.op(oid, "memory", [h], [out], ws, block=bid, state=["batch", "mem_heads", "mem_key", "mem_value"])
        self.facts[oid] = dict(heads=Hv, key_heads=Hk, key=dk, value=dv, conv=m.conv1d.weight.shape[-1])
        return out

    def read_mlp(self, c, oid, h, bid):
        W, R = self.ax("width"), self.rows()
        lins = [k for k in c.kids if role_of(k.cls) in ("linear", "part") and hasattr(k.module, "weight")]
        acts = [k for k in c.kids if role_of(k.cls) == "act"]
        if len(acts) != 1 or len(lins) not in (2, 3):
            raise SpecError(f"{c.path}: an MLP is read as two layers (up, down) or three (gate, up, down) with one bend; "
                            f"found {len(lins)} layers and {len(acts)} bends.")
        tw = lambda k: k.module.weight.T if k.cls == "Conv1D" else k.module.weight
        M = self.ax("mlp_width")
        ws = []
        gated = len(lins) == 3
        if gated:
            g, u, d = (self.kid(c, n) for n in ("gate_proj", "up_proj", "down_proj"))
            self.tensor(f"{oid}.pre", g.out, R + [M], note="the gate layer's sums, before the bend")
            self.tensor(f"{oid}.up", u.out, R + [M], note="the up layer's sums")
            ws += self.lin_w(f"{oid}.gate", tw(g), g.module.bias, None, M, W)
            ws += self.lin_w(f"{oid}.up", tw(u), u.module.bias, None, M, W)
        else:
            u, d = lins
            self.tensor(f"{oid}.pre", u.out, R + [M], note="the first layer's sums, before the bend")
            ws += self.lin_w(f"{oid}.up", tw(u), u.module.bias, None, M, W)
        self.tensor(f"{oid}.act", d.inp, R + [M], note="after the bend" + (", times the up layer" if gated else ""))
        ws += self.lin_w(f"{oid}.down", tw(d), d.module.bias, None, W, M)
        out = self.tensor(f"{oid}.out", c.out, R + [W], note="what the MLP adds to the stream")
        act = ACT_NAMES.get(acts[0].cls, acts[0].cls)
        self.op(oid, "mlp", [h], [out], ws, block=bid, act=act, **({"gated": True} if gated else {}))
        self.facts[oid] = dict(act=act, gated=gated)
        return out

    # ---------- the model as a whole ----------
    def spec(self, source):
        main = "language" if self.kind == "vlm" else ""
        stream = self.streams.get(main, [])
        self.ops = sort_inputs_first(self.ops)
        for o in self.ops:      # a plain name right after the id, for the step strip and the labels
            nm = texts.name(o, self)
            o_items = list(o.items())
            o.clear(); o.update([o_items[0], ("name", nm)] + o_items[1:])
        for blk in self.blocks:
            nm = texts.block_name(blk, self)
            items = list(blk.items())
            blk.clear(); blk.update([items[0], ("name", nm)] + items[1:])
        spec = dict(source=source, axes=dict(self.axes), tensors=list(self.tensors.values()), ops=self.ops,
                    blocks=self.blocks, stream=stream, steps=texts.steps(self))
        others = {t: ids for t, ids in self.streams.items() if t != main and ids}
        if others:
            spec["streams"] = others      # the other towers' streams, each drawn as its own lane
        return spec


def sort_inputs_first(ops):
    """Input ops are made when the tracer first sees what they feed; put each one first in its tower."""
    return [o for o in ops if o["kind"] == "input"] + [o for o in ops if o["kind"] != "input"]


def build_llm(roots, hf_id, ids):
    b = Builder("llm", hf_id)
    b.tensor("ids", ids, ["batch", "tokens"], note="the tokens, each a number")
    b.op("tokens_in", "input", [], ["ids"], note="the text, cut into tokens")
    b.walk(roots)
    return b


def build_vlm(roots, hf_id, ids, at):
    """roots: the vision tower's call, the token table's, the language model's, the unembedding's."""
    b = Builder("vlm", hf_id)
    vis, emb, lang, head = roots
    b.tower = "vision"
    b.walk([vis])
    vis_tokens = b.cur
    b.tower = "language"
    b.cur = None
    b.tensor("ids", ids, ["batch", "tokens"], note="the tokens, each a number; the picture's rows are placeholders")
    b.op("tokens_in", "input", [], ["ids"], note="the text with one placeholder token per picture row")
    b.walk([emb])
    # the rows the language model gets, text and picture joined: what its first block reads
    first_block = next((k for k in lang.kids if role_of(k.cls) == "block"), None)
    if first_block is None or first_block.inp is None:
        raise SpecError("The language model's first block was not caught, so the joined rows are unknown.")
    data = first_block.inp
    j = b.tensor("join.out", data, ["batch", "tokens", "width"], note=f"the picture's {b.axes['v_merged']} rows put in place of its placeholders, from token {at['image_at']}")
    b.op("join", "join", [b.cur, vis_tokens], [j], note="the picture's rows go in among the words")
    b.facts["join"] = dict(at=at["image_at"], rows=b.axes["v_merged"])
    if b.streams.get("language") == [b.cur]:
        b.streams["language"].pop()        # the stream starts once the picture is in
    b.stream_to(j)
    b.walk([k for k in lang.kids if k.cls != "Embedding"])
    b.walk([head])
    return b

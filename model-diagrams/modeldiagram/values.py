"""modeldiagram values: real numbers for one input, in two steps, as the format asks.

1. The run. The model, with its real weights, writes its reply to the input (greedy), and then one pass
   over the input and the reply is traced with the same code as the spec, so every tensor has the same id.
   Every tensor is kept, whole, in runs/<run id>/dump.pt. That file is the raw dump.
2. The trim. From the dump, values/<input>.json keeps only what the drawing uses: the first rows (up to 32)
   and the columns the engine draws of each tensor (8 of 768, 16 of 3,072), at 3 significant digits, every
   head's attention map, row lengths, and the reply. It records the run id
   and the dump's sha256, so every number traces back to the run.
"""
import datetime
import hashlib
import json
import math
import pathlib

import torch

N = 32           # rows and columns kept per tensor
TOP = 8          # tokens shown at the score and chance steps
MAX_ROWS = 512   # row lengths are sampled above this


def r4(x):
    """A number at the precision the drawing prints: 3 significant digits."""
    x = float(x)
    if math.isinf(x) or math.isnan(x):
        return -1e9 if x < 0 else None
    return float(f"{x:.3g}")


def drawn(size):
    """How many columns the engine draws for an axis of this size (modelDiagramColumnsBlock in Thinking Space):
    768 gives 8, 3,072 gives 16, never more than 32. The values keep that many and no more."""
    if size <= 32:
        return max(1, size)
    return max(4, min(32, 2 * round((4 * math.log2(size) - 30.34) / 2)))


def keep(rows, shape, role):
    """Cut a slice's rows to what is drawn: an activation keeps its rows (up to 32) and the drawn columns;
    a weight (out, in) keeps the drawn neurons of its output and the drawn columns of its input."""
    if not rows:
        return rows
    flat = [int(x) for x in shape]
    n_in = flat[-1] if len(flat) < 3 or role == "activation" else math.prod(flat[1:])
    c = drawn(n_in)
    r = drawn(flat[0]) if role == "weight" and len(flat) >= 2 else N
    return [row[:c] for row in rows[:r]]


def corner(t, axes):
    """The first rows and columns of a tensor: batch 0 and head 0, then the first 32 x 32 of what is left."""
    t = t.detach().float()
    if t.dim() == 1:
        return [[r4(v) for v in t[:N]]]
    axes = list(axes)
    while t.dim() > 2 and (axes[0] in ("batch", 1) or (isinstance(axes[0], str) and axes[0].endswith("heads"))):
        t, axes = t[0], axes[1:]
    if t.dim() > 2:
        t = t.reshape(t.shape[0], -1)
    return [[r4(v) for v in row[:N]] for row in t[:N]]


def sig3(x):
    return float(f"{float(x):.3g}")


def row_lengths(t, axes):
    t = t.detach().float()
    if "tokens" in axes and t.dim() == 3:
        t = t[0]
    elif "patches" not in axes and "v_merged" not in axes:
        return None
    lens = t.norm(dim=-1).tolist()
    if len(lens) > MAX_ROWS:
        step = len(lens) / MAX_ROWS
        lens = [lens[int(i * step)] for i in range(MAX_ROWS)]
    return [sig3(v) for v in lens]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_llm_values(hf_id, input_path, folder, new_tokens, runs_dir):
    from transformers import AutoTokenizer
    from . import trace, build
    text = input_path.read_text().rstrip("\n")
    name = input_path.stem
    tok = AutoTokenizer.from_pretrained(hf_id)
    cfg, kind, model = trace.load(hf_id, meta=False)
    if kind != "llm":
        raise SystemExit("values runs language models here; a vision-language model runs on Minerva (see the README).")
    ids = tok(text, return_tensors="pt").input_ids
    with torch.no_grad():
        full = model.generate(ids, max_new_tokens=new_tokens, do_sample=False, pad_token_id=tok.eos_token_id or 0)
    roots = trace.run_llm(model, full)
    b = build.build_llm(roots, hf_id, full)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"local-{hf_id.replace('/', '_')}-{name}-{stamp}"
    run = runs_dir / run_id
    run.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in b.data.items()}, run / "dump.pt")
    (run / "run.json").write_text(json.dumps(dict(model=hf_id, input=text, new_tokens=new_tokens, tokens=full[0].tolist()), indent=1))
    out = trim(b, tok, full, ids.shape[1], run_id, sha256(run / "dump.pt"))
    (folder / "values").mkdir(parents=True, exist_ok=True)
    path = folder / "values" / f"{name}.json"
    path.write_text(json.dumps(out, separators=(",", ":"), ensure_ascii=False))
    print(f"{name}: {ids.shape[1]} tokens in, wrote {out['reply']!r}")
    print(f"raw dump {run / 'dump.pt'} (sha256 {out['sha256'][:12]}...), wrote {path} ({path.stat().st_size / 1024:.0f} KB)")
    return 0


def word(tok, i):
    s = tok.decode([i])
    return s.replace("\n", "\\n").strip() or repr(s)


def trim(b, tok, full, prompt_len, run_id, sha):
    ids = full[0].tolist()
    slices, lengths, stats = {}, {}, {}
    probs = b.data["output.out"][0]
    top = probs[prompt_len - 1].topk(TOP).indices          # the tokens the drawing names: the question's last row's best 8
    for tid, t in b.data.items():
        axes = b.tensors[tid]["axes"]
        if tid in ("unembed.top", "output.top"):
            src = b.data["unembed.out" if tid == "unembed.top" else "output.out"][0]
            rows = [[r4(v) for v in src[r, top]] for r in range(min(N, src.shape[0]))]
            slices[tid] = dict(shape=list(t.shape), rows=rows)
            continue
        slices[tid] = dict(shape=list(t.shape), rows=keep(corner(t, axes), t.shape, b.tensors[tid]["role"]))
        if b.tensors[tid]["role"] == "activation" and tid != "ids":
            rl = row_lengths(t, axes)
            if rl is not None:
                lengths[tid] = rl
    W = b.embed_weight.detach()
    cw = drawn(W.shape[1])
    slices["unembed.top.weight"] = dict(shape=list(W.shape), rows=[[r4(v) for v in W[i, :cw]] for i in top])
    for o in b.ops:
        if o["kind"] == "norm":
            x = b.data[o["in"][0]].float()
            x = x[0] if x.dim() == 3 else x
            stats[o["in"][0]] = dict(mean=[r4(v) for v in x.mean(-1)[:N]], sd=[r4(v) for v in x.std(-1, unbiased=False)[:N]],
                                     rms=[r4(v) for v in x.pow(2).mean(-1).sqrt()[:N]])
    attention = {}
    for o in b.ops:
        if o["kind"] == "attention":
            w = b.data[f"{o['id']}.weights"][0]
            attention[o["id"]] = {str(h): [[r4(v) for v in row[:N]] for row in w[h, :N]] for h in range(w.shape[0])}
    labels = [word(tok, i) for i in top.tolist()]
    # the same three slices for every pass of the reply: pass k sees the first prompt_len + k tokens,
    # and its columns are the top 8 tokens of its own last row
    passes = []
    scores = b.data["unembed.out"][0]
    for k in range(len(ids) - prompt_len + 1):
        last = prompt_len + k - 1
        tk = probs[last].topk(TOP).indices
        rows = range(min(N, last + 1))
        passes.append({"unembed.top": [[r4(v) for v in scores[r, tk]] for r in rows],
                       "output.top": [[r4(v) for v in probs[r, tk]] for r in rows],
                       "unembed.top.weight": [[r4(v) for v in W[i, :cw]] for i in tk],
                       "labels": [word(tok, i) for i in tk.tolist()]})
    return dict(run_id=run_id, sha256=sha, tokens=[word(tok, i) for i in ids], token_ids=ids, prompt_tokens=prompt_len,
                slices=slices, attention=attention, row_lengths=lengths, row_stats=stats,
                labels={"unembed.top": labels, "output.top": labels, "unembed.top.weight": labels},
                top_by_pass=passes, reply=tok.decode(ids[prompt_len:]))

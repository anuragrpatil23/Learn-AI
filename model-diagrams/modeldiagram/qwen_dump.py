"""modeldiagram trim-qwen-dump: values/ for Qwen3.5-9B from the dumps made for vlm-in-one-drawing.

Those dumps were made on Minerva on 2026-11-26 (activations: job 283648526, dump_activations.py; weights:
job 283648536, dump_weights.py; both scripts sit beside the dumps in the vault). This reads them and writes one
values file per picture and box: dots-hint, dots-empty, squares-hint, squares-empty. Nothing is rerun.

The dumps chose their rows (letter, object and text tokens), so every slice says which rows it holds in
`row_ids`. A weight slice is the first 32 rows of the weight file's tensor; where the spec splits one tensor
into parts (the query half of q_proj, the q rows of qkv), only the part the first 32 rows fall in is given.
"""
import hashlib
import json
import pathlib

import yaml

from .values import keep

ACT = {   # dump label -> spec tensor
    "v.patch": "vis_patchify.out", "v.b1.in": "vis_position.out", "v.b1.norm1": "vis_norm1.out", "v.b1.attn": "vis_attn.out",
    "v.b1.norm2": "vis_norm2.out", "v.b1.mlp.mid": "vis_mlp.act", "v.b1.mlp": "vis_mlp.out", "v.b1.out": "vis_add2.out",
    "v.last.out": "vis_after_blocks", "v.merge.norm": "vis_norm.out", "v.merge.in": "vis_join.out", "v.merge.mid": "vis_act.out",
    "v.merge.out": "vis_linear2.out",
    "l.b1.in": "join.out", "l.b1.norm1": "mem_norm1.out", "l.b1.mem": "mem_memory.out", "l.b1.norm2": "mem_norm2.out",
    "l.b1.mlp.mid": "mem_mlp.act", "l.b1.mlp": "mem_mlp.out", "l.b1.out": "mem_add2.out",
    "l.b4.in": "att_in", "l.b4.q": "att_attn.q", "l.b4.k": "att_attn.k", "l.b4.v": "att_attn.v", "l.b4.attn": "att_attn.out",
    "l.b4.out": "att_add2.out", "l.last.out": "after_blocks", "l.norm": "norm.out",
}
NOTES = {"att_attn.q": "the dump kept q_proj's output; its first 32 columns are head 1's first 32 query numbers, so they equal the query slice"}
P = "model.visual."
L = "model.language_model.layers."
WEIGHT = {
    P + "patch_embed.proj.weight": "vis_patchify.weight", P + "patch_embed.proj.bias": "vis_patchify.bias",
    P + "pos_embed.weight": "vis_position.weight",
    P + "blocks.0.norm1.weight": "vis_norm1.weight", P + "blocks.0.norm1.bias": "vis_norm1.bias",
    P + "blocks.0.norm2.weight": "vis_norm2.weight", P + "blocks.0.norm2.bias": "vis_norm2.bias",
    P + "blocks.0.attn.qkv.weight": "vis_attn.q.weight", P + "blocks.0.attn.qkv.bias": "vis_attn.q.bias",
    P + "blocks.0.attn.proj.weight": "vis_attn.out.weight", P + "blocks.0.attn.proj.bias": "vis_attn.out.bias",
    P + "blocks.0.mlp.linear_fc1.weight": "vis_mlp.up.weight", P + "blocks.0.mlp.linear_fc1.bias": "vis_mlp.up.bias",
    P + "blocks.0.mlp.linear_fc2.weight": "vis_mlp.down.weight", P + "blocks.0.mlp.linear_fc2.bias": "vis_mlp.down.bias",
    P + "merger.norm.weight": "vis_norm.weight", P + "merger.norm.bias": "vis_norm.bias",
    P + "merger.linear_fc1.weight": "vis_linear.weight", P + "merger.linear_fc1.bias": "vis_linear.bias",
    P + "merger.linear_fc2.weight": "vis_linear2.weight", P + "merger.linear_fc2.bias": "vis_linear2.bias",
    L + "0.input_layernorm.weight": "mem_norm1.weight", L + "0.post_attention_layernorm.weight": "mem_norm2.weight",
    L + "0.linear_attn.in_proj_qkv.weight": "mem_memory.q.weight", L + "0.linear_attn.conv1d.weight": "mem_memory.conv.weight",
    L + "0.linear_attn.in_proj_z.weight": "mem_memory.gate.weight", L + "0.linear_attn.in_proj_b.weight": "mem_memory.write.weight",
    L + "0.linear_attn.in_proj_a.weight": "mem_memory.fade.weight", L + "0.linear_attn.A_log": "mem_memory.fade_scale",
    L + "0.linear_attn.dt_bias": "mem_memory.fade_bias", L + "0.linear_attn.norm.weight": "mem_memory.norm.weight",
    L + "0.linear_attn.out_proj.weight": "mem_memory.out.weight",
    L + "0.mlp.gate_proj.weight": "mem_mlp.gate.weight", L + "0.mlp.up_proj.weight": "mem_mlp.up.weight",
    L + "0.mlp.down_proj.weight": "mem_mlp.down.weight",
    L + "3.input_layernorm.weight": "att_norm1.weight", L + "3.post_attention_layernorm.weight": "att_norm2.weight",
    L + "3.self_attn.q_proj.weight": "att_attn.q.weight", L + "3.self_attn.k_proj.weight": "att_attn.k.weight",
    L + "3.self_attn.v_proj.weight": "att_attn.v.weight", L + "3.self_attn.o_proj.weight": "att_attn.out.weight",
    L + "3.self_attn.q_norm.weight": "att_attn.q_norm.weight", L + "3.self_attn.k_norm.weight": "att_attn.k_norm.weight",
    L + "3.mlp.gate_proj.weight": "att_mlp.gate.weight", L + "3.mlp.up_proj.weight": "att_mlp.up.weight",
    L + "3.mlp.down_proj.weight": "att_mlp.down.weight",
    "model.language_model.norm.weight": "norm.weight", "lm_head.weight": "unembed.weight",
}
MAX_ROWS = 512


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def sig(x):
    """Every number at 3 significant digits, the precision the drawing prints."""
    if isinstance(x, list):
        return [sig(v) for v in x]
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(f"{x:.3g}")
    return x


def stride(lens):
    s = -(-len(lens) // MAX_ROWS)
    return lens[::s], s


def trim(a):
    from transformers import AutoTokenizer
    out = pathlib.Path(a.out)
    spec = yaml.safe_load(next(out.glob("*.model.yaml")).read_text())
    T = {t["id"]: t for t in spec["tensors"]}
    size = lambda ax: ax if isinstance(ax, int) else spec["axes"][ax]
    full = lambda tid, rows=None: [size(x) for x in T[tid]["axes"]] if rows is None else \
        [rows if x == "tokens" else 1 if x == "batch" else size(x) for x in T[tid]["axes"]]
    acts = json.loads(pathlib.Path(a.acts).read_text())
    wts = json.loads(pathlib.Path(a.weights).read_text())
    sha_a, sha_w = sha256(a.acts), sha256(a.weights)
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    weight_slices = {}
    for name, w in wts["tensors"].items():
        tid = WEIGHT.get(name)
        if tid is None:
            continue
        rows = w["slice"] if isinstance(w["slice"][0], list) else [w["slice"]]
        weight_slices[tid] = dict(shape=full(tid), rows=rows)
    missing = sorted(t for t in WEIGHT.values() if t not in weight_slices)
    if missing:
        print("weights not in the dump:", missing)
    (out / "values").mkdir(exist_ok=True)
    for pic, case in acts["cases"].items():
        ids = case["tokens"]
        n = len(ids)
        words = [tok.decode([i]).replace("\n", "\\n") for i in ids]
        for box in ("hint", "empty"):
            d = case[box]
            slices, lengths, every = dict(weight_slices), {}, {}
            for label, tid in ACT.items():
                if label not in d:
                    print(f"{pic}-{box}: {label} not in the dump")
                    continue
                x = d[label]
                shape = full(tid, n) if "tokens" in T[tid]["axes"] else full(tid)
                s = dict(shape=shape, rows=x["slice"], row_ids=x["rows"])
                if tid in NOTES:
                    s["note"] = NOTES[tid]
                slices[tid] = s
                lengths[tid], every[tid] = stride(x["rowlen"])
            vrows = case["vision_rows"]
            slices["pixels"] = dict(shape=full("pixels"), rows=case["pixel_slice"][box], row_ids=vrows)
            lengths["pixels"], every["pixels"] = stride(d["pixels_rowlen"])
            m = d["l.b1.memory"]
            slices["mem_memory.state"] = dict(shape=m["shape"], rows=m["head1_slice"])
            lg = d["logits"]
            slices["unembed.out"] = dict(shape=full("unembed.out", n), rows=lg["slice"], row_ids=lg["rows"])
            attention, last_row = {}, {}
            for op, key in (("vis_attn", "v.b1.att"), ("att_attn", "l.b4.att")):
                if key in d:
                    # head 1 only, on the chosen rows; the columns are the same chosen tokens
                    attention[op] = {"0": dict(rows=d[key]["head1_slice"], row_ids=d[key]["rows"])}
            if "l.b4.att" in d:
                lr = d["l.b4.att"]["last_row_all_heads"]
                last_row["att_attn"] = {str(h): stride(r)[0] for h, r in enumerate(lr)}
            v = dict(run_id="minerva-283648526", sha256=sha_a,
                     weights_run_id="minerva-283648536", weights_sha256=sha_w,
                     source="vlm-in-one-drawing's dumps (vlm-activations.json and weights.json), trimmed by modeldiagram trim-qwen-dump",
                     tokens=words, token_ids=ids, prompt_tokens=n,
                     picture=dict(name=f"{pic}-{box}", first_row=case["image_positions"][0], last_row=case["image_positions"][1],
                                  grid=case["grid"], letter_tokens=case["letter_tokens"], object_tokens=case["object_tokens"]),
                     slices=slices, attention=attention,
                     attention_last_row=last_row, attention_last_row_every=2,
                     row_lengths=lengths, row_lengths_every=every, reply=d["reply"])
            path = out / "values" / f"{pic}-{box}.json"
            for tid, sl in v["slices"].items():
                sl["rows"] = keep(sig(sl["rows"]), sl["shape"], T[tid]["role"])
            for heads in v["attention"].values():
                for m in heads.values():
                    m["rows"] = sig(m["rows"])
            path.write_text(json.dumps(v, separators=(",", ":"), ensure_ascii=False))
            print(f"wrote {path.name}: {path.stat().st_size / 1024:.0f} KB, reply {d['reply']!r}")
    print(f"activations sha256 {sha_a}\nweights sha256 {sha_w}")
    return 0

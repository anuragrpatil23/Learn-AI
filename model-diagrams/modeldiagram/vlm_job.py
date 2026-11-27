"""The labbook job that makes a vision-language model's values on Minerva, and its trim.

    labbook submit 01-vlm-values modeldiagram.vlm_job INPUTS=experiments/01-vlm-values/inputs/counting.json

(the full line, with KIND, CONDITION and LABBOOK_WHY, is in experiments/01-vlm-values/README.md).

The job (python -m modeldiagram.vlm_job, from the model-diagrams folder) reads the model from MODEL_PATH, the
shared copy labbook submit checked. For each input in the INPUTS manifest it:
  1. writes the reply (greedy, NEW_TOKENS tokens), with no hooks;
  2. runs one pass of the prompt and picture, traced by the same code as `modeldiagram spec`, so every tensor
     has the spec's name (trace.run_vlm keeps tensors only for the blocks the spec draws);
  3. keeps, in the run folder's dump.pt, more than the drawing needs: for each tensor its full shape, its
     chosen rows by the first 64 columns, and every row's length. For each attention op, every head on the
     chosen rows and columns, the scores before the softmax the same way, and every head's last row in full.
     For each weight, its first 64 rows and columns, the key and value parts included.
Then the trim (also `modeldiagram trim-vlm-dump <dump.pt> --out <model folder>`, which needs no GPU) writes
values/<input>.json and weights.json from dump.pt, cut to what the drawing shows.

Rows. ROWS=first takes the first 32 rows of each tensor. ROWS=hint-pairs (the default) takes the rows
dump_activations.py took for vlm-in-one-drawing, for manifests whose inputs come in pairs named <x>-hint and
<x>-empty. Letter tokens are image tokens whose pixels differ between the two, and object tokens are image
tokens below the box (box_bottom in the manifest, in pixels) whose pixels vary most. Vision: the 4 patches of each of 4 letter
and 4 object tokens. Language: the 8 tokens before the picture, 8 letter and 8 object tokens, and the last 8.
The manifest says the rest: {"box_bottom": 184, "inputs": [{"name", "image", "prompt"}]}, paths relative to it,
each prompt the exact text with one <|image_pad|> where the picture goes.
"""
import json
import os
import pathlib
import sys

import torch

KEEP = 64           # columns (and weight rows) kept in the raw dump
NR = 32             # chosen rows


def chosen_rows(proc, inputs_by_name, names, model_cfg, rule, box_bottom):
    """{name: {"vision": [...], "merged": [...], "language": [...]}} for every input."""
    out = {}
    if rule == "first":
        for n in names:
            out[n] = dict(vision=list(range(NR)), merged=list(range(NR // 4)), language=list(range(NR)))
        return out
    pairs = {}
    for n in names:
        base, _, box = n.rpartition("-")
        if box not in ("hint", "empty"):
            raise SystemExit(f"ROWS=hint-pairs needs inputs named <x>-hint and <x>-empty; {n} is neither")
        pairs.setdefault(base, {})[box] = n
    img_id = model_cfg.image_token_id
    for base, two in pairs.items():
        if set(two) != {"hint", "empty"}:
            raise SystemExit(f"{base} lacks its {'hint' if 'hint' not in two else 'empty'} input")
        h, e = inputs_by_name[two["hint"]], inputs_by_name[two["empty"]]
        pv_h, pv_e = h["pixel_values"].float().cpu(), e["pixel_values"].float().cpu()
        grid = h["image_grid_thw"][0].tolist()
        tr, tc = grid[1] // 2, grid[2] // 2
        ids = h["input_ids"][0].cpu()
        s0 = int((ids == img_id).nonzero()[0])
        pdiff = (pv_h - pv_e).abs().sum(-1).reshape(-1, 4).sum(-1)
        letters = (pdiff > 1e-3).nonzero().flatten().tolist()
        band = (box_bottom * grid[1] * 16 // h["height"]) // 32 + 1      # first token row below the box
        var = pv_e.std(-1).reshape(-1, 4).mean(-1)
        objects = sorted([k for k in range(tr * tc) if k // tc >= band], key=lambda k: -float(var[k]))[:40]
        pick = lambda xs, n: [xs[round(i * (len(xs) - 1) / (n - 1))] for i in range(n)] if len(xs) >= n else xs
        L8, O8 = pick(letters, 8), pick(sorted(objects), 8)
        tok = L8[:4] + O8[:4]
        rows = dict(vision=[4 * k + j for k in tok for j in range(4)], merged=tok,
                    language=list(range(s0 - 8, s0)) + [s0 + k for k in L8] + [s0 + k for k in O8] + list(range(len(ids) - 8, len(ids))),
                    letter_tokens=letters, object_tokens=sorted(objects))
        out[two["hint"]] = out[two["empty"]] = rows
    return out


def rows_of(axes, rows):
    """Which of the chosen rows a tensor's leading axis takes."""
    if "tokens" in axes:
        return rows["language"]
    if "patches" in axes:
        return rows["vision"]
    if "v_merged" in axes:
        return rows["merged"]
    return None


def dump_one(b, rows):
    """The raw dump of one pass: generous, so the trim can change without a rerun."""
    d = {}
    top = b.data["output.out"][0, -1].float().topk(8).indices            # the last token's best 8, for the score steps
    lang = rows["language"]
    for tid, t in b.data.items():
        meta = b.tensors[tid]
        axes, role = meta["axes"], meta["role"]
        if tid == "ids":
            continue                                                       # the values carry token_ids
        x = t.detach()
        e = dict(shape=list(x.shape), role=role)
        if tid in ("unembed.top", "output.top"):
            src = b.data["unembed.out" if tid == "unembed.top" else "output.out"][0].float()
            e.update(row_ids=lang, slice=src[lang][:, top].half().cpu(), top_ids=top.tolist())
        elif tid == "unembed.top.weight":
            e.update(slice=x.float()[top, :KEEP].half().cpu(), top_ids=top.tolist())
        elif role == "weight":
            x = x.float()
            x = x if x.dim() > 1 else x[None]
            e["slice"] = x.reshape(x.shape[0], -1)[:KEEP, :KEEP].half().cpu()
        elif tid.endswith((".scores", ".weights")):
            r = rows_of(axes[2:], rows) or list(range(min(NR, x.shape[-1])))
            x = x[0].float()
            e.update(row_ids=r, slice=x[:, r][:, :, r].half().cpu(), last_row=x[:, -1].half().cpu())
        else:
            x = x.float()
            lead = rows_of(axes, rows)
            if "batch" in axes and x.dim() >= 3:
                x = x[0]
            elif "batch" in axes and x.dim() == 2:
                x = x[:1]
            if x.dim() > 2:
                while x.dim() > 2:
                    x = x[0]
                lead = None
            flat = x.reshape(x.shape[0], -1)
            r = lead if lead is not None else list(range(min(NR, flat.shape[0])))
            e.update(row_ids=r, slice=flat[r, :KEEP].half().cpu(), rowlen=flat.norm(dim=-1).half().cpu())
        d[tid] = e
    return d


def main():
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from PIL import Image
    from labbook import Run
    from . import build, cli
    model_path = os.environ["MODEL_PATH"]
    manifest_path = pathlib.Path(os.environ.get("INPUTS", "experiments/01-vlm-values/inputs/counting.json"))
    new_tokens = int(os.environ.get("NEW_TOKENS", "12"))
    rule = os.environ.get("ROWS", "hint-pairs")
    manifest = json.loads(manifest_path.read_text())
    here = manifest_path.parent
    run = Run("experiments/01-vlm-values", settings=dict(INPUTS=str(manifest_path), NEW_TOKENS=new_tokens, ROWS=rule))
    for item in manifest["inputs"]:
        run.input(str(here / item["image"]))
        run.input(str(here / item["prompt"]))
    proc = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                                        attn_implementation="eager").to("cuda").eval()
    inputs, extra = {}, {}
    for item in manifest["inputs"]:
        img = Image.open(here / item["image"]).convert("RGB")
        text = (here / item["prompt"]).read_text()
        x = proc(text=[text], images=[img], return_tensors="pt").to("cuda")
        inputs[item["name"]] = x
        extra[item["name"]] = dict(height=img.height)
    names = [i["name"] for i in manifest["inputs"]]
    rows = chosen_rows(proc, {n: dict(inputs[n], **extra[n]) for n in names}, names, model.config, rule,
                       manifest.get("box_bottom", 184))
    dump = dict(model=model_path, revision=os.environ.get("MODEL_REVISION"), rows_rule=rule, inputs={})
    for item in manifest["inputs"]:
        n, x = item["name"], inputs[item["name"]]
        ids = x["input_ids"]
        with torch.no_grad():
            g = model.generate(**x, max_new_tokens=new_tokens, do_sample=False)
        reply = proc.batch_decode(g[:, ids.shape[1]:], skip_special_tokens=True)[0]
        s0 = int((ids[0] == model.config.image_token_id).nonzero()[0])
        roots = __import__("modeldiagram.trace", fromlist=["run_vlm"]).run_vlm(model, dict(x))
        b = build.build_vlm(roots, model_path, ids, dict(image_at=s0))
        tensors = dump_one(b, rows[n])
        dump["inputs"][n] = dict(tokens=ids[0].tolist(), words=[proc.tokenizer.decode([i]).replace("\n", "\\n") for i in ids[0].tolist()],
                                 vocab_words={str(i): proc.tokenizer.decode([i]).strip() or repr(proc.tokenizer.decode([i]))
                                              for i in b.data["output.out"][0, -1].float().topk(8).indices.tolist()},
                                 reply=reply, image_at=s0, grid=x["image_grid_thw"][0].tolist(), rows=rows[n], tensors=tensors)
        run.example(n, condition={"model": "untrained"}, prompt=(here / item["prompt"]).read_text(),
                    image=str(here / item["image"]), output=reply)
        print(f"{n}: {ids.shape[1]} tokens, picture from {s0}, reply {reply!r}, {len(tensors)} tensors", flush=True)
        del roots, b
        torch.cuda.empty_cache()
    path = pathlib.Path(run.dir) / "dump.pt"
    torch.save(dump, path)
    out = pathlib.Path(run.dir) / "model-folder"
    out.mkdir(exist_ok=True)
    trim(path, out, run_id=f"minerva-{os.environ.get('LSB_JOBID', 'local')}")
    run.finish({"inputs": len(names), "dump_sha256": cli_sha(path)})


def cli_sha(path):
    from .values import sha256
    return sha256(path)


def trim(dump_path, folder, run_id=None):
    """values/<input>.json and weights.json from a dump.pt this job wrote."""
    from .values import keep, r4, write_weights, sha256, MAX_ROWS
    dump_path, folder = pathlib.Path(dump_path), pathlib.Path(folder)
    dump = torch.load(dump_path, weights_only=False)
    sha = sha256(dump_path)
    run_id = run_id or dump_path.parent.name
    sig = lambda m: [[r4(v) for v in row] for row in m.float().tolist()]
    weights = None
    (folder / "values").mkdir(parents=True, exist_ok=True)
    for n, rec in dump["inputs"].items():
        slices, attention, lengths, every, last = {}, {}, {}, {}, {}
        labels = []
        dec = lambda i: rec["vocab_words"].get(str(i), str(i))
        for tid, e in rec["tensors"].items():
            if tid in ("unembed.top", "output.top", "unembed.top.weight"):
                rows_ = sig(e["slice"]) if tid != "unembed.top.weight" else keep(sig(e["slice"]), e["shape"], "activation")
                slices[tid] = dict(shape=e["shape"], rows=rows_, **({"row_ids": e["row_ids"]} if "row_ids" in e else {}))
                labels = [w for w in (dec(i) for i in e["top_ids"])]
                continue
            if e["role"] == "weight":
                if weights is None or tid not in weights:
                    weights = weights or {}
                    weights[tid] = dict(shape=e["shape"], rows=keep(sig(e["slice"]), e["shape"], "weight"))
                continue
            if tid.endswith((".scores", ".weights")):
                op = tid.rsplit(".", 1)[0]
                if tid.endswith(".weights"):
                    attention[op] = {str(h): dict(rows=sig(e["slice"][h]), row_ids=e["row_ids"]) for h in range(e["slice"].shape[0])}
                    s = max(1, -(-e["last_row"].shape[-1] // MAX_ROWS))
                    last[op] = {str(h): sig(e["last_row"][h:h + 1, ::s])[0] for h in range(e["last_row"].shape[0])}
                slices[tid] = dict(shape=e["shape"], rows=sig(e["slice"][0]), row_ids=e["row_ids"], col_ids=e["row_ids"])
                continue
            slices[tid] = dict(shape=e["shape"], rows=keep(sig(e["slice"]), e["shape"], "activation"), row_ids=e["row_ids"])
            rl = e["rowlen"].float()
            s = max(1, -(-rl.shape[0] // MAX_ROWS))
            lengths[tid], every[tid] = [float(f"{v:.3g}") for v in rl[::s].tolist()], s
        r = rec["rows"]
        v = dict(run_id=run_id, sha256=sha, source=f"modeldiagram.vlm_job, rows {dump.get('rows_rule')}",
                 tokens=rec["words"], token_ids=rec["tokens"], prompt_tokens=len(rec["tokens"]),
                 picture=dict(name=n, first_row=rec["image_at"], grid=rec["grid"],
                              letter_tokens=r.get("letter_tokens"), object_tokens=r.get("object_tokens")),
                 slices=slices, attention=attention, attention_last_row=last,
                 labels={"unembed.top": labels, "output.top": labels, "unembed.top.weight": labels},
                 row_lengths=lengths, row_lengths_every=every, reply=rec["reply"])
        p = folder / "values" / f"{n}.json"
        p.write_text(json.dumps(v, separators=(",", ":"), ensure_ascii=False))
        print(f"wrote {p.name}: {p.stat().st_size / 1024:.0f} KB, reply {rec['reply']!r}")
    wp = write_weights(folder, dict(run_id=run_id, sha256=sha, slices=weights or {}))
    print(f"wrote {wp.name}: {wp.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())

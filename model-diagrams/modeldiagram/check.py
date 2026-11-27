"""modeldiagram check: does a model folder connect?

The same rules as the engine's checker (Thinking Space, modelDiagramFormatBlock.ts), plus the ones only the
generator can promise: every op reads something made before it, and every values file fits its spec.
Each problem names where it is, so it can be fixed by hand.
"""
import json
import pathlib

import yaml

from .kinds import OP_KINDS

ELEMENTWISE = {"norm", "act", "add", "position"}
LEADING_KEPT = {"linear", "mlp", "attention", "unembed"}
VARYING = {"batch", "tokens", "patches"}
MAX_VALUES = 2 * 1024 * 1024


def check_spec(spec):
    p = []
    axes = spec.get("axes") or {}
    for a, s in axes.items():
        if not (isinstance(s, int) and s > 0):
            p.append(f"axes.{a}: size {s!r} is not a positive whole number")
    if not (spec.get("source") or {}).get("model"):
        p.append("source: no model id")
    T = {}
    for t in spec.get("tensors", []):
        if t["id"] in T:
            p.append(f"tensors {t['id']}: id used twice")
        T[t["id"]] = t
        if t.get("role") not in ("activation", "weight"):
            p.append(f"tensors {t['id']}: role {t.get('role')!r}")
        for a in t.get("axes", []):
            if isinstance(a, str) and a not in axes:
                p.append(f"tensors {t['id']}: axis {a} is not one of the model's axes")
    size = lambda a: a if isinstance(a, int) else axes.get(a)
    shape = lambda i: [size(a) for a in T[i]["axes"]]
    blocks = {b["id"]: b for b in spec.get("blocks", [])}
    made = set()
    ids = set(blocks)
    stream = spec.get("stream", [])
    all_outs = {t for o in spec.get("ops", []) for t in o.get("out", [])}
    after_block = False
    for o in spec.get("ops", []):
        at = f"ops {o['id']}"
        if o["id"] in ids:
            p.append(f"{at}: id used twice")
        ids.add(o["id"])
        if o.get("kind") not in OP_KINDS:
            p.append(f"{at}: kind {o.get('kind')!r} is not one of the known kinds")
        for f in ("in", "out", "weights"):
            for r in o.get(f, []):
                if r not in T:
                    p.append(f"{at}: {f} names {r}, which is not in tensors")
                elif f == "weights" and T[r]["role"] != "weight":
                    p.append(f"{at}: weights names {r}, an activation")
                elif f == "in" and T[r]["role"] == "activation" and r not in made and r not in stream \
                        and not (after_block and r not in all_outs):
                    # a tensor no op makes is fine after a block: the repeated blocks made it
                    p.append(f"{at}: reads {r} before any op makes it")
        made.update(o.get("out", []))
        after_block = after_block or bool(o.get("block"))
        if not o.get("out") and o.get("kind") != "output":
            p.append(f"{at}: no output")
        if o.get("block") and o["block"] not in blocks:
            p.append(f"{at}: block {o['block']} is not in blocks")
        if o.get("kind") == "attention":
            if o.get("mask") not in ("causal", "none"):
                p.append(f"{at}: mask {o.get('mask')!r}")
            h, kv = o.get("heads"), o.get("kv_heads", o.get("heads"))
            if not (isinstance(h, int) and h > 0 and isinstance(kv, int) and h % kv == 0):
                p.append(f"{at}: heads {h} over kv_heads {kv}")
        if o.get("kind") == "memory" and not o.get("state"):
            p.append(f"{at}: a memory op needs its state shape")
        acts = [i for i in o.get("in", []) if i in T and T[i]["role"] == "activation"]
        if acts and all(i in T for i in o.get("out", [])):
            first = shape(acts[0])
            for i in (acts[1:] + o["out"]) if o["kind"] in ELEMENTWISE else (o["out"] if o["kind"] in LEADING_KEPT else []):
                a, b = (first, shape(i)) if o["kind"] in ELEMENTWISE else (first[:-1], shape(i)[:-1])
                if a != b:
                    p.append(f"{at}: a {o['kind']} keeps {'its shape' if o['kind'] in ELEMENTWISE else 'every axis but the last'}, "
                             f"but {acts[0]} {T[acts[0]]['axes']} and {i} {T[i]['axes']} disagree")
    for b in blocks.values():
        for oid in b.get("ops", []):
            op = next((o for o in spec["ops"] if o["id"] == oid), None)
            if op is None:
                p.append(f"blocks {b['id']}: names op {oid}, not in ops")
            elif op.get("block") != b["id"]:
                p.append(f"blocks {b['id']}: op {oid} says it is in {op.get('block')}")
    for tower, pts in (spec.get("streams") or {}).items():
        for s in pts:
            if s not in T or T[s]["role"] != "activation":
                p.append(f"streams.{tower}: {s} is not an activation in tensors")
    for i, s in enumerate(stream):
        if s not in T or T[s]["role"] != "activation":
            p.append(f"stream: {s} is not an activation in tensors")
        elif i and shape(stream[i - 1])[-1] != shape(s)[-1]:
            p.append(f"stream: the width changes between {stream[i - 1]} and {s}")
    if not spec.get("steps"):
        p.append("steps: none")
    for i, s in enumerate(spec.get("steps", [])):
        if s.get("id") not in ids:
            p.append(f"steps[{i}]: {s.get('id')} is neither an op nor a block")
        if not s.get("text"):
            p.append(f"steps[{i}]: no sentence")
    return p


def check_values(spec, text, name, weights_file=False):
    """A values file holds activations (and the per-input unembed.top.weight); weights.json holds the weights."""
    p = []
    if len(text.encode()) > MAX_VALUES:
        p.append(f"{name}: {len(text.encode()) / 1048576:.1f} MB, over the 2 MB cap")
    v = json.loads(text)
    for f in ("run_id", "sha256"):
        if not v.get(f):
            p.append(f"{name}: no {f}")
    T = {t["id"]: t for t in spec["tensors"]}
    axes = spec["axes"]
    for tid, s in (v.get("slices") or {}).items():
        if tid not in T:
            p.append(f"{name} slices.{tid}: not a tensor of the spec")
            continue
        if weights_file != (T[tid]["role"] == "weight") and tid != "unembed.top.weight":
            p.append(f"{name} slices.{tid}: a {T[tid]['role']}, which belongs in " + ("values/" if weights_file else "weights.json"))
        ax = T[tid]["axes"]
        if len(s["shape"]) != len(ax):
            p.append(f"{name} slices.{tid}: {len(s['shape'])} axes, the tensor has {len(ax)}")
            continue
        for a, got in zip(ax, s["shape"]):
            want = a if isinstance(a, int) else axes[a]
            if a not in VARYING and got != want:
                p.append(f"{name} slices.{tid}: axis {a} is {got} here and {want} in the spec")
        rows = s.get("rows") or []
        if len(rows) > 32 or any(len(r) > 32 for r in rows):
            p.append(f"{name} slices.{tid}: more than 32 rows or columns")
        if "row_ids" in s and len(s["row_ids"]) != len(rows):
            p.append(f"{name} slices.{tid}: {len(s['row_ids'])} row ids for {len(rows)} rows")
    att = {o["id"] for o in spec["ops"] if o["kind"] == "attention"}
    for key in ("attention", "attention_scores"):          # weights and scores per head, laid out alike
        for oid, heads in (v.get(key) or {}).items():
            if oid not in att:
                p.append(f"{name} {key}.{oid}: not an attention op")
            for h, m in heads.items():
                if isinstance(m, dict) and len(m.get("row_ids", [])) != len(m.get("rows", [])):
                    p.append(f"{name} {key}.{oid}.{h}: {len(m.get('row_ids', []))} row ids for {len(m.get('rows', []))} rows")
                rows = m.get("rows", []) if isinstance(m, dict) else m
                if len(rows) > 32 or any(len(r) > 32 for r in rows):
                    p.append(f"{name} {key}.{oid}.{h}: more than 32 rows or columns")
    for tid, r in (v.get("row_lengths") or {}).items():
        if tid not in T:
            p.append(f"{name} row_lengths.{tid}: not a tensor of the spec")
        elif len(r) > 512 and "row_lengths_sampled" not in v:
            p.append(f"{name} row_lengths.{tid}: {len(r)} rows, over 512 and not sampled")
    return p


def check_folder(folder):
    folder = pathlib.Path(folder)
    specs = list(folder.glob("*.model.yaml"))
    if len(specs) != 1:
        return [f"{folder}: {len(specs)} files named *.model.yaml, expected one"], {}
    spec = yaml.safe_load(specs[0].read_text())
    problems = check_spec(spec)
    name = specs[0].name[: -len(".model.yaml")]
    if not (folder / f"{name}.md").exists():
        problems.append(f"{folder}: no {name}.md beside the spec")
    stats = dict(spec=specs[0].name, tensors=len(spec["tensors"]), ops=len(spec["ops"]), blocks=len(spec["blocks"]),
                 steps=len(spec["steps"]), values={})
    if not check_spec(spec):
        wf = folder / "weights.json"
        if wf.exists():
            text = wf.read_text()
            problems += check_values(spec, text, "weights.json", weights_file=True)
            stats["values"]["../weights.json"] = f"{len(text.encode()) / 1024:.0f} KB"
        for vf in sorted((folder / "values").glob("*.json")):
            text = vf.read_text()
            problems += check_values(spec, text, f"values/{vf.name}")
            stats["values"][vf.name] = f"{len(text.encode()) / 1024:.0f} KB"
    return problems, stats

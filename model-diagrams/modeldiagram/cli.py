"""modeldiagram: write a model's structure and real values in the vault's model-diagram format.

  modeldiagram spec   <hf id> --out <model folder> [--batch B --tokens T] [--grid 1,60,48 --image-at 61]
  modeldiagram values <hf id> --input <text file> --out <model folder>       (language models, run on the CPU)
  modeldiagram trim-qwen-dump --acts <vlm-activations.json> --weights <weights.json> --out <model folder> ...
  modeldiagram trim-vlm-dump  <run folder>/dump.pt --out <model folder>     (after the Minerva job, see vlm_job.py)
  modeldiagram check  <model folder> [<model folder> ...]

The format is described in the vault at kai-workspace/universal/model-diagrams.md.
"""
import argparse
import datetime
import json
import pathlib
import subprocess
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parent


def generator_commit():
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=HERE.parent, capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def revision_of(hf_id):
    try:
        from huggingface_hub import try_to_load_from_cache
        p = try_to_load_from_cache(hf_id, "config.json")
        return pathlib.Path(p).parent.name if isinstance(p, str) else None
    except Exception:
        return None


def flow(x):
    return yaml.safe_dump(x, default_flow_style=True, width=10 ** 9, sort_keys=False, allow_unicode=True).strip()


def write_yaml(spec, path, header):
    lines = [f"# {h}" for h in header]
    for key in ("source", "axes"):
        lines.append(f"{key}: {flow(spec[key])}")
    for key in ("tensors", "ops", "blocks"):
        lines.append(f"{key}:")
        lines += [f"  - {flow(x)}" for x in spec[key]]
    lines.append(f"stream: {flow(spec['stream'])}")
    if spec.get("streams"):
        lines.append(f"streams: {flow(spec['streams'])}")
    lines.append("steps:")
    lines += [f"  - {flow(x)}" for x in spec["steps"]]
    pathlib.Path(path).write_text("\n".join(lines) + "\n")


def cmd_spec(a):
    import torch
    from . import trace, build
    cfg, kind, model = trace.load(a.hf_id, meta=True)
    if kind == "llm":
        ids = torch.zeros(a.batch, a.tokens, dtype=torch.long, device="meta")
        b = build.build_llm(trace.run_llm(model, ids), a.hf_id, ids)
    else:
        grid = [int(x) for x in a.grid.split(",")]
        roots, at = trace.run_vlm_meta(model, cfg, a.tokens, grid, a.image_at)
        b = build.build_vlm(roots, a.hf_id, torch.zeros(1, a.tokens, dtype=torch.long, device="meta"), at)
    source = dict(model=a.hf_id, revision=revision_of(a.hf_id), generator_commit=generator_commit(),
                  date=datetime.date.today().isoformat())
    spec = build.Builder.spec(b, source)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = a.name or out.name
    path = out / f"{name}.model.yaml"
    pass_desc = (f"batch {a.batch}, {a.tokens} tokens" if kind == "llm"
                 else f"a picture of {grid[1]} x {grid[2]} patches, {a.tokens} tokens with the picture's rows from token {a.image_at}")
    write_yaml(spec, path, [f"{a.hf_id}, a {kind}, written by modeldiagram spec (Learn-AI/model-diagrams) on {source['date']}.",
                            f"Every shape is from one pass on the meta device ({pass_desc}); batch, tokens and patches vary with the input.",
                            "Do not edit by hand: change the generator and run it again."])
    from .check import check_spec
    problems = check_spec(yaml.safe_load(path.read_text()))
    print(f"wrote {path}: {len(spec['tensors'])} tensors, {len(spec['ops'])} ops, {len(spec['blocks'])} blocks, {len(spec['steps'])} steps")
    for p in problems:
        print("  problem:", p)
    return 1 if problems else 0


def cmd_values(a):
    from .values import run_llm_values
    return run_llm_values(a.hf_id, pathlib.Path(a.input), pathlib.Path(a.out), a.new_tokens, pathlib.Path(a.runs))


def cmd_trim_qwen(a):
    from .qwen_dump import trim
    return trim(a)


def cmd_trim_vlm(a):
    from .vlm_job import trim
    run_id = a.run_id
    rj = pathlib.Path(a.dump).parent / "run.json"
    if run_id is None and rj.exists():
        job = json.loads(rj.read_text()).get("job")
        run_id = f"minerva-{job}" if job else None
    return trim(a.dump, a.out, run_id=run_id)


def cmd_check(a):
    from .check import check_folder
    bad = 0
    for f in a.folders:
        problems, stats = check_folder(f)
        print(f"{f}: " + ("connects" if not problems else f"{len(problems)} problems"))
        if stats:
            print("  " + ", ".join(f"{k} {v}" for k, v in stats.items() if k != "values"))
            for vf, size in stats["values"].items():
                print(f"  {'weights.json' if vf.startswith('..') else 'values/' + vf}: {size}, fits the spec")
        for p in problems:
            print("  problem:", p)
        bad += bool(problems)
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="modeldiagram", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("spec", help="write <name>.model.yaml from one pass on the meta device")
    s.add_argument("hf_id")
    s.add_argument("--out", required=True)
    s.add_argument("--name", help="the model's name in the folder (default: the folder's name)")
    s.add_argument("--batch", type=int, default=1)
    s.add_argument("--tokens", type=int, default=16)
    s.add_argument("--grid", default="1,32,32", help="vision models: the picture's patch grid, t,h,w")
    s.add_argument("--image-at", type=int, default=1, help="vision models: the token where the picture's rows start")
    s.set_defaults(fn=cmd_spec)
    v = sub.add_parser("values", help="run a language model on one input and write values/<input>.json")
    v.add_argument("hf_id")
    v.add_argument("--input", required=True, help="a text file; its name without .txt is the input's name")
    v.add_argument("--out", required=True)
    v.add_argument("--new-tokens", type=int, default=8, help="how many tokens the model writes after the input")
    v.add_argument("--runs", default=str(HERE.parent / "runs"), help="where the raw dump is kept")
    v.set_defaults(fn=cmd_values)
    q = sub.add_parser("trim-qwen-dump", help="write values/ from the Qwen3.5 dumps of vlm-in-one-drawing")
    q.add_argument("--acts", required=True)
    q.add_argument("--weights", required=True)
    q.add_argument("--acts-job", required=True)
    q.add_argument("--weights-job", required=True)
    q.add_argument("--tokenizer", default="Qwen/Qwen3.5-9B")
    q.add_argument("--out", required=True)
    q.set_defaults(fn=cmd_trim_qwen)
    t = sub.add_parser("trim-vlm-dump", help="write values/ and weights.json from a dump.pt of modeldiagram.vlm_job (no GPU)")
    t.add_argument("dump")
    t.add_argument("--out", required=True)
    t.add_argument("--run-id", help="default: minerva-<job id> from the run folder's run.json, else the folder's name")
    t.set_defaults(fn=cmd_trim_vlm)
    c = sub.add_parser("check", help="check that a model folder connects")
    c.add_argument("folders", nargs="+")
    c.set_defaults(fn=cmd_check)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())

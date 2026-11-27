"""Check GPT-2's values against the numbers of the hand-built page gpt-with-sparse-network.html.

That page's data (gpt2-sae-window.json) was exported by Learn-AI/gpt2-window-export/export.py, with a forward
pass written out by hand. Each of its grids is matched with the tensor the generator named for the same
step, over the page's window (every row, the first 8 columns), and the largest difference is printed: plain
below 1, relative above, since the values keep 3 significant digits.

Usage: python tests/compare_reference.py <gpt2-sae-window.json> <model folder>
"""
import json
import pathlib
import sys

REF_TO_OURS = {   # reference path -> our tensor id
    "embed.wte": "embed.out", "embed.out": "position.out",
    "b0.ln1.out": "norm1.out", "b0.q": "attn.q", "b0.k": "attn.k", "b0.v": "attn.v",
    "b0.scores": "attn.scores", "b0.att": "attn.weights", "b0.headsOut": "attn.mixed", "b0.attnOut": "attn.out",
    "b0.residMid": "add1.out", "b0.ln2.out": "norm2.out", "b0.fc": "mlp.pre", "b0.act": "mlp.act",
    "b0.mlpOut": "mlp.out", "b0.residOut": "add2.out", "fin.residIn": "after_blocks", "fin.lnF.out": "norm.out",
}
W_TO_OURS = {"q.W": "attn.q.weight", "k.W": "attn.k.weight", "v.W": "attn.v.weight", "attnOut.W": "attn.out.weight",
             "fc.W": "mlp.up.weight", "mlpOut.W": "mlp.down.weight", "ln1.g": "norm1.weight", "ln2.g": "norm2.weight",
             "lnF.g": "norm.weight", "q.b": "attn.q.bias", "fc.b": "mlp.up.bias"}


def get(d, path):
    for k in path.split("."):
        d = d[k]
    return d


def diff(a, b):
    a = a if isinstance(a[0], list) else [a]
    b = b if isinstance(b[0], list) else [b]
    m = 0.0
    for ra, rb in zip(a, b):
        for x, y in zip(ra, rb):
            if x is None or y is None or x < -1e8 or y < -1e8:
                continue
            m = max(m, abs(x - y) / max(1.0, abs(x)))   # relative above 1: the values keep 3 significant digits
    return m, min(len(a), len(b)), min(len(a[0]), len(b[0]))


def main(ref_path, folder):
    ref = json.loads(pathlib.Path(ref_path).read_text())
    worst = 0.0
    for run in ref["runs"]:
        name = run["prompt"].split("the ")[1].split()[0]          # sky, sea, grass
        ours = json.loads((pathlib.Path(folder) / "values" / f"{name}.json").read_text())
        assert ours["tokens"] == run["strs"], (name, ours["tokens"], run["strs"])
        for rp, tid in REF_TO_OURS.items():
            m, r, c = diff(get(run, rp), ours["slices"][tid]["rows"])
            worst = max(worst, m)
            print(f"{name:6s} {rp:14s} vs {tid:14s} {r:2d} x {c:2d}  max diff {m:.1e}")
        probs = ours["slices"]["output.top"]["rows"]
        refp = run["fin"]["vocab"][ours["prompt_tokens"] - 1]
        top_ok = refp["strs"] == ours["labels"]["output.top"]
        m, _, _ = diff(refp["probs"], probs)
        print(f"{name:6s} top 8 words at the question's last token agree: {top_ok}; chances max diff {m:.1e}")
        worst = max(worst, m)
        assert top_ok
    ours = json.loads((pathlib.Path(folder) / "weights.json").read_text())
    for rp, tid in W_TO_OURS.items():
        m, r, c = diff(get(ref["weights"], rp), ours["slices"][tid]["rows"])
        worst = max(worst, m)
        print(f"weight {rp:10s} vs {tid:16s} {r:2d} x {c:2d}  max diff {m:.1e}")
    print(f"largest difference anywhere: {worst:.1e}, relative for numbers above 1 (the reference rounds to 4 decimals, the values keep 3 significant digits)")
    return 0 if worst < 6e-3 else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))

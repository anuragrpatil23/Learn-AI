"""Add a real sparse autoencoder to the GPT-2 window used by the interactive drawing.

Takes the JSON written by gpt2-window-export/export.py and, for each of its prompts, runs GPT-2 small,
reads the 3,072 MLP neurons of the first block (after GELU) at every position, and passes each row
through the sparse autoencoder OpenAI published for exactly those neurons (32,768 features, trained
with a reconstruction error plus an L1 penalty). It saves a window of the numbers at every step:
the same first 8 neurons the drawing already shows, and 32 feature columns. The first 8 are chosen
as the strongest feature of 8 different words of the prompt. The other 24 are spread evenly through
the numbering, kept to show what an ordinary feature looks like: nearly always off. (The lowest
numbers are not used for this: features 0 to 23 turned out to be on far more often than average.)
The MLP neurons are saved 16 columns wide, so the drawing can show them wider than the stream.

It also runs one small test to say what each chosen feature responds to: every token GPT-2 knows is
placed after "Q: Why is the", and the tokens that fire the feature hardest are kept.

Usage: python export.py <gpt2-window.json> <autoencoder .pt> <output.json>
"""
import sys, json, torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

WIN = 8
EXTRA = 24
NEUR = 16      # MLP neuron columns saved
src, sae_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
data = json.load(open(src))
tok = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
sd = torch.load(sae_path, map_location="cpu")
W_enc, W_dec, b_pre, b_enc = sd["encoder.weight"], sd["decoder.weight"], sd["pre_bias"], sd["latent_bias"]
N_FEAT, N_IN = W_enc.shape

grab = {}
model.transformer.h[0].mlp.c_fc.register_forward_hook(lambda m, i, o: grab.__setitem__("fc", o.detach()))
model.transformer.h[0].mlp.act.register_forward_hook(lambda m, i, o: grab.__setitem__("act", o.detach()))

def mlp_neurons(ids):
    """The first block's MLP neurons after GELU, one row per position: (T, 3072)."""
    with torch.no_grad():
        model(torch.tensor([ids]))
    return grab["act"][0], grab["fc"][0]

def sae(x):
    xin = x - b_pre
    pre = xin @ W_enc.T + b_enc
    f = torch.relu(pre)
    xhat = f @ W_dec.T + b_pre
    return xin, pre, f, xhat

r4 = lambda t: [[round(float(v), 4) for v in row] for row in t]
passes = []
for run in data["runs"]:
    ids = run["tokens"]
    x, fc = mlp_neurons(ids)
    have = torch.tensor(run["b0"]["act"])
    n = have.shape[0]
    assert (x[:n, :WIN] - have).abs().max() < 2e-3, "the MLP neurons do not match the window already saved"
    xin, pre, f, xhat = sae(x[:n])
    run['_fc'] = fc[:n]
    passes.append((run, x[:n], xin, pre, f, xhat))

# the 8 feature columns to draw for a prompt: the strongest feature of each word, question words first
def pick(run, f):
    order = list(range(2, run["qlen"])) + [0, 1] + list(range(run["qlen"], f.shape[0]))
    feats = []
    for t in order:
        j = int(f[t].argmax())
        if j not in feats:
            feats.append(j)
        if len(feats) == WIN:
            break
    return feats

chosen = [pick(run, f) for run, x, xin, pre, f, xhat in passes]
union = sorted({j for c in chosen for j in c})

# what each chosen feature responds to: every token, placed after "Q: Why is the"
frame = tok("Q: Why is the").input_ids
best = torch.zeros(len(union), 50257)
with torch.no_grad():
    for start in range(0, 50257, 1024):
        v = torch.arange(start, min(start + 1024, 50257))
        ids = torch.cat([torch.tensor(frame).repeat(len(v), 1), v[:, None]], dim=1)
        model(ids)
        last = grab["act"][:, -1, :]
        best[:, start:start + len(v)] = torch.relu((last - b_pre) @ W_enc[union].T + b_enc[union]).T
vocab_top = {}
for n, j in enumerate(union):
    top = best[n].topk(8)
    vocab_top[j] = [[tok.decode([int(i)]), round(float(v), 3)] for v, i in zip(top.values, top.indices)]

for (run, x, xin, pre, f, xhat), feats in zip(passes, chosen):
    spread = [int((k + 0.5) * N_FEAT / EXTRA) for k in range(EXTRA)]
    allf = feats + [j if j not in feats else j + 1 for j in spread]
    cols = torch.tensor(allf)
    rows = []
    for t in range(x.shape[0]):
        top = f[t].topk(5)
        rows.append({
            "on": int((f[t] > 0).sum()),
            "lost": round(float(((x[t] - xhat[t]) ** 2).sum() / (x[t] ** 2).sum()), 4),
            "errsq": round(float(((x[t] - xhat[t]) ** 2).sum()), 4),
            "l1": round(float(f[t].sum()), 4),
            "top": [[int(i), round(float(v), 3)] for v, i in zip(top.values, top.indices)],
        })
        # the two halves of the loss if only this word's k strongest features are kept, for k = 0, 1, 2, ...
        order = f[t].argsort(descending=True)[: rows[-1]["on"]]
        part = b_pre.clone(); errs = [float(((x[t] - part) ** 2).sum())]; fire = [0.0]
        for j in order:
            part = part + f[t, j] * W_dec[:, j]
            errs.append(float(((x[t] - part) ** 2).sum())); fire.append(fire[-1] + float(f[t, j]))
        rows[-1]["keepErr"] = [round(v, 3) for v in errs]; rows[-1]["keepFire"] = [round(v, 3) for v in fire]
    info = []
    for j in feats:
        fires = []
        for (run2, x2, xin2, pre2, f2, xhat2) in passes:
            for t in range(f2.shape[0]):
                if float(f2[t, j]) > 0.05:
                    fires.append([run2["strs"][t], round(float(f2[t, j]), 2)])
        seen, uniq = set(), []
        for w, v in sorted(fires, key=lambda p: -p[1]):
            if w not in seen:
                seen.add(w); uniq.append([w, v])
        info.append({"id": j, "fires": uniq[:6], "vocab": vocab_top[j]})
    run["sae"] = {
        "feats": allf, "nPick": len(feats), "info": info, "rows": rows,
        "act16": r4(x[:, :NEUR]), "fc16": r4(run.pop("_fc")[:, :NEUR]),
        "xin": r4(xin[:, :NEUR]), "pre": r4(pre[:, cols]), "f": r4(f[:, cols]),
        "xhat": r4(xhat[:, :NEUR]), "err": r4((x - xhat)[:, :NEUR]),
        "enc": {"W": r4(W_enc[cols][:, :WIN]), "b": [round(float(v), 4) for v in b_enc[cols]]},
        "dec": {"W": r4(W_dec[:NEUR][:, cols]), "b": [round(float(v), 4) for v in b_pre[:NEUR]]},
        "bpre": [round(float(v), 4) for v in b_pre[:NEUR]],
    }

lens = W_dec.norm(dim=0)
fcl = model.transformer.h[0].mlp.c_fc
data["meta"]["sae"] = {
    "fc": {"W": r4(fcl.weight[:WIN, :NEUR].T), "b": [round(float(v), 4) for v in fcl.bias[:NEUR]]},
    "nFeat": N_FEAT, "nIn": N_IN, "decLenMin": round(float(lens.min()), 2), "decLenMax": round(float(lens.max()), 2),
    "source": "OpenAI sparse_autoencoder, gpt2-small, mlp_post_act, layer index 0",
    "frame": "Q: Why is the",
}
json.dump(data, open(out_path, "w"), separators=(",", ":"))

for (run, x, xin, pre, f, xhat), feats in zip(passes, chosen):
    q = run["qlen"]
    print(repr(run["prompt"]), "features drawn:", feats)
    print("  on per word: %d to %d of %d;  not rebuilt: %.1f%% to %.1f%%" % (
        min(r["on"] for r in run["sae"]["rows"]), max(r["on"] for r in run["sae"]["rows"]), N_FEAT,
        100 * min(r["lost"] for r in run["sae"]["rows"]), 100 * max(r["lost"] for r in run["sae"]["rows"])))
    for i in run["sae"]["info"]:
        print("  #%d fires on %s | tokens that fire it most: %s" % (i["id"], [w for w, v in i["fires"]], [w for w, v in i["vocab"][:6]]))

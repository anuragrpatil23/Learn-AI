"""A scanner for the Train Run Tracker's Scan view: GPT-2 small, with the sparse autoencoders trained here.

It serves the embedding and first block of GPT-2 small, every step of it, with a sparse autoencoder
attached to the first block's MLP neurons. Which sparse autoencoder is the "snapshot": any
step_*.pt saved by train.py found under --runs, or the one OpenAI published for the same neurons.

Everything a scanner does that is the same for any network (the server, the snapshots, the grids,
the contrast, taking a feature apart) comes from scankit, a copy of which sits beside this file
(github.com/anuragrpatil23/run-tracker, scankit/). What is written here is GPT-2's own.

Usage: python scan_server.py --runs ~/run-tracker-data/runs --port 8790 [--openai path/to/0.pt]
       python scan_server.py --runs ... --write-fixtures scan_fixtures     (recorded replies, then exit)
"""
import argparse, json, math, os
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import scankit

p = argparse.ArgumentParser()
p.add_argument("--runs", required=True, help="folder of run folders; snapshots are step_*.pt at any depth under it")
p.add_argument("--port", type=int, default=8790)
p.add_argument("--openai", default=None, help="the sparse autoencoder OpenAI published for these neurons, as one more snapshot")
p.add_argument("--write-fixtures", default=None)
args = p.parse_args()

C, H, HS, M = 768, 12, 64, 3072
FRAME = "Q: Why is the"                      # a unit's words: every token placed after this, strongest first
torch.set_grad_enabled(False)
tok = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
tr, b0 = model.transformer, model.transformer.h[0]

def gelu(x): return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))
def norm(x, m):
    mean = x.mean(-1, keepdim=True); spread = ((x - mean) ** 2).mean(-1, keepdim=True).add(m.eps).sqrt()
    return (x - mean) / spread * m.weight + m.bias

def first_block(ids):
    """ids: list of token ids. Returns every step's numbers for the first block, keyed by node id."""
    T = len(ids); x = tr.wte.weight[ids] + tr.wpe.weight[:T]
    l1 = norm(x, b0.ln_1)
    q, k, v = (l1 @ b0.attn.c_attn.weight + b0.attn.c_attn.bias).split(C, dim=-1)
    heads = lambda z: z.view(T, H, HS).transpose(0, 1)
    scores = heads(q) @ heads(k).transpose(-2, -1) / math.sqrt(HS)
    masked = torch.tril(torch.ones(T, T)) == 0
    att = scores.masked_fill(masked, float("-inf")).softmax(-1)
    mix = (att @ heads(v)).transpose(0, 1).reshape(T, C)
    attn_out = mix @ b0.attn.c_proj.weight + b0.attn.c_proj.bias
    add1 = x + attn_out
    l2 = norm(add1, b0.ln_2)
    act = gelu(l2 @ b0.mlp.c_fc.weight + b0.mlp.c_fc.bias)
    mlp_out = act @ b0.mlp.c_proj.weight + b0.mlp.c_proj.bias
    return {"embed": x, "b0.ln1": l1, "b0.q": q, "b0.k": k, "b0.v": v,
            "b0.scores": scores[0].masked_fill(masked, float("nan")), "b0.att": att[0],     # head 1 of 12
            "b0.mix": mix, "b0.attn_out": attn_out, "b0.add1": add1, "b0.ln2": l2,
            "b0.mlp.act": act, "b0.mlp.out": mlp_out, "b0.add2": add1 + mlp_out}

def vocab_rows():
    """The MLP neurons for every token GPT-2 knows, each placed after FRAME: (50257, 3072). Made once, when words are first wanted."""
    if not hasattr(vocab_rows, "kept"):
        frame, parts = tok(FRAME).input_ids, []
        for v in torch.arange(50257).split(2048):
            ids = torch.cat([torch.tensor(frame).repeat(len(v), 1), v[:, None]], dim=1)
            parts.append(scankit.hooked({"act": b0.mlp.act}, lambda: b0(tr.wte(ids) + tr.wpe(torch.arange(ids.shape[1]))))["act"][:, -1])
        vocab_rows.kept = torch.cat(parts)
    return vocab_rows.kept

VOCAB = [tok.decode([i]) for i in range(50257)]

def node(id, label, kind, width, lane, order, about, group=None, per="token", **more):
    return dict({"id": id, "label": label, "group": group, "kind": kind, "width": width, "per": per, "lane": lane, "order": order, "about": about}, **more)

class GPT2Scanner(scankit.Scanner):
    name = "GPT-2 small with sparse autoencoders"

    def load(self, path):
        sd = torch.load(path, map_location="cpu")
        if "encoder.weight" in sd:            # the file OpenAI published
            W_enc, b_enc, W_dec, b_dec = sd["encoder.weight"], sd["latent_bias"], sd["decoder.weight"], sd["pre_bias"]
        else:                                 # a file train.py saved
            W_enc, b_enc, W_dec, b_dec = sd["W_enc"].float(), sd["b_enc"].float(), sd["W_dec"].float(), sd["b_dec"].float()
        vocab_rows()                          # made here, not at the first run, so the wait shows as "loading"
        return scankit.SparseAutoencoder(detector=W_enc, detector_bias=b_enc, pattern=W_dec, usual=b_dec, reads="b0.mlp.act",
                                         probe=vocab_rows, probe_words=VOCAB, cache=path + ".words.json",
                                         test="every token placed after \"%s\", strongest first" % FRAME)

    def others(self):
        return [{"id": "openai-mlp0", "path": os.path.expanduser(args.openai), "label": "OpenAI, 32,768 features"}] if args.openai else []

    def tokens(self, text):
        return [(t, tok.decode([t])) for t in tok(text).input_ids]

    def run(self, ids, snapshot):
        vals = first_block(ids)
        vals.update(snapshot.data.run(vals["b0.mlp.act"]))
        return vals

    def next(self, ids, n=5):
        probs = model(torch.tensor([ids])).logits[0, -1].softmax(-1).topk(n)
        return [{"text": tok.decode([int(i)]), "chance": round(float(v), 4)} for v, i in zip(probs.values, probs.indices)]

    def words(self, snapshot, node, units): return snapshot.data.words(units)
    def usual(self, snapshot, node): return snapshot.data.usual(node)
    def unit(self, snapshot, node, unit):
        out = snapshot.data.unit(node, unit)
        if node == "sae.features":            # the whole pattern and detector, one number per MLP neuron, for drawing on the sheet
            sae = snapshot.data
            out["made_of"]["all"] = [scankit.r4(v) for v in sae.pattern[:, unit] / sae.lengths[unit]]
            out["listens_to"]["all"] = [scankit.r4(v) for v in sae.detector[unit] / (sae.detector[unit] ** 2).sum() ** 0.5]
        return out

    def graph(self, snapshot):
        n = snapshot.data.n
        nodes = [
            node("embed", "Embedding", "lookup", C, 0, 0, "Each token picks a row from the word table and a row from the position table, and the two are added. This is the stream.", unit="channel", weights=50257 * C + 1024 * C),
            node("b0.ln1", "Layer norm", "norm", C, 2, 1, "Each token's row is shifted and rescaled to mean 0 and spread 1, then given a learnt scale and shift.", "attention", unit="channel", weights=2 * C),
            node("b0.q", "Query", "linear", C, 3, 2, "What each token is looking for. 768 neurons, each with 768 slopes and an intercept.", "attention", unit="channel", weights=C * C + C, each=C),
            node("b0.k", "Key", "linear", C, 2, 2, "What each token offers to be found by.", "attention", unit="channel", weights=C * C + C, each=C),
            node("b0.v", "Value", "linear", C, 1, 2, "What each token hands over if another token looks at it.", "attention", unit="channel", weights=C * C + C, each=C),
            node("b0.scores", "Similarity", "scores", None, 3, 3, "A query times a key, for every pair of tokens. A token cannot look at later ones. Head 1 of 12 is shown.", "attention", per="token-pair"),
            node("b0.att", "Attention weights", "weights-over-positions", None, 3, 4, "Each row of scores turned into shares that add to 1: how much each token takes from each earlier one. Head 1 of 12.", "attention", per="token-pair"),
            node("b0.mix", "Mix the values", "mix", C, 2, 5, "Each earlier token's value times the attention given to it, added up. The one step where tokens exchange information.", "attention", unit="channel"),
            node("b0.attn_out", "Output layer", "linear", C, 2, 6, "A linear layer turning the mixed values into what attention adds to the stream.", "attention", unit="channel", weights=C * C + C, each=C),
            node("b0.add1", "Add to the stream", "add", C, 0, 10, "Attention's result is added to the stream; nothing is overwritten.", unit="channel"),
            node("b0.ln2", "Layer norm", "norm", C, -1, 11, "The same rescaling, with its own scale and shift.", "mlp", unit="channel", weights=2 * C),
            node("b0.mlp.act", "MLP neurons", "linear+bend", M, -1, 12, "3,072 neurons, each a weighted sum of the 768 channels passed through GELU. These are what the sparse autoencoder reads.", "mlp", bend="GELU", unit="neuron", weights=M * C + M, each=C),
            node("b0.mlp.out", "MLP layer 2", "linear", C, -1, 13, "A linear layer going back down to 768.", "mlp", unit="channel", weights=C * M + C, each=M),
            node("b0.add2", "Add to the stream", "add", C, 0, 14, "The MLP's result is added to the stream, which goes on to the next block.", unit="channel"),
            {"id": "rest", "label": "11 more blocks, then a score for every token", "group": None, "kind": "collapsed", "lane": 0, "order": 20},
            node("sae.in", "Minus the usual level", "difference", M, -2, 13, "One learnt number per neuron, close to its average, is taken off. What is left is how this token differs from the usual.", "sae", unit="neuron", weights=M),
            node("sae.features", "Features", "linear+bend", n, -2, 14, "%s neurons, each a weighted sum of the 3,072 MLP neurons passed through ReLU. For any one token almost all are zero." % format(n, ","), "sae", bend="ReLU", unit="feature", weights=n * M + n, sparse=True, described=True, each=M),
            node("sae.rebuild", "The rebuild", "linear", M, -2, 15, "Each feature's pattern times how hard it fired, added up, plus the usual level. The sparse autoencoder's attempt at the MLP neurons.", "sae", unit="neuron", weights=M * n, each=n),
        ]
        flow = [("embed", "b0.ln1"), ("b0.ln1", "b0.q"), ("b0.ln1", "b0.k"), ("b0.ln1", "b0.v"), ("b0.q", "b0.scores"), ("b0.k", "b0.scores"),
                ("b0.scores", "b0.att"), ("b0.att", "b0.mix"), ("b0.v", "b0.mix"), ("b0.mix", "b0.attn_out"), ("b0.attn_out", "b0.add1"),
                ("b0.add1", "b0.ln2"), ("b0.ln2", "b0.mlp.act"), ("b0.mlp.act", "b0.mlp.out"), ("b0.mlp.out", "b0.add2"), ("b0.add2", "rest"),
                ("sae.in", "sae.features"), ("sae.features", "sae.rebuild")]
        edges = [{"from": a, "to": b} for a, b in flow] + [{"from": "embed", "to": "b0.add1", "kind": "carried"},
                {"from": "b0.add1", "to": "b0.add2", "kind": "carried"}, {"from": "b0.mlp.act", "to": "sae.in", "kind": "reads"}]
        return {"title": "GPT-2 small, first block, with a sparse autoencoder on its MLP neurons",
                "groups": [{"id": "attention", "label": "Attention"}, {"id": "mlp", "label": "MLP"}, {"id": "sae", "label": "Sparse autoencoder", "attached": True}],
                "nodes": nodes, "edges": edges}

if args.write_fixtures:
    kit = scankit.Kit(GPT2Scanner(), args.runs); os.makedirs(args.write_fixtures, exist_ok=True)
    def keep(name, value): json.dump(value, open(os.path.join(args.write_fixtures, name + ".json"), "w"), indent=1)
    keep("snapshots_before_load", kit.list())
    s = kit.snapshot(kit.default(), ready=False); kit.load_now(s)
    r = kit.run("Q: Why is the sky blue?\nA:", s)
    sky = r["nodes"]["sae.features"]["top"][5][0][0]; neuron = kit.unit(s, "sae.features", sky)["made_of"]["top"][0][0]
    keep("health", kit.health()); keep("snapshots", kit.list()); keep("load", {"snapshot": s.id, "state": "ready"}); keep("graph", kit.graph(s)); keep("run", r)
    keep("values", kit.values(r["run"], "b0.mlp.act", 5, 0, 4096)); keep("values_token_pair", kit.values(r["run"], "b0.att", 5, 0, 0))
    keep("unit_feature", kit.unit(s, "sae.features", sky)); keep("unit_neuron", kit.unit(s, "b0.mlp.act", neuron))
    keep("contrast", kit.contrast("Q: Why is the sky blue?\nA:", "Q: Why is the sea blue?\nA:", s, "sae.features"))
    keep("error_not_loaded", {"error": {"code": "not_loaded", "message": "Snapshot 'x' is not loaded yet. Ask for it with load, then try again."}})
    print("fixtures written for", s.id, "- sky feature", sky, "- neuron", neuron)
else:
    scankit.serve(GPT2Scanner(), args.runs, args.port)

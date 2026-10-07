"""A scanner for the Train Run Tracker's Scan view: GPT-2 small, with the sparse autoencoders trained here.

It answers the questions in the Scan contract (run-tracker-scan-spec.md): what the network looks
like, what each step produces for a piece of text, what one unit is, and how two texts differ.
The tracker never loads a model itself; it asks this program, over 127.0.0.1 only.

The network served is the embedding and first block of GPT-2 small, every step of it, with a sparse
autoencoder attached to the first block's MLP neurons. Which sparse autoencoder is the "snapshot":
any step_*.pt saved by train.py found under --runs, or the one OpenAI published for the same neurons.

Usage: python scan_server.py --runs ~/run-tracker-data/runs --port 8790 [--openai path/to/0.pt]
       python scan_server.py --runs ... --write-fixtures scan_fixtures     (recorded replies, then exit)
"""
import argparse, collections, glob, json, math, os, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

p = argparse.ArgumentParser()
p.add_argument("--runs", required=True, help="folder of run folders; snapshots are step_*.pt at any depth under it")
p.add_argument("--port", type=int, default=8790)
p.add_argument("--openai", default=None, help="the sparse autoencoder OpenAI published for these neurons, as one more snapshot")
p.add_argument("--write-fixtures", default=None)
args = p.parse_args()

RUNS = os.path.abspath(os.path.expanduser(args.runs))
MAX_TOKENS, C, H, HS, M = 64, 768, 12, 64, 3072
FRAME = "Q: Why is the"                      # a unit's words: every token placed after this, strongest first
torch.set_grad_enabled(False)
tok = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
tr, b0 = model.transformer, model.transformer.h[0]
lock = threading.Lock()

class Problem(Exception):
    def __init__(self, code, message, status=400):
        self.code, self.message, self.status = code, message, status

# ---------- the model: the embedding and every step of the first block ----------
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

def next_tokens(ids, n=5):
    probs = model(torch.tensor([ids])).logits[0, -1].softmax(-1).topk(n)
    return [{"text": tok.decode([int(i)]), "chance": round(float(v), 4)} for v, i in zip(probs.values, probs.indices)]

vocab_rows = None                             # the MLP neurons for every token placed after FRAME: (50257, 3072)
def make_vocab_rows():
    global vocab_rows
    grab = {}; hook = b0.mlp.act.register_forward_hook(lambda m, i, o: grab.__setitem__("a", o))
    frame, parts = tok(FRAME).input_ids, []
    for v in torch.arange(50257).split(2048):
        ids = torch.cat([torch.tensor(frame).repeat(len(v), 1), v[:, None]], dim=1)
        b0(tr.wte(ids) + tr.wpe(torch.arange(ids.shape[1]))); parts.append(grab["a"][:, -1].clone())
    hook.remove(); vocab_rows = torch.cat(parts)

# ---------- snapshots: sparse autoencoders that can be attached ----------
class Snapshot:
    def __init__(self, id, path, run, step, label):
        self.id, self.path, self.run, self.step, self.label = id, path, run, step, label
        self.state, self.words_of = "cold", {}
    def describe(self):
        return {"id": self.id, "label": self.label, "run": self.run, "step": self.step,
                "bytes": os.path.getsize(self.path), "state": self.state}
    def load(self):
        sd = torch.load(self.path, map_location="cpu")
        if "encoder.weight" in sd:            # the file OpenAI published
            self.W_enc, self.b_enc, self.W_dec, self.b_dec = sd["encoder.weight"], sd["latent_bias"], sd["decoder.weight"], sd["pre_bias"]
        else:                                 # a file train.py saved
            self.W_enc, self.b_enc, self.W_dec, self.b_dec = sd["W_enc"].float(), sd["b_enc"].float(), sd["W_dec"].float(), sd["b_dec"].float()
        self.n = self.W_enc.shape[0]
        self.lengths = self.W_dec.norm(dim=0)                       # each feature's pattern, to compare patterns on one scale
        if vocab_rows is None:
            make_vocab_rows()
        self.state = "ready"
    def nodes(self, act):
        x_in = act - self.b_dec
        f = torch.relu(x_in @ self.W_enc.T + self.b_enc)
        return {"sae.in": x_in, "sae.features": f, "sae.rebuild": f @ self.W_dec.T + self.b_dec}
    def words(self, units, n=4):
        """For each feature, the tokens that fire it hardest when placed after FRAME."""
        todo = [u for u in dict.fromkeys(units) if u not in self.words_of]
        for i in range(0, len(todo), 64):
            part = todo[i:i + 64]
            fired = torch.relu((vocab_rows - self.b_dec) @ self.W_enc[part].T + self.b_enc[part]).T.topk(8, dim=1)
            for u, vals, idx in zip(part, fired.values, fired.indices):
                self.words_of[u] = [[tok.decode([int(t)]), round(float(v), 3)] for v, t in zip(vals, idx) if v > 0]
        return [[w for w, v in self.words_of[u][:n]] or None for u in units]

snapshots = {}
def find_snapshots():
    for path in sorted(glob.glob(os.path.join(RUNS, "**", "step_*.pt"), recursive=True)):
        run = os.path.relpath(os.path.dirname(path), RUNS)
        step = int(os.path.basename(path)[5:-3]); id = "%s@%d" % (run, step)
        if id not in snapshots:
            snapshots[id] = Snapshot(id, path, run, step, "%s, step %s" % (os.path.basename(run), format(step, ",")))
    if args.openai and "openai-mlp0" not in snapshots:
        snapshots["openai-mlp0"] = Snapshot("openai-mlp0", os.path.expanduser(args.openai), None, None, "OpenAI, 32,768 features")
    for id in [i for i, s in snapshots.items() if not os.path.exists(s.path)]:
        del snapshots[id]

def snapshot(id, ready=True):
    find_snapshots()
    if id not in snapshots:
        raise Problem("unknown_snapshot", "No snapshot called %r is on this machine." % id, 404)
    if ready and snapshots[id].state != "ready":
        raise Problem("not_loaded", "Snapshot %r is not loaded yet. Ask for it with load, then try again." % id, 409)
    return snapshots[id]

def default_snapshot():
    find_snapshots()
    ours = [s for s in snapshots.values() if s.run]
    return max(ours, key=lambda s: (os.path.getsize(s.path), s.step)).id if ours else next(iter(snapshots), None)

# ---------- the drawing ----------
def node(id, label, kind, width, lane, order, about, group=None, per="token", **more):
    return dict({"id": id, "label": label, "group": group, "kind": kind, "width": width, "per": per, "lane": lane, "order": order, "about": about}, **more)

def graph(snap):
    n = snap.n
    nodes = [
        node("embed", "Embedding", "lookup", C, 0, 0, "Each token picks a row from the word table and a row from the position table, and the two are added. This is the stream.", unit="channel", weights=50257 * C + 1024 * C),
        node("b0.ln1", "Layer norm", "norm", C, 2, 1, "Each token's row is shifted and rescaled to mean 0 and spread 1, then given a learnt scale and shift.", "attention", unit="channel", weights=2 * C),
        node("b0.q", "Query", "linear", C, 3, 2, "What each token is looking for. 768 neurons, each with 768 slopes and an intercept.", "attention", unit="channel", weights=C * C + C),
        node("b0.k", "Key", "linear", C, 2, 2, "What each token offers to be found by.", "attention", unit="channel", weights=C * C + C),
        node("b0.v", "Value", "linear", C, 1, 2, "What each token hands over if another token looks at it.", "attention", unit="channel", weights=C * C + C),
        node("b0.scores", "Similarity", "scores", None, 3, 3, "A query times a key, for every pair of tokens. A token cannot look at later ones. Head 1 of 12 is shown.", "attention", per="token-pair"),
        node("b0.att", "Attention weights", "weights-over-positions", None, 3, 4, "Each row of scores turned into shares that add to 1: how much each token takes from each earlier one. Head 1 of 12.", "attention", per="token-pair"),
        node("b0.mix", "Mix the values", "mix", C, 2, 5, "Each earlier token's value times the attention given to it, added up. The one step where tokens exchange information.", "attention", unit="channel"),
        node("b0.attn_out", "Output layer", "linear", C, 2, 6, "A linear layer turning the mixed values into what attention adds to the stream.", "attention", unit="channel", weights=C * C + C),
        node("b0.add1", "Add to the stream", "add", C, 0, 10, "Attention's result is added to the stream; nothing is overwritten.", unit="channel"),
        node("b0.ln2", "Layer norm", "norm", C, -1, 11, "The same rescaling, with its own scale and shift.", "mlp", unit="channel", weights=2 * C),
        node("b0.mlp.act", "MLP neurons", "linear+bend", M, -1, 12, "3,072 neurons, each a weighted sum of the 768 channels passed through GELU. These are what the sparse autoencoder reads.", "mlp", bend="GELU", unit="neuron", weights=M * C + M),
        node("b0.mlp.out", "MLP layer 2", "linear", C, -1, 13, "A linear layer going back down to 768.", "mlp", unit="channel", weights=C * M + C),
        node("b0.add2", "Add to the stream", "add", C, 0, 14, "The MLP's result is added to the stream, which goes on to the next block.", unit="channel"),
        {"id": "rest", "label": "11 more blocks, then a score for every token", "group": None, "kind": "collapsed", "lane": 0, "order": 20},
        node("sae.in", "Minus the usual level", "difference", M, -2, 13, "One learnt number per neuron, close to its average, is taken off. What is left is how this token differs from the usual.", "sae", unit="neuron", weights=M),
        node("sae.features", "Features", "linear+bend", n, -2, 14, "%s neurons, each a weighted sum of the 3,072 MLP neurons passed through ReLU. For any one token almost all are zero." % format(n, ","), "sae", bend="ReLU", unit="feature", weights=n * M + n, sparse=True, described=True),
        node("sae.rebuild", "The rebuild", "linear", M, -2, 15, "Each feature's pattern times how hard it fired, added up, plus the usual level. The sparse autoencoder's attempt at the MLP neurons.", "sae", unit="neuron", weights=M * n),
    ]
    flow = [("embed", "b0.ln1"), ("b0.ln1", "b0.q"), ("b0.ln1", "b0.k"), ("b0.ln1", "b0.v"), ("b0.q", "b0.scores"), ("b0.k", "b0.scores"),
            ("b0.scores", "b0.att"), ("b0.att", "b0.mix"), ("b0.v", "b0.mix"), ("b0.mix", "b0.attn_out"), ("b0.attn_out", "b0.add1"),
            ("b0.add1", "b0.ln2"), ("b0.ln2", "b0.mlp.act"), ("b0.mlp.act", "b0.mlp.out"), ("b0.mlp.out", "b0.add2"), ("b0.add2", "rest"),
            ("sae.in", "sae.features"), ("sae.features", "sae.rebuild")]
    edges = [{"from": a, "to": b} for a, b in flow] + [{"from": "embed", "to": "b0.add1", "kind": "carried"},
            {"from": "b0.add1", "to": "b0.add2", "kind": "carried"}, {"from": "b0.mlp.act", "to": "sae.in", "kind": "reads"}]
    return {"title": "GPT-2 small, first block, with a sparse autoencoder on its MLP neurons",
            "groups": [{"id": "attention", "label": "Attention"}, {"id": "mlp", "label": "MLP"}, {"id": "sae", "label": "Sparse autoencoder", "attached": true_}],
            "nodes": nodes, "edges": edges}
true_ = True

# ---------- running text ----------
results = collections.OrderedDict()           # the last few runs, for follow-up questions
r4 = lambda v: None if v != v else round(float(v), 4)

def compute(text, snap):
    ids = tok(text).input_ids
    if not ids:
        raise Problem("bad_request", "There is no text to run.")
    if len(ids) > MAX_TOKENS:
        raise Problem("too_long", "That is %d tokens; this scanner takes at most %d." % (len(ids), MAX_TOKENS), 413)
    vals = first_block(ids); vals.update(snap.nodes(vals["b0.mlp.act"]))
    name = "r%05x" % (abs(hash((text, snap.id, len(results), os.urandom(4)))) % 0xfffff)
    results[name] = {"snapshot": snap, "ids": ids, "vals": vals}
    while len(results) > 16:
        results.popitem(last=False)
    return name, ids, vals

def tokens_of(ids): return [{"i": i, "text": tok.decode([t]), "id": t} for i, t in enumerate(ids)]

def run(text, snap, top=12):
    name, ids, vals = compute(text, snap)
    spec = {n["id"]: n for n in graph(snap)["nodes"]}; out = {}
    for id, x in vals.items():
        if spec[id]["per"] == "token-pair":
            out[id] = {"grid": [[r4(v) for v in row] for row in x]}; continue
        best = x.abs().topk(min(top, x.shape[1]), dim=1)
        described = spec[id].get("described")
        strength = torch.zeros(x.shape[1]); strength[best.indices.flatten().unique()] = 1
        units = (x.abs().max(0).values * strength).topk(min(48, int(strength.sum()))).indices.tolist()
        entry = {"size": [r4(v) for v in x.norm(dim=1)],
                 "top": [[[int(u), r4(x[t, u])] for u in best.indices[t] if x[t, u] != 0] for t in range(len(ids))],
                 "grid": {"units": units, "values": [[r4(v) for v in x[t, units]] for t in range(len(ids))], "words": snap.words(units) if described else None}}
        if described:
            for t, row in enumerate(entry["top"]):
                for item, w in zip(row, snap.words([u for u, v in row])):
                    item.append(w)
        if spec[id].get("sparse"):
            entry["on"] = [int(v) for v in (x != 0).sum(1)]
        out[id] = entry
    return {"run": name, "snapshot": snap.id, "tokens": tokens_of(ids), "nodes": out, "next": next_tokens(ids)}

def result(name):
    if name not in results:
        raise Problem("unknown_run", "Run %r is no longer kept. Run the text again." % name, 404)
    return results[name]

def values(name, node_id, token, start, count):
    r = result(name)
    if node_id not in r["vals"]:
        raise Problem("unknown_node", "There is no node called %r." % node_id, 404)
    x = r["vals"][node_id]
    if not 0 <= token < x.shape[0]:
        raise Problem("bad_request", "Token %d is outside the text." % token)
    if node_id in ("b0.scores", "b0.att"):
        return {"values": [r4(v) for v in x[token]]}
    count = max(1, min(count, 4096)); out = {"from": start, "count": len(x[token, start:start + count]), "values": [r4(v) for v in x[token, start:start + count]]}
    if node_id == "b0.mlp.act":
        out["usual"] = [r4(v) for v in r["snapshot"].b_dec[start:start + count]]
    return out

def spread(v):
    """How many entries carry half, and ninety percent, of a list's squared size."""
    c = (v ** 2).sort(descending=True).values.cumsum(0) / (v ** 2).sum()
    return int((c < 0.5).sum()) + 1, int((c < 0.9).sum()) + 1

def unit(snap, node_id, u):
    widths = {n["id"]: n.get("width") for n in graph(snap)["nodes"]}
    if node_id not in widths or widths[node_id] is None:
        raise Problem("unknown_node", "There is no node called %r with units." % node_id, 404)
    if not 0 <= u < widths[node_id]:
        raise Problem("unknown_unit", "%s has units 0 to %d." % (node_id, widths[node_id] - 1), 404)
    out = {"node": node_id, "unit": u}
    if node_id == "sae.features":
        snap.words([u]); pattern = snap.W_dec[:, u] / snap.lengths[u]; detector = snap.W_enc[u] / snap.W_enc[u].norm()
        for key, v in (("made_of", pattern), ("listens_to", detector)):
            half, ninety = spread(v); best = v.abs().topk(12).indices
            out[key] = {"node": "b0.mlp.act", "top": [[int(i), r4(v[i])] for i in best], "half": half, "ninety": ninety}
        out["words"] = snap.words_of[u]; out["words_test"] = "every token placed after \"%s\", strongest first" % FRAME
        out["agreement"] = r4(torch.dot(pattern, detector))
    elif node_id in ("b0.mlp.act", "sae.in", "sae.rebuild"):
        lean = snap.W_dec[u] / snap.lengths; best = lean.abs().topk(8).indices.tolist()
        out["used_by"] = {"node": "sae.features", "count": int((lean.abs() > 0.1).sum()), "threshold": 0.1,
                          "top": [[j, r4(lean[j]), w] for j, w in zip(best, snap.words(best))]}
    return out

def differ(a, b, snap, described, top):
    """Units on for one row and off or much weaker for the other, and units shared, as lists for the reply."""
    gap = a - b
    only_a = [int(u) for u in gap.topk(top).indices if gap[u] > 0 and abs(b[u]) < 0.25 * abs(a[u])]
    only_b = [int(u) for u in (-gap).topk(top).indices if gap[u] < 0 and abs(a[u]) < 0.25 * abs(b[u])]
    both = [int(u) for u in torch.minimum(a.abs(), b.abs()).topk(top).indices if a[u] != 0 and b[u] != 0]
    w = dict(zip(only_a + only_b + both, snap.words(only_a + only_b + both))) if described else {}
    return {"only_a": [[u, r4(a[u]), w.get(u)] for u in only_a], "only_b": [[u, r4(b[u]), w.get(u)] for u in only_b],
            "both": [[u, r4(a[u]), r4(b[u]), w.get(u)] for u in both]}

def contrast(text_a, text_b, snap, node_id, top=12):
    spec = {n["id"]: n for n in graph(snap)["nodes"]}
    if node_id not in spec or spec[node_id].get("per") != "token":
        raise Problem("unknown_node", "Contrast needs a node with one row per token; %r is not one." % node_id, 404)
    na, ia, va = compute(text_a, snap); nb, ib, vb = compute(text_b, snap)
    if len(ia) != len(ib):
        raise Problem("bad_request", "The two texts must split into the same number of tokens: the first has %d and the second %d." % (len(ia), len(ib)))
    xa, xb, described = va[node_id], vb[node_id], spec[node_id].get("described")
    per_pair = [dict({"same_text": ia[t] == ib[t], "distance": r4((xa[t] - xb[t]).norm())}, **differ(xa[t], xb[t], snap, described, top)) for t in range(len(ia))]
    whole = differ(xa.sum(0), xb.sum(0), snap, described, top); whole.pop("both")
    return {"run_a": na, "run_b": nb, "node": node_id, "tokens_a": tokens_of(ia), "tokens_b": tokens_of(ib),
            "pairs": [[t, t] for t in range(len(ia))], "per_pair": per_pair, "whole": whole,
            "first_difference": next((t for t in range(len(ia)) if ia[t] != ib[t]), None)}

# ---------- the routes ----------
def health():
    return {"ok": True, "protocol": 1, "name": "GPT-2 small with sparse autoencoders", "device": "cpu",
            "max_tokens": MAX_TOKENS, "supports": ["run", "values", "unit", "contrast", "load"]}

def list_snapshots():
    find_snapshots()
    return {"snapshots": [s.describe() for s in snapshots.values()], "default": default_snapshot(), "files": "step_*.pt"}

def load(id):
    s = snapshot(id, ready=False)
    if s.state == "cold":
        s.state = "loading"
        def work():
            with lock:
                try: s.load()
                except Exception: s.state = "cold"; raise
        threading.Thread(target=work, daemon=True).start()
    return {"snapshot": s.id, "state": s.state}

def answer(method, path, q, body):
    one = lambda k, d=None: (q.get(k) or [d])[0]
    if path == "/scan/v1/health": return health()
    if path == "/scan/v1/snapshots": return list_snapshots()
    if path == "/scan/v1/load" and method == "POST": return load(body.get("snapshot"))
    if path == "/scan/v1/steer": raise Problem("not_loaded", "This scanner cannot yet set a unit by hand.", 501)
    with lock:
        if path == "/scan/v1/graph": return graph(snapshot(one("snapshot") or default_snapshot()))
        if path == "/scan/v1/run" and method == "POST":
            return run(body.get("text", ""), snapshot(body.get("snapshot") or default_snapshot()), int(body.get("top", 12)))
        if path == "/scan/v1/values":
            return values(one("run"), one("node"), int(one("token", 0)), int(one("from", 0)), int(one("count", 4096)))
        if path == "/scan/v1/unit":
            return unit(snapshot(one("snapshot") or default_snapshot()), one("node"), int(one("unit", -1)))
        if path == "/scan/v1/contrast" and method == "POST":
            return contrast(body.get("a", ""), body.get("b", ""), snapshot(body.get("snapshot") or default_snapshot()),
                            body.get("node") or "sae.features", int(body.get("top", 12)))
    raise Problem("bad_request", "There is no route %s %s." % (method, path), 404)

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def reply(self, method):
        url = urlparse(self.path); body = {}
        try:
            if method == "POST":
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                try: body = json.loads(raw or b"{}")
                except ValueError: raise Problem("bad_request", "The request body is not JSON.")
            out, status = answer(method, url.path, parse_qs(url.query), body), 200
        except Problem as e:
            out, status = {"error": {"code": e.code, "message": e.message}}, e.status
        except (ValueError, TypeError) as e:
            out, status = {"error": {"code": "bad_request", "message": str(e)}}, 400
        except Exception as e:
            out, status = {"error": {"code": "internal", "message": "%s: %s" % (type(e).__name__, e)}}, 500
        data = json.dumps(out).encode()
        self.send_response(status); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data)))
        self.end_headers(); self.wfile.write(data)
    def do_GET(self): self.reply("GET")
    def do_POST(self): self.reply("POST")

if args.write_fixtures:
    os.makedirs(args.write_fixtures, exist_ok=True)
    def keep(name, value): json.dump(value, open(os.path.join(args.write_fixtures, name + ".json"), "w"), indent=1)
    keep("snapshots_before_load", list_snapshots())
    s = snapshot(default_snapshot(), ready=False); s.load()
    r = run("Q: Why is the sky blue?\nA:", s)
    sky = r["nodes"]["sae.features"]["top"][5][0][0]; neuron = unit(s, "sae.features", sky)["made_of"]["top"][0][0]
    keep("health", health()); keep("snapshots", list_snapshots()); keep("load", {"snapshot": s.id, "state": "ready"}); keep("graph", graph(s)); keep("run", r)
    keep("values", values(r["run"], "b0.mlp.act", 5, 0, 4096)); keep("values_token_pair", values(r["run"], "b0.att", 5, 0, 0))
    keep("unit_feature", unit(s, "sae.features", sky)); keep("unit_neuron", unit(s, "b0.mlp.act", neuron))
    keep("contrast", contrast("Q: Why is the sky blue?\nA:", "Q: Why is the sea blue?\nA:", s, "sae.features"))
    keep("error_not_loaded", {"error": {"code": "not_loaded", "message": "Snapshot 'x' is not loaded yet. Ask for it with load, then try again."}})
    print("fixtures written for", s.id, "- sky feature", sky, "- neuron", neuron)
else:
    find_snapshots()
    print("scanner on http://127.0.0.1:%d with %d snapshots under %s" % (args.port, len(snapshots), RUNS), flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()

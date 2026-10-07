"""scankit: the part of a scanner that is the same every time.

A scanner is the program behind the Train Run Tracker's Scan view: it runs text through a trained network and answers
the questions in the scan contract (docs/scan.md). Most of one does not depend on the network: the server and its
routes and errors, finding saved weights and loading them, remembering recent runs, turning a step's numbers into what
the view draws, setting two texts against each other, and taking a unit apart. This file carries that part.

What an experiment writes is a class with four functions, and one line to serve it:

    import scankit

    class Mine(scankit.Scanner):
        name = "My network"
        def graph(self, snapshot): ...     # the steps, for the drawing: {"title", "groups", "nodes", "edges"}
        def tokens(self, text): ...        # [(token id, its text), ...]
        def run(self, ids, snapshot): ...  # {node id: numbers, one row per token}
        def load(self, path): ...          # read one file of saved weights; what it returns is kept as snapshot.data

    scankit.serve(Mine(), runs="~/run-tracker-data/runs", port=8790)

Numbers can be NumPy arrays, PyTorch tensors, or anything NumPy can convert. This file needs NumPy and nothing else,
and can be installed (pip install run-tracker-scankit) or copied beside an experiment.

Two helpers save more: hooked() collects the outputs of named layers of a PyTorch model, so the forward pass is not
written out by hand, and SparseAutoencoder gives a features step, a unit's breakdown and the list of what leans on a
unit from three tables.

    python scankit.py check http://127.0.0.1:8790     # does a running scanner keep to the contract?
"""
import collections, glob, json, os, re, sys, threading, urllib.error, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

PROTOCOL = 1
GRID_UNITS = 48                               # how many units the grid of a step shows at most


class Problem(Exception):
    """A refusal the view can show: one of the contract's codes, a sentence, and the HTTP status that goes with it."""
    def __init__(self, code, message, status=400):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


def array(x):
    """Numbers from anywhere as a NumPy array: a tensor is taken off its device and its graph first."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def r4(v):
    """A number as the replies carry it: four decimal places, and null where there is none (a masked cell)."""
    v = float(v)
    return None if v != v or v in (float("inf"), float("-inf")) else round(v, 4)


def strongest(v, k):
    """The positions of the k largest entries by size, largest first; ties go to the earlier position."""
    return np.argsort(-np.abs(v), kind="stable")[:k]


def spread(v):
    """How many entries carry half, and ninety percent, of a list's squared size."""
    c = np.cumsum(np.sort(v ** 2)[::-1]) / max(float((v ** 2).sum()), 1e-30)
    return int((c < 0.5).sum()) + 1, int((c < 0.9).sum()) + 1


class Snapshot:
    """One file of saved weights that can be loaded. `data` is whatever the scanner's load() returned for it."""
    def __init__(self, id, path, run, step, label):
        self.id, self.path, self.run, self.step, self.label = id, path, run, step, label
        self.state, self.data = "cold", None

    def describe(self):
        return {"id": self.id, "label": self.label, "run": self.run, "step": self.step, "bytes": os.path.getsize(self.path), "state": self.state}


class Scanner:
    """What an experiment fills in. The four functions in the module's description are required; the rest are optional."""
    name, device = "A scanner", "cpu"
    max_tokens = 64
    files = "step_*.pt"                       # which files under the runs folder are saved weights

    def graph(self, snapshot): raise NotImplementedError
    def tokens(self, text): raise NotImplementedError
    def run(self, ids, snapshot): raise NotImplementedError
    def load(self, path): raise NotImplementedError

    def next(self, ids):
        """What the model would write next, as [{"text", "chance"}], or None if it cannot say."""
        return None

    def words(self, snapshot, node, units):
        """For a node marked `described`: for each unit, a short list of words it responds to, or None."""
        return [None] * len(units)

    def usual(self, snapshot, node):
        """Each unit's usual level at a node, if known: the strip is drawn with it taken off."""
        return None

    def unit(self, snapshot, node, unit):
        """More about one unit, as fields of the contract's unit reply (words, made_of, listens_to, used_by)."""
        return {}

    def step_of(self, path):
        """The training step a weights file was saved at: the last number in its name."""
        found = re.findall(r"\d+", os.path.basename(path))
        return int(found[-1]) if found else 0

    def others(self):
        """Weights that belong to no run, as [{"id", "path", "label"}]: a published network to compare with, say."""
        return []


class Kit:
    """A scanner being served: its snapshots, its recent runs, and the contract's routes.

    One lock is held round every question that touches the model, so a slow one makes the others wait. That is right
    for one reader at a desk and would not be for many.
    """

    def __init__(self, scanner, runs, keep=16):
        self.scanner, self.runs, self.keep = scanner, os.path.abspath(os.path.expanduser(runs)), keep
        self.snapshots, self.results, self.lock = {}, collections.OrderedDict(), threading.Lock()

    # ---------- saved weights ----------
    def find(self):
        for path in sorted(glob.glob(os.path.join(self.runs, "**", self.scanner.files), recursive=True)):
            run = os.path.relpath(os.path.dirname(path), self.runs)
            step = self.scanner.step_of(path)
            id = "%s@%d" % (run, step)
            if id not in self.snapshots:
                self.snapshots[id] = Snapshot(id, path, run, step, "%s, step %s" % (os.path.basename(run), format(step, ",")))
        for o in self.scanner.others():
            if o["id"] not in self.snapshots:
                self.snapshots[o["id"]] = Snapshot(o["id"], os.path.expanduser(o["path"]), None, None, o.get("label", o["id"]))
        for id in [i for i, s in self.snapshots.items() if not os.path.exists(s.path)]:
            del self.snapshots[id]

    def default(self):
        """The snapshot opened first: the largest network's latest step among those that belong to a run."""
        self.find()
        ours = [s for s in self.snapshots.values() if s.run]
        return max(ours, key=lambda s: (os.path.getsize(s.path), s.step)).id if ours else next(iter(self.snapshots), None)

    def snapshot(self, id, ready=True):
        self.find()
        if id not in self.snapshots:
            raise Problem("unknown_snapshot", "No snapshot called %r is on this machine." % id, 404)
        if ready and self.snapshots[id].state != "ready":
            raise Problem("not_loaded", "Snapshot %r is not loaded yet. Ask for it with load, then try again." % id, 409)
        return self.snapshots[id]

    def load_now(self, s):
        s.data = self.scanner.load(s.path)
        s.state = "ready"

    def load(self, id):
        """Start loading in the background and say at once what state the snapshot is in."""
        s = self.snapshot(id, ready=False)
        if s.state == "cold":
            s.state = "loading"
            def work():
                with self.lock:
                    try:
                        self.load_now(s)
                    except Exception:
                        s.state = "cold"
                        raise
            threading.Thread(target=work, daemon=True).start()
        return {"snapshot": s.id, "state": s.state}

    # ---------- running text ----------
    def graph(self, snap):
        """The drawing for a snapshot, made once and kept: many questions need a node's fields."""
        if getattr(snap, "drawing", None) is None:
            snap.drawing = self.scanner.graph(snap)
        return snap.drawing

    def spec(self, snap):
        return {n["id"]: n for n in self.graph(snap)["nodes"]}

    def compute(self, text, snap):
        tokens = [tuple(t) if not isinstance(t, dict) else (t["id"], t["text"]) for t in self.scanner.tokens(text)]
        ids = [t[0] for t in tokens]
        if not ids:
            raise Problem("bad_request", "There is no text to run.")
        if len(ids) > self.scanner.max_tokens:
            raise Problem("too_long", "That is %d tokens; this scanner takes at most %d." % (len(ids), self.scanner.max_tokens), 413)
        vals = {k: array(v) for k, v in self.scanner.run(ids, snap).items()}
        name = "r%05x" % (int.from_bytes(os.urandom(3), "big") % 0xfffff)
        self.results[name] = {"snapshot": snap, "ids": ids, "tokens": tokens, "vals": vals}
        while len(self.results) > self.keep:
            self.results.popitem(last=False)
        return name, tokens, vals

    @staticmethod
    def tokens_of(tokens):
        return [{"i": i, "text": text, "id": id} for i, (id, text) in enumerate(tokens)]

    def node_reply(self, x, node, snap, top):
        """One step's numbers as the view draws them: each token's strongest units, the grid, the row sizes."""
        if node.get("per") == "token-pair":
            return {"grid": [[r4(v) for v in row] for row in x]}
        described = node.get("described")
        best = [[int(u) for u in strongest(row, min(top, x.shape[1])) if row[u] != 0] for row in x]
        # the grid's columns: every unit that is among some token's strongest, the strongest anywhere first
        union = sorted({u for row in best for u in row})
        peak = np.abs(x[:, union]).max(0) if union else np.zeros(0)
        units = [union[i] for i in np.argsort(-peak, kind="stable")[:GRID_UNITS]]
        entry = {"size": [r4(v) for v in np.linalg.norm(x, axis=1)],
                 "top": [[[u, r4(x[t, u])] for u in row] for t, row in enumerate(best)],
                 "grid": {"units": units, "values": [[r4(v) for v in x[t, units]] for t in range(len(x))],
                          "words": self.scanner.words(snap, node["id"], units) if described else None}}
        if described:
            for row in entry["top"]:
                for item, w in zip(row, self.scanner.words(snap, node["id"], [u for u, _ in row])):
                    item.append(w)
        if node.get("sparse"):
            entry["on"] = [int(v) for v in (x != 0).sum(1)]
        return entry

    def run(self, text, snap, top=12):
        name, tokens, vals = self.compute(text, snap)
        spec = self.spec(snap)
        out = {"run": name, "snapshot": snap.id, "tokens": self.tokens_of(tokens),
               "nodes": {id: self.node_reply(x, spec[id], snap, top) for id, x in vals.items() if id in spec}}
        nxt = self.scanner.next([t[0] for t in tokens])
        if nxt is not None:
            out["next"] = nxt
        return out

    def result(self, name):
        if name not in self.results:
            raise Problem("unknown_run", "Run %r is no longer kept. Run the text again." % name, 404)
        return self.results[name]

    def values(self, name, node_id, token, start, count):
        r = self.result(name)
        if node_id not in r["vals"]:
            raise Problem("unknown_node", "There is no node called %r." % node_id, 404)
        x = r["vals"][node_id]
        if not 0 <= token < x.shape[0]:
            raise Problem("bad_request", "Token %d is outside the text." % token)
        if self.spec(r["snapshot"]).get(node_id, {}).get("per") == "token-pair":
            return {"values": [r4(v) for v in x[token]]}
        count = max(1, min(count, 4096))
        part = x[token, start:start + count]
        out = {"from": start, "count": len(part), "values": [r4(v) for v in part]}
        usual = self.scanner.usual(r["snapshot"], node_id)
        if usual is not None:
            out["usual"] = [r4(v) for v in array(usual)[start:start + count]]
        return out

    def unit(self, snap, node_id, u):
        widths = {n["id"]: n.get("width") for n in self.graph(snap)["nodes"]}
        if widths.get(node_id) is None:
            raise Problem("unknown_node", "There is no node called %r with units." % node_id, 404)
        if not 0 <= u < widths[node_id]:
            raise Problem("unknown_unit", "%s has units 0 to %d." % (node_id, widths[node_id] - 1), 404)
        out = {"node": node_id, "unit": u}
        out.update(self.scanner.unit(snap, node_id, u) or {})
        return out

    # ---------- two texts against each other ----------
    def differ(self, a, b, snap, node, top):
        """Units on for one row and off or much weaker for the other, and units both have, as lists for the reply."""
        gap = a - b
        only_a = [int(u) for u in np.argsort(-gap, kind="stable")[:top] if gap[u] > 0 and abs(b[u]) < 0.25 * abs(a[u])]
        only_b = [int(u) for u in np.argsort(gap, kind="stable")[:top] if gap[u] < 0 and abs(a[u]) < 0.25 * abs(b[u])]
        both = [int(u) for u in np.argsort(-np.minimum(np.abs(a), np.abs(b)), kind="stable")[:top] if a[u] != 0 and b[u] != 0]
        every = only_a + only_b + both
        w = dict(zip(every, self.scanner.words(snap, node["id"], every))) if node.get("described") and every else {}
        return {"only_a": [[u, r4(a[u]), w.get(u)] for u in only_a], "only_b": [[u, r4(b[u]), w.get(u)] for u in only_b],
                "both": [[u, r4(a[u]), r4(b[u]), w.get(u)] for u in both]}

    def contrast(self, text_a, text_b, snap, node_id, top=12):
        spec = self.spec(snap)
        if node_id not in spec or spec[node_id].get("per") != "token":
            raise Problem("unknown_node", "Contrast needs a node with one row per token; %r is not one." % node_id, 404)
        na, ta, va = self.compute(text_a, snap)
        nb, tb, vb = self.compute(text_b, snap)
        if len(ta) != len(tb):
            raise Problem("bad_request", "The two texts must split into the same number of tokens: the first has %d and the second %d." % (len(ta), len(tb)))
        xa, xb, node = va[node_id], vb[node_id], spec[node_id]
        per_pair = [dict({"same_text": ta[t][0] == tb[t][0], "distance": r4(np.linalg.norm(xa[t] - xb[t]))}, **self.differ(xa[t], xb[t], snap, node, top)) for t in range(len(ta))]
        whole = self.differ(xa.sum(0), xb.sum(0), snap, node, top)
        whole.pop("both")
        return {"run_a": na, "run_b": nb, "node": node_id, "tokens_a": self.tokens_of(ta), "tokens_b": self.tokens_of(tb),
                "pairs": [[t, t] for t in range(len(ta))], "per_pair": per_pair, "whole": whole,
                "first_difference": next((t for t in range(len(ta)) if ta[t][0] != tb[t][0]), None)}

    # ---------- the routes ----------
    def health(self):
        s = self.scanner
        return {"ok": True, "protocol": PROTOCOL, "name": s.name, "device": s.device, "max_tokens": s.max_tokens,
                "supports": ["run", "values", "unit", "contrast", "load"]}

    def list(self):
        self.find()
        return {"snapshots": [s.describe() for s in self.snapshots.values()], "default": self.default(), "files": self.scanner.files}

    def answer(self, method, path, q, body):
        one = lambda k, d=None: (q.get(k) or [d])[0]
        route = path.replace("/scan/v%d/" % PROTOCOL, "", 1)
        if route == "health": return self.health()
        if route == "snapshots": return self.list()
        if route == "load" and method == "POST": return self.load(body.get("snapshot"))
        if route == "steer": raise Problem("not_loaded", "This scanner cannot yet set a unit by hand.", 501)
        with self.lock:
            if route == "graph": return self.graph(self.snapshot(one("snapshot") or self.default()))
            if route == "run" and method == "POST":
                return self.run(body.get("text", ""), self.snapshot(body.get("snapshot") or self.default()), int(body.get("top", 12)))
            if route == "values":
                return self.values(one("run"), one("node"), int(one("token", 0)), int(one("from", 0)), int(one("count", 4096)))
            if route == "unit":
                return self.unit(self.snapshot(one("snapshot") or self.default()), one("node"), int(one("unit", -1)))
            if route == "contrast" and method == "POST":
                snap = self.snapshot(body.get("snapshot") or self.default())
                node = body.get("node") or next((n["id"] for n in self.graph(snap)["nodes"] if n.get("sparse")), None)
                return self.contrast(body.get("a", ""), body.get("b", ""), snap, node, int(body.get("top", 12)))
        raise Problem("bad_request", "There is no route %s %s." % (method, path), 404)

    def handler(kit):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def reply(self, method):
                url, body = urlparse(self.path), {}
                try:
                    if method == "POST":
                        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                        try:
                            body = json.loads(raw or b"{}")
                        except ValueError:
                            raise Problem("bad_request", "The request body is not JSON.")
                    out, status = kit.answer(method, url.path, parse_qs(url.query), body), 200
                except Problem as e:
                    out, status = {"error": {"code": e.code, "message": e.message}}, e.status
                except (ValueError, TypeError) as e:
                    out, status = {"error": {"code": "bad_request", "message": str(e)}}, 400
                except Exception as e:
                    out, status = {"error": {"code": "internal", "message": "%s: %s" % (type(e).__name__, e)}}, 500
                data = json.dumps(out).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self.reply("GET")

            def do_POST(self):
                self.reply("POST")
        return Handler


def server(scanner, runs, port=8790):
    """The scanner as an HTTP server on 127.0.0.1, not yet started. Port 0 lets the system choose one."""
    kit = Kit(scanner, runs)
    kit.find()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), kit.handler())
    httpd.kit = kit
    return httpd


def serve(scanner, runs, port=8790):
    """Serve the scanner on this machine until stopped."""
    httpd = server(scanner, runs, port)
    print("scanner on http://127.0.0.1:%d with %d snapshots under %s" % (httpd.server_address[1], len(httpd.kit.snapshots), httpd.kit.runs), flush=True)
    httpd.serve_forever()


def record(scanner, runs, folder, text, other, node=None, token=0):
    """Write recorded replies for every route into a folder, one JSON file each, for building a view against."""
    kit = Kit(scanner, runs)
    os.makedirs(folder, exist_ok=True)
    def keep(name, value):
        with open(os.path.join(folder, name + ".json"), "w") as f:
            json.dump(value, f, indent=1)
    keep("snapshots_before_load", kit.list())
    s = kit.snapshot(kit.default(), ready=False)
    kit.load_now(s)
    graph = kit.graph(s)
    node = node or next((n["id"] for n in graph["nodes"] if n.get("sparse")), graph["nodes"][-1]["id"])
    r = kit.run(text, s)
    keep("health", kit.health()); keep("snapshots", kit.list()); keep("load", {"snapshot": s.id, "state": "ready"}); keep("graph", graph); keep("run", r)
    keep("values", kit.values(r["run"], node, token, 0, 4096))
    top = r["nodes"][node]["top"][token]
    if top:
        keep("unit", kit.unit(s, node, top[0][0]))
    keep("contrast", kit.contrast(text, other, s, node))
    return s.id


# ---------- helpers for common kinds of network ----------
def hooked(layers, call):
    """The outputs of named layers of a PyTorch model, collected while `call()` runs it.

        vals = scankit.hooked({"b0.mlp.act": block.mlp.act, "b0.ln2": block.ln_2}, lambda: model(ids))

    `layers` maps a node id to a module. Each module's output is caught by a forward hook, so the forward pass does not
    have to be written out step by step. If a module returns several things, the first is kept.

    What comes back is an ordinary dict, so a step that is not any one layer's output (an output split in three, one
    head's attention grid) is worked out from the caught values and added to it before returning from run().
    """
    caught, hooks = {}, []
    for id, module in layers.items():
        hooks.append(module.register_forward_hook(lambda m, i, o, id=id: caught.__setitem__(id, o[0] if isinstance(o, (tuple, list)) else o)))
    try:
        call()
    finally:
        for h in hooks:
            h.remove()
    return caught


class SparseAutoencoder:
    """A sparse autoencoder read off one step of a network, as three steps of its own and everything a unit page needs.

        features = ReLU((x - usual) @ detector.T + detector_bias)        x is the step it reads, one row per token
        rebuild  = features @ pattern.T + usual

    detector      one row per feature: what it listens for among the units it reads   (features, read units)
    pattern       one column per feature: what it adds back                            (read units, features)
    usual         one number per read unit, taken off before detecting
    reads         the id of the node it reads
    ids           the ids of its three nodes: the difference, the features, the rebuild
    probe, probe_words, test
                  to say what a feature responds to: rows of the read step for a list of inputs (probe), the text of
                  each (probe_words), and the test in words. A feature's words are the inputs that fire it hardest.
                  probe may be a function that returns the rows, called the first time words are wanted, since they
                  can be large and slow to make. Leave all three out and features simply have no words.
    cache         a file to keep the words in once worked out, so the next start does not work them out again:
                  usually the weights file's own path with ".words.json" added
    """

    def __init__(self, detector, detector_bias, pattern, usual, reads, ids=("sae.in", "sae.features", "sae.rebuild"),
                 probe=None, probe_words=None, test=None, cache=None):
        f = lambda x: array(x).astype(np.float32)
        self.detector, self.detector_bias, self.pattern, self.usual_level = f(detector), f(detector_bias), f(pattern), f(usual)
        self.reads, self.ids = reads, tuple(ids)
        self.n = self.detector.shape[0]
        self.lengths = np.linalg.norm(self.pattern, axis=0)             # each feature's pattern, to compare patterns on one scale
        self.probe, self.probe_words, self.test, self.cache = probe, probe_words, test, cache
        self.words_of = {}
        if cache and os.path.exists(cache):
            try:
                with open(cache) as fh:
                    kept = json.load(fh)
                if kept.get("test") == test and kept.get("features") == self.n:
                    self.words_of = {int(u): w for u, w in kept["words"].items()}
            except (ValueError, KeyError, OSError):
                pass                                                    # a damaged cache is worked out again

    def run(self, x):
        """The three steps for the rows of the step it reads."""
        x_in = array(x) - self.usual_level
        feats = np.maximum(x_in @ self.detector.T + self.detector_bias, 0)
        return {self.ids[0]: x_in, self.ids[1]: feats, self.ids[2]: feats @ self.pattern.T + self.usual_level}

    def words(self, units, n=4):
        """For each feature, the probe inputs that fire it hardest, or None where there is no probe or none fire it."""
        if self.probe is None:
            return [None] * len(units)
        todo = [u for u in dict.fromkeys(units) if u not in self.words_of]
        if todo and callable(self.probe):
            self.probe = self.probe()
        if todo:
            self.probe = array(self.probe)
        for i in range(0, len(todo), 64):
            part = todo[i:i + 64]
            fired = np.maximum((self.probe - self.usual_level) @ self.detector[part].T + self.detector_bias[part], 0).T
            for u, row in zip(part, fired):
                best = np.argsort(-row, kind="stable")[:8]
                self.words_of[u] = [[self.probe_words[int(t)], round(float(row[t]), 3)] for t in best if row[t] > 0]
        if todo and self.cache:
            try:
                tmp = self.cache + ".tmp"
                with open(tmp, "w") as fh:
                    json.dump({"test": self.test, "features": self.n, "words": {str(u): w for u, w in self.words_of.items()}}, fh)
                os.replace(tmp, self.cache)
            except OSError:
                pass                                                    # nowhere to keep them: they are worked out again next time
        return [[w for w, _ in self.words_of[u][:n]] or None for u in units]

    def usual(self, node):
        return self.usual_level if node == self.reads else None

    def unit(self, node, u):
        """A feature's breakdown, or for a unit it reads or rebuilds, the features that lean on it."""
        out = {}
        if node == self.ids[1]:
            pattern = self.pattern[:, u] / self.lengths[u]
            detector = self.detector[u] / np.linalg.norm(self.detector[u])
            for key, v in (("made_of", pattern), ("listens_to", detector)):
                half, ninety = spread(v)
                out[key] = {"node": self.reads, "top": [[int(i), r4(v[i])] for i in strongest(v, 12)], "half": half, "ninety": ninety}
            if self.probe is not None:
                self.words([u])
                out["words"], out["words_test"] = self.words_of[u], self.test
            out["agreement"] = r4(float(pattern @ detector))
        elif node in (self.reads, self.ids[0], self.ids[2]):
            lean = self.pattern[u] / self.lengths
            best = [int(j) for j in strongest(lean, 8)]
            out["used_by"] = {"node": self.ids[1], "count": int((np.abs(lean) > 0.1).sum()), "threshold": 0.1,
                              "top": [[j, r4(lean[j]), w] for j, w in zip(best, self.words(best))]}
        return out


# ---------- does a scanner keep to the contract? ----------
def check(url, text="the sky is blue", other="the sea is blue", say=print):
    """Ask a running scanner every question in the contract and say what is wrong with its answers. True if nothing is."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def ask(route, query="", body=None):
        req = urllib.request.Request("%s/scan/v%d/%s%s" % (url.rstrip("/"), PROTOCOL, route, "?" + query if query else ""),
                                     data=None if body is None else json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            with opener.open(req, timeout=180) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")
    wrong = []
    def must(ok, what):
        say(("  ok     " if ok else "  WRONG  ") + what)
        if not ok:
            wrong.append(what)
    q = urllib.request.quote

    _, h = ask("health")
    must(h.get("protocol") == PROTOCOL and isinstance(h.get("name"), str), "health gives protocol %d and a name" % PROTOCOL)
    must(isinstance(h.get("max_tokens"), int) and isinstance(h.get("supports"), list), "health gives max_tokens and supports")
    _, s = ask("snapshots")
    snaps = s.get("snapshots") or []
    must(bool(snaps) and all({"id", "label", "run", "state"} <= set(x) for x in snaps), "snapshots lists at least one, each with id, label, run and state")
    must(s.get("default") in [x.get("id") for x in snaps] and isinstance(s.get("files"), str), "snapshots names a default that is in the list, and the files pattern")
    snap = s.get("default") or (snaps[0]["id"] if snaps else "")
    st, e = ask("graph", "snapshot=" + q("no such snapshot", safe=""))
    must(st == 404 and (e.get("error") or {}).get("code") == "unknown_snapshot", "an unknown snapshot is refused with unknown_snapshot, 404")
    ask("load", body={"snapshot": snap})
    import time
    for _ in range(180):
        if next((x for x in ask("snapshots")[1].get("snapshots", []) if x["id"] == snap), {}).get("state") == "ready":
            break
        time.sleep(1)
    must(next((x for x in ask("snapshots")[1]["snapshots"] if x["id"] == snap), {}).get("state") == "ready", "load makes the default snapshot ready")

    _, g = ask("graph", "snapshot=" + q(snap, safe=""))
    nodes = g.get("nodes") or []
    must(bool(nodes) and all({"id", "label", "kind", "lane", "order"} <= set(n) for n in nodes), "graph gives nodes with id, label, kind, lane and order")
    must(all(isinstance(n["lane"], int) and isinstance(n["order"], int) for n in nodes) and len({(n["lane"], n["order"]) for n in nodes}) == len(nodes),
         "lane and order are whole numbers, and no two nodes share both")
    ids = {n["id"] for n in nodes}
    must(all(e_.get("from") in ids and e_.get("to") in ids for e_ in g.get("edges", [])), "every edge joins two nodes that exist")
    must(all(n.get("group") is None or n["group"] in {x["id"] for x in g.get("groups", [])} for n in nodes), "every node's group exists")

    _, r = ask("run", body={"text": text, "snapshot": snap, "top": 6})
    toks = r.get("tokens") or []
    must(bool(toks) and [t.get("i") for t in toks] == list(range(len(toks))), "run gives tokens numbered 0 upward with no gaps")
    by = {n["id"]: n for n in nodes}
    shaped = True
    for id, n in (r.get("nodes") or {}).items():
        if id not in by:
            shaped = False
        elif by[id].get("per") == "token-pair":
            shaped &= isinstance(n.get("grid"), list) and len(n["grid"]) == len(toks) and all(len(row) == len(toks) for row in n["grid"])
        else:
            grid = n.get("grid") or {}
            shaped &= len(n.get("size", [])) == len(toks) and len(n.get("top", [])) == len(toks)
            shaped &= len(grid.get("values", [])) == len(toks) and all(len(row) == len(grid.get("units", [])) for row in grid.get("values", []))
            if by[id].get("sparse"):
                shaped &= len(n.get("on", [])) == len(toks)
    must(bool(r.get("nodes")) and bool(shaped), "every node in the run reply is in the graph and has size, top and grid of the right shape")
    per_token = [n for n in nodes if n.get("per", "token") == "token" and n["id"] in (r.get("nodes") or {})]
    if per_token and r.get("run"):
        node = next((n for n in per_token if n.get("sparse")), per_token[-1])
        _, v = ask("values", "run=%s&node=%s&token=0&from=0&count=8" % (r["run"], q(node["id"], safe="")))
        must(isinstance(v.get("values"), list) and v.get("count") == len(v["values"]) <= 8, "values gives a slice of one token's row")
        st, e = ask("values", "run=nosuchrun&node=%s&token=0" % q(node["id"], safe=""))
        must(st == 404 and (e.get("error") or {}).get("code") == "unknown_run", "an unknown run is refused with unknown_run, 404")
        st, u = ask("unit", "snapshot=%s&node=%s&unit=0" % (q(snap, safe=""), q(node["id"], safe="")))
        must(st == 200 and u.get("node") == node["id"] and u.get("unit") == 0, "unit answers for unit 0 of a node")
        st, e = ask("unit", "snapshot=%s&node=%s&unit=%d" % (q(snap, safe=""), q(node["id"], safe=""), 10 ** 9))
        must(st == 404 and (e.get("error") or {}).get("code") == "unknown_unit", "a unit past the end is refused with unknown_unit, 404")
        st, c = ask("contrast", body={"a": text, "b": other, "snapshot": snap, "node": node["id"], "top": 6})
        ok = st == 200 and len(c.get("pairs", [])) == len(c.get("per_pair", [])) == len(c.get("tokens_a", []))
        must(ok and all({"same_text", "distance", "only_a", "only_b", "both"} <= set(p) for p in c.get("per_pair", [])) and {"only_a", "only_b"} <= set(c.get("whole", {})),
             "contrast lines the two texts up and says what is only in each")
        st, e = ask("contrast", body={"a": text, "b": text + " " + text, "snapshot": snap, "node": node["id"]})
        must(st == 400 and (e.get("error") or {}).get("code") == "bad_request", "texts of different lengths are refused with bad_request")
    st, e = ask("run", body={"text": " ".join([text] * 400), "snapshot": snap})
    must(st == 413 and (e.get("error") or {}).get("code") == "too_long", "text past the limit is refused with too_long, 413")
    say("%s: %s" % (url, "keeps to the contract" if not wrong else "%d things wrong" % len(wrong)))
    return not wrong


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "check":
        sys.exit(0 if check(sys.argv[2]) else 1)
    sys.exit("usage: python scankit.py check http://127.0.0.1:8790")

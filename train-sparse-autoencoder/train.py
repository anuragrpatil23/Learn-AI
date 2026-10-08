"""Train a sparse autoencoder on the MLP neurons of GPT-2 small's first block, and watch it form.

The method is the one in Towards Monosemanticity (Anthropic, 2023): a two-layer network, wider in
the middle, trained to rebuild its own input with a penalty on how much the middle fires.

  f     = ReLU( W_enc (x - b_dec) + b_enc )
  x_hat = W_dec f + b_dec
  loss  = |x - x_hat|^2  +  lam * sum(f)

One training example is the 3,072 MLP neurons (after GELU) at one word position. Rows are made on
the fly: text goes through GPT-2's embedding and first block only, the rows are held in a buffer,
and batches are drawn from the buffer at random so that a batch mixes many documents.

Every --log-every steps a line is written to log.jsonl with the numbers worth watching, and with
what the strongest feature for " sky" and for " the" currently responds to.

The run is recorded with the Weights & Biases client if it is installed, always offline: nothing is sent
anywhere, and the run file stays in the run's folder to be copied and read with the run tracker
(github.com/anuragrpatil23/run-tracker). Without the client, tracker.py, a copy of that project's own
writer, records the same things as plain files. --logger chooses one outright.

Usage: python train.py --tokens '/path/edufineweb_train_*.npy' --features 8192 --lam 0.02 --out runs/f8192
"""
import argparse, glob, json, math, os, time
import numpy as np
import torch
import tracker
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

p = argparse.ArgumentParser()
p.add_argument("--tokens", required=True, help="glob of .npy files of GPT-2 token ids (uint16), as written by build-nanogpt/fineweb.py")
p.add_argument("--out", required=True)
p.add_argument("--features", type=int, default=8192)
p.add_argument("--lam", type=float, default=0.02)
p.add_argument("--rows", type=float, default=1e8, help="how many rows to train on in total")
p.add_argument("--batch", type=int, default=4096)
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--context", type=int, default=128, help="tokens of text GPT-2 sees at once")
p.add_argument("--buffer", type=int, default=2 ** 19, help="rows held for shuffling")
p.add_argument("--log-every", type=int, default=200)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--device", default=None, help="cuda, mps or cpu; chosen automatically if left out")
p.add_argument("--project", default="sae-gpt2-mlp0", help="the project the run is filed under in the viewer")
p.add_argument("--logger", default="auto", choices=["auto", "wandb", "files"], help="wandb (offline) if installed, else plain files; or say which")
p.add_argument("--checkpoint-hours", type=float, default=2.0, help="how often to save everything needed to carry on, should the job stop")
args = p.parse_args()

torch.manual_seed(args.seed); np.random.seed(args.seed)
dev = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
os.makedirs(args.out, exist_ok=True)
N_IN, N_FEAT = 3072, args.features

# ---------- GPT-2: only the embedding and the first block are ever run ----------
tok = GPT2TokenizerFast.from_pretrained("gpt2")
gpt = GPT2LMHeadModel.from_pretrained("gpt2").eval().to(dev).transformer
grab = {}
gpt.h[0].mlp.act.register_forward_hook(lambda m, i, o: grab.__setitem__("act", o))

@torch.no_grad()
def mlp_rows(ids):
    """ids: (B, T) token ids. Returns the first block's MLP neurons, one row per position: (B*T, 3072)."""
    pos = torch.arange(ids.shape[1], device=dev)
    gpt.h[0](gpt.wte(ids) + gpt.wpe(pos))
    return grab["act"].reshape(-1, N_IN)

# ---------- the text, as a stream of rows ----------
shards = sorted(glob.glob(args.tokens))
assert shards, "no token files match " + args.tokens

def row_stream(skip=0):
    """Yields blocks of rows, about 32,768 at a time, reading the token files in order, over and over.
    `skip` is how many tokens to pass over first, so that a run carried on from a checkpoint reads new text."""
    per = max(1, 32768 // args.context)
    while True:
        for path in shards:
            toks = np.load(path, mmap_mode="r")
            n = (len(toks) // args.context) * args.context
            if skip >= n:
                skip -= n; continue
            for start in range(skip - skip % (per * args.context), n - per * args.context + 1, per * args.context):
                chunk = torch.from_numpy(toks[start:start + per * args.context].astype(np.int64))
                yield mlp_rows(chunk.view(per, args.context).to(dev))
            skip = 0

class Buffer:
    """Holds rows from many documents and hands them out in random order, refilling as it empties."""
    def __init__(self, stream, size):
        self.stream, self.size = stream, size
        self.rows = torch.empty(0, N_IN, device=dev)
        self.refill()
    def refill(self):
        parts, have = [self.rows], self.rows.shape[0]
        while have < self.size:
            r = next(self.stream); parts.append(r); have += r.shape[0]
        self.rows = torch.cat(parts)[torch.randperm(have, device=dev)]
    def batch(self, n):
        if self.rows.shape[0] < self.size // 2:
            self.refill()
        out, self.rows = self.rows[:n], self.rows[n:]
        return out

# A run that stopped part way carries on from its last checkpoint, reading on from where it had got to in the text.
CKPT = os.path.join(args.out, "checkpoint.pt")
resume = torch.load(CKPT, map_location=dev) if os.path.exists(CKPT) else None
start_step = resume["step"] if resume else 0
buf = Buffer(row_stream(skip=start_step * args.batch), args.buffer)   # one row per token, so rows used = tokens read

# ---------- the sparse autoencoder ----------
W_dec = torch.randn(N_IN, N_FEAT, device=dev)
W_dec /= W_dec.norm(dim=0, keepdim=True)                     # each feature's pattern starts, and stays, at length 1
W_enc = W_dec.T.clone()                                      # the encoder starts as the decoder turned over
b_enc = torch.zeros(N_FEAT, device=dev)
b_dec = buf.rows[:65536].mean(0).clone()                     # each neuron's usual level starts at its average
if resume:
    for t, name in ((W_enc, "W_enc"), (b_enc, "b_enc"), (W_dec, "W_dec"), (b_dec, "b_dec")):
        t.copy_(resume[name])
params = [W_enc.requires_grad_(), b_enc.requires_grad_(), W_dec.requires_grad_(), b_dec.requires_grad_()]
opt = torch.optim.Adam(params, lr=args.lr)
if resume:
    opt.load_state_dict(resume["opt"])

def forward(x):
    f = torch.relu((x - b_dec) @ W_enc.T + b_enc)
    return f @ W_dec.T + b_dec, f

# ---------- what to watch ----------
PROMPT = "Q: Why is the sky blue?\nA:"
p_ids = tok(PROMPT, return_tensors="pt").input_ids.to(dev)
p_rows = mlp_rows(p_ids).clone()
p_words = [tok.decode(i) for i in p_ids[0]]
frame = tok("Q: Why is the").input_ids
vocab_rows = torch.cat([mlp_rows(torch.cat([torch.tensor(frame).repeat(len(v), 1), v[:, None]], dim=1).to(dev)).view(len(v), -1, N_IN)[:, -1].half()
                        for v in torch.arange(50257).split(2048)])   # every token placed after "Q: Why is the": (50257, 3072)

@torch.no_grad()
def probe(word):
    """The strongest feature for one word of the prompt, and the tokens that fire that feature hardest."""
    t = p_words.index(word)
    _, f = forward(p_rows[t:t + 1])
    j = int(f[0].argmax())
    over_vocab = torch.relu((vocab_rows.float() - b_dec) @ W_enc[j] + b_enc[j])
    return {"feature": j, "fired": round(float(f[0, j]), 3), "on": int((f[0] > 0).sum()),
            "responds_to": [tok.decode([int(i)]) for i in over_vocab.topk(6).indices]}

fired_ever = resume["fired_ever"].to(dev) if resume else torch.zeros(N_FEAT, dtype=torch.bool, device=dev)
fired_window = torch.zeros(N_FEAT, device=dev); rows_window = 0
steps = int(args.rows // args.batch)
held_out = buf.batch(args.batch).clone()                      # one batch set aside before training starts, for the val/ numbers

# ---------- the record of the run ----------
# Names follow the usual convention: train/ for what is measured on the batch being trained on, val/ for the batch set
# aside, features/ for the state of the dictionary. The viewer gives each its own section and draws train/ and val/ of
# the same name on one chart.
ABOUT = {   # a sentence each, shown under the charts
    "train/loss": "What is minimised: how far the rebuilt input is from the real one, plus lam times how much the features fire.",
    "train/rebuild_error": "How far the rebuilt input is from the real one. The first term of the loss.",
    "train/not_rebuilt": "The same, as a share of what there was to explain. Lower means more of the input is rebuilt.",
    "train/total_firing": "How much the features fire in total, per word. The second term of the loss, before lam.",
    "val/loss": "The loss on a batch set aside before training. If it rises while train/loss falls, the network is fitting its batches, not the data.",
    "features/on": "How many features are on per word, out of all of them.",
    "features/never_fired": "Features that have not fired once since training began.",
    "features/silent_lately": "Features that did not fire at all since the last logged line.",
    "features/busy": "Features that were on for more than 1 word in 10 since the last logged line.",
    "lr": "The learning rate.",
    "grad_norm": "The size of the gradient at the step before each logged line. A spike is a step that could throw the weights.",
    "rows_per_second": "How fast training has gone so far.",
    "sky": "The strongest feature for the word “ sky” in the prompt, and the tokens that fire it hardest.",
    "the": "The strongest feature for the word “ the” in the prompt, and the tokens that fire it hardest.",
    "section:train": "Measured on the batch being trained on.",
    "section:val": "Measured on one batch set aside before training.",
    "section:features": "The state of the dictionary: how many features are in use, and how many have gone quiet.",
}
CONFIG = {**vars(args), "device": dev}
NAME, GROUP = os.path.basename(os.path.normpath(args.out)), "f%d" % N_FEAT


class Record:
    """Where the run is written: the W&B client, offline, or tracker.py's plain files."""
    def __init__(self):
        self.wandb = self.files = None
        if args.logger != "files":
            try:
                import wandb
                self.wandb = wandb.init(project=args.project, name=NAME, group=GROUP, dir=args.out, mode="offline",
                                        config={**CONFIG, "about": ABOUT}, tags=["sae", "gpt2-small", "mlp0"])
            except ImportError:
                if args.logger == "wandb":
                    raise
        if self.wandb is None:
            self.files = tracker.start(args.out, CONFIG, total=steps, project=args.project, group=GROUP)
            self.files.describe(ABOUT)
        # the snapshots of the weights go in the run's own folder, beside its record
        self.folder = os.path.dirname(self.wandb.dir) if self.wandb else args.out

    def log(self, step, rec):
        self.wandb.log(rec, step=step) if self.wandb else self.files.log(step, **rec)

    def saved(self, path, step):
        if self.files:
            self.files.save(path, kind="weights", step=step)

    def finish(self):
        self.wandb.finish() if self.wandb else self.files.finish()


record = Record()
grad_norm = None                                              # the size of the latest gradient, set in the loop


def terms(x, x_hat, f):
    """The loss and its parts on one batch, as plain numbers."""
    err = ((x - x_hat) ** 2).sum(-1); size = ((x - x.mean(0)) ** 2).sum(-1)
    return {"loss": round(float(err.mean()) + args.lam * float(f.sum(-1).mean()), 4),   # what is minimised: both terms together
            "rebuild_error": round(float(err.mean()), 4),                               # first term of the loss
            "not_rebuilt": round(float(err.sum() / size.sum()), 4),                     # the same, as a share of what there was to explain
            "total_firing": round(float(f.sum(-1).mean()), 4)}                          # second term, before lam


def write_log(step, rows_seen, x, x_hat, f, t0):
    global fired_window, rows_window
    rate = fired_window / max(rows_window, 1)
    elapsed = time.time() - t0
    rec = {"rows": int(rows_seen), "minutes": round(elapsed / 60, 2), "lr": opt.param_groups[0]["lr"]}
    if rows_seen:
        rec["rows_per_second"] = round(rows_seen / max(elapsed, 1e-9))
    if grad_norm is not None:
        rec["grad_norm"] = round(grad_norm, 4)
    rec.update({"train/" + k: v for k, v in terms(x, x_hat, f).items()})
    rec.update({"val/" + k: v for k, v in terms(held_out, *forward(held_out)).items()})
    rec.update({"features/on": round(float((f > 0).sum(-1).float().mean()), 1),         # per word, out of N_FEAT
                "features/never_fired": int((~fired_ever).sum()),                       # since the start of training
                "features/silent_lately": int((rate == 0).sum()),                       # no firing since the last log line
                "features/busy": int((rate > 0.1).sum()),                               # on for more than 1 word in 10 since the last log line
                "sky": probe(" sky"), "the": probe(" the")})
    record.log(step, rec)
    print(json.dumps({"step": step, **rec}), flush=True)
    fired_window = torch.zeros(N_FEAT, device=dev); rows_window = 0

# ---------- training ----------
saves = {0, steps // 100, steps // 10, steps}                  # snapshots, to look back at how the features formed
t0 = time.time() - (resume["seconds"] if resume else 0)       # minutes keep counting across a restart
last_ckpt = time.time()

def checkpoint(step):
    """Everything needed to carry on from this step. Written to a new file first, so a job stopped mid-write leaves the old one."""
    tmp = CKPT + ".tmp"
    torch.save({"step": step, "W_enc": W_enc.detach(), "b_enc": b_enc.detach(), "W_dec": W_dec.detach(), "b_dec": b_dec.detach(),
                "opt": opt.state_dict(), "fired_ever": fired_ever, "seconds": time.time() - t0}, tmp)
    os.replace(tmp, CKPT)

if resume:
    print("carrying on from step", start_step, flush=True)
for step in range(start_step, steps + 1):
    x = buf.batch(args.batch)
    x_hat, f = forward(x)
    loss = ((x - x_hat) ** 2).sum(-1).mean() + args.lam * f.sum(-1).mean()
    with torch.no_grad():
        on = (f > 0)
        fired_ever |= on.any(0); fired_window += on.sum(0); rows_window += x.shape[0]
        if step % args.log_every == 0 or step == steps:
            write_log(step, step * args.batch, x, x_hat, f, t0)
        if step in saves:
            snapshot = os.path.join(record.folder, "step_%07d.pt" % step)
            torch.save({"W_enc": W_enc.detach().half().cpu(), "b_enc": b_enc.detach().cpu(), "W_dec": W_dec.detach().half().cpu(),
                        "b_dec": b_dec.detach().cpu(), "step": step}, snapshot)
            record.saved(snapshot, step)
    if step == steps:
        break
    if time.time() - last_ckpt > args.checkpoint_hours * 3600:
        checkpoint(step); last_ckpt = time.time()
    opt.zero_grad()
    loss.backward()
    with torch.no_grad():                                      # do not let the update change a pattern's length, only its direction
        W_dec.grad -= (W_dec.grad * W_dec).sum(0, keepdim=True) * W_dec
        if (step + 1) % args.log_every == 0 or step + 1 == steps:              # only where the next line will report it
            grad_norm = float(torch.sqrt(sum((q.grad ** 2).sum() for q in params)))
    opt.step()
    with torch.no_grad():
        W_dec /= W_dec.norm(dim=0, keepdim=True)
record.finish()

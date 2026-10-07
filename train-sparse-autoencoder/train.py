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

Usage: python train.py --tokens '/path/edufineweb_train_*.npy' --features 8192 --lam 0.02 --out runs/f8192
"""
import argparse, glob, json, math, os, time
import numpy as np
import torch
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

def row_stream():
    """Yields blocks of rows, about 32,768 at a time, reading the token files in order, over and over."""
    per = max(1, 32768 // args.context)
    while True:
        for path in shards:
            toks = np.load(path, mmap_mode="r")
            n = (len(toks) // args.context) * args.context
            for start in range(0, n - per * args.context + 1, per * args.context):
                chunk = torch.from_numpy(toks[start:start + per * args.context].astype(np.int64))
                yield mlp_rows(chunk.view(per, args.context).to(dev))

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

buf = Buffer(row_stream(), args.buffer)

# ---------- the sparse autoencoder ----------
W_dec = torch.randn(N_IN, N_FEAT, device=dev)
W_dec /= W_dec.norm(dim=0, keepdim=True)                     # each feature's pattern starts, and stays, at length 1
W_enc = W_dec.T.clone()                                      # the encoder starts as the decoder turned over
b_enc = torch.zeros(N_FEAT, device=dev)
b_dec = buf.rows[:65536].mean(0).clone()                     # each neuron's usual level starts at its average
params = [W_enc.requires_grad_(), b_enc.requires_grad_(), W_dec.requires_grad_(), b_dec.requires_grad_()]
opt = torch.optim.Adam(params, lr=args.lr)

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

fired_ever = torch.zeros(N_FEAT, dtype=torch.bool, device=dev)
fired_window = torch.zeros(N_FEAT, device=dev); rows_window = 0
log = open(os.path.join(args.out, "log.jsonl"), "a")
json.dump({"args": vars(args), "device": dev}, open(os.path.join(args.out, "config.json"), "w"))

def write_log(step, rows_seen, x, x_hat, f, t0):
    global fired_window, rows_window
    err = ((x - x_hat) ** 2).sum(-1); size = ((x - x.mean(0)) ** 2).sum(-1)
    rate = fired_window / max(rows_window, 1)
    rec = {"step": step, "rows": int(rows_seen), "minutes": round((time.time() - t0) / 60, 2),
           "rebuild_error": round(float(err.mean()), 4),                   # first term of the loss
           "not_rebuilt": round(float(err.sum() / size.sum()), 4),         # the same, as a share of what there was to explain
           "total_firing": round(float(f.sum(-1).mean()), 4),              # second term, before lam
           "features_on": round(float((f > 0).sum(-1).float().mean()), 1), # per word, out of N_FEAT
           "never_fired": int((~fired_ever).sum()),                        # since the start of training
           "silent_lately": int((rate == 0).sum()),                        # no firing since the last log line
           "busy": int((rate > 0.1).sum()),                                # on for more than 1 word in 10 since the last log line
           "sky": probe(" sky"), "the": probe(" the")}
    log.write(json.dumps(rec) + "\n"); log.flush()
    print(json.dumps(rec), flush=True)
    fired_window = torch.zeros(N_FEAT, device=dev); rows_window = 0

# ---------- training ----------
steps = int(args.rows // args.batch)
saves = {0, steps // 100, steps // 10, steps}                  # snapshots, to look back at how the features formed
t0 = time.time()
for step in range(steps + 1):
    x = buf.batch(args.batch)
    x_hat, f = forward(x)
    loss = ((x - x_hat) ** 2).sum(-1).mean() + args.lam * f.sum(-1).mean()
    with torch.no_grad():
        on = (f > 0)
        fired_ever |= on.any(0); fired_window += on.sum(0); rows_window += x.shape[0]
        if step % args.log_every == 0 or step == steps:
            write_log(step, step * args.batch, x, x_hat, f, t0)
        if step in saves:
            torch.save({"W_enc": W_enc.detach().half().cpu(), "b_enc": b_enc.detach().cpu(), "W_dec": W_dec.detach().half().cpu(),
                        "b_dec": b_dec.detach().cpu(), "step": step}, os.path.join(args.out, "step_%07d.pt" % step))
    if step == steps:
        break
    opt.zero_grad()
    loss.backward()
    with torch.no_grad():                                      # do not let the update change a pattern's length, only its direction
        W_dec.grad -= (W_dec.grad * W_dec).sum(0, keepdim=True) * W_dec
    opt.step()
    with torch.no_grad():
        W_dec /= W_dec.norm(dim=0, keepdim=True)

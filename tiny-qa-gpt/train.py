"""Train a very small GPT that answers a dozen fixed questions, for the interactive drawings.

Same parts as minGPT / GPT-2 (token and position embeddings, layer norm, causal self-attention,
an MLP, a final layer norm, an unembedding), with ReLU as the nonlinearity. It memorises the
question-answer pairs below; it does not know anything else. The point is a real trained network
small enough that every number in it can be drawn and traced.

Writes the weights as JSON (same layout as bbycroft/llm-viz's gpt-nano-sort-model.json).
Usage: python train.py <output.json>
"""
import sys, json, math, base64, itertools
import torch, torch.nn as nn, torch.nn.functional as F

QA = [
    ("why is the sky blue ?", "air scatters blue light"),
    ("why is the sea blue ?", "water absorbs red light"),
    ("why is the grass green ?", "leaves reflect green light"),
    ("why is the snow white ?", "snow reflects all light"),
    ("why is the night dark ?", "the sun is away"),
    ("why is the sunset red ?", "blue light scatters away"),
    ("what color is the sky ?", "blue"),
    ("what color is the sea ?", "blue"),
    ("what color is the grass ?", "green"),
    ("what color is the snow ?", "white"),
    ("where does the sun rise ?", "in the east"),
    ("where does the sun set ?", "in the west"),
]
words = ["<pad>", "<end>"]
for q, a in QA:
    for w in (q + " " + a).split():
        if w not in words: words.append(w)
stoi = {w: i for i, w in enumerate(words)}
seqs = [[stoi[w] for w in q.split()] + [stoi[w] for w in a.split()] + [1] for q, a in QA]
qlen = [len(q.split()) for q, _ in QA]
maxlen = max(len(s) for s in seqs)
BLOCK = maxlen - 1
X = torch.zeros(len(seqs), BLOCK, dtype=torch.long)
Y = torch.full((len(seqs), BLOCK), -100, dtype=torch.long)
for n, s in enumerate(seqs):
    X[n, :len(s) - 1] = torch.tensor(s[:-1])
    for t in range(qlen[n] - 1, len(s) - 1):   # only the answer words and <end> are targets
        Y[n, t] = s[t + 1]


class Attn(nn.Module):
    def __init__(s, C, H):
        super().__init__(); s.c_attn = nn.Linear(C, 3 * C); s.c_proj = nn.Linear(C, C); s.H = H
    def forward(s, x):
        B, T, C = x.shape
        q, k, v = s.c_attn(x).split(C, dim=2)
        sh = lambda z: z.view(B, T, s.H, C // s.H).transpose(1, 2)
        q, k, v = sh(q), sh(k), sh(v)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(C // s.H)
        att = att.masked_fill(torch.tril(torch.ones(T, T)) == 0, float("-inf")).softmax(-1)
        return s.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))

class MLP(nn.Module):
    def __init__(s, C, M):
        super().__init__(); s.c_fc = nn.Linear(C, M); s.c_proj = nn.Linear(M, C)
    def forward(s, x): return s.c_proj(F.relu(s.c_fc(x)))

class Block(nn.Module):
    def __init__(s, C, H, M):
        super().__init__(); s.ln_1 = nn.LayerNorm(C); s.attn = Attn(C, H); s.ln_2 = nn.LayerNorm(C); s.mlp = MLP(C, M)
    def forward(s, x):
        x = x + s.attn(s.ln_1(x)); return x + s.mlp(s.ln_2(x))

class GPT(nn.Module):
    def __init__(s, V, T, C, H, M, L):
        super().__init__()
        s.transformer = nn.ModuleDict(dict(wte=nn.Embedding(V, C), wpe=nn.Embedding(T, C),
                                           h=nn.ModuleList([Block(C, H, M) for _ in range(L)]), ln_f=nn.LayerNorm(C)))
        s.lm_head = nn.Linear(C, V, bias=False)
    def forward(s, idx):
        T = idx.shape[1]
        x = s.transformer.wte(idx) + s.transformer.wpe(torch.arange(T))
        for b in s.transformer.h: x = b(x)
        return s.lm_head(s.transformer.ln_f(x))

def answers(model):
    out = []
    for n, (q, a) in enumerate(QA):
        ids = [stoi[w] for w in q.split()]
        for _ in range(8):
            nxt = model(torch.tensor([ids]))[0, -1].argmax().item()
            if nxt == 1 or len(ids) >= BLOCK: break
            ids.append(nxt)
        out.append(" ".join(words[i] for i in ids[qlen[n]:]))
    return out

def train(C, H, M, L, seed):
    torch.manual_seed(seed)
    model = GPT(len(words), BLOCK, C, H, M, L)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 4000)
    for step in range(4000):
        loss = F.cross_entropy(model(X).view(-1, len(words)), Y.view(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    model.eval()
    with torch.no_grad(): got = answers(model)
    right = sum(g == a for g, (_, a) in zip(got, QA))
    return model, loss.item(), right, got

best = None
for (L, C, M) in [(1, 8, 16), (1, 12, 24), (2, 8, 16), (1, 16, 32), (2, 12, 24), (2, 16, 32)]:
    for seed in (0, 1, 2):
        model, loss, right, got = train(C, 2, M, L, seed)
        n = sum(p.numel() for p in model.parameters())
        print(f"blocks={L} C={C} mlp={M} seed={seed}: loss {loss:.4f}, {right}/{len(QA)} answers right, {n} weights", flush=True)
        if right == len(QA): best = (model, L, C, M, got, n); break
    if best: break
if not best: sys.exit("no size answered everything")

model, L, C, M, got, n = best
def enc(t): return {"shape": list(t.shape), "dtype": "torch.float32", "data": base64.b64encode(t.detach().float().numpy().tobytes()).decode()}
out = {"config": {"n_layer": L, "n_head": 2, "n_embd": C, "n_mlp": M, "vocab_size": len(words), "block_size": BLOCK,
                  "act": "relu", "vocab": words, "qa": [[q, a] for q, a in QA], "n_weights": n}}
for k, v in model.state_dict().items(): out[k] = enc(v)
json.dump(out, open(sys.argv[1], "w"))
print("chosen:", f"blocks={L} C={C} mlp={M}, {n} weights, vocab {len(words)}, block size {BLOCK}")
for (q, a), g in zip(QA, got): print(f"  {q}  ->  {g}")

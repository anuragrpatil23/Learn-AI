"""Causal tracing, from Meng et al. 2022, "Locating and Editing Factual Associations in GPT", section 2.

"The Space Needle is in downtown" -> " Seattle". Three kinds of run:
  clean    the prompt as written; every row (each block's output at each word) is saved
  broken   noise added to the subject's input embeddings, so the answer's chance falls
  patched  the broken run again, with one saved clean row put back; later blocks work out the rest
The score of a row is the share of the lost answer it brings back: (patched - broken) / (clean - broken).
Besides the whole row, the MLP output alone and the attention output alone are put back, over a window
of blocks at once, as the paper does (10 of GPT-2 XL's 48; scaled for smaller models).

Usage: python causal_tracing.py [model, default gpt2-xl] [output.json]
"""
import sys, json, torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

NAME = sys.argv[1] if len(sys.argv) > 1 else "gpt2-xl"
OUT = sys.argv[2] if len(sys.argv) > 2 else "causal_tracing.json"
PROMPT, SUBJECT, ANSWER = "The Space Needle is in downtown", " Space Needle", " Seattle"
NOISE_DRAWS = 10

torch.manual_seed(0)
tok = GPT2TokenizerFast.from_pretrained(NAME)
model = GPT2LMHeadModel.from_pretrained(NAME).eval()
tr = model.transformer
L, C = model.config.n_layer, model.config.n_embd
WIN = max(1, round(L * 10 / 48))

ids = tok(PROMPT, return_tensors="pt").input_ids
words = [tok.decode(t) for t in ids[0]]
T = len(words)
sub = tok(SUBJECT).input_ids
start = next(i for i in range(T) if ids[0, i:i + len(sub)].tolist() == sub)
subject = list(range(start, start + len(sub)))  # GPT-2 cuts " Space Needle" into three pieces
answer = tok(ANSWER).input_ids[0]

def p_answer(logits): return logits[:, -1].softmax(-1)[:, answer]

def part(kind, l):  # where each kind of row comes out of block l
    return tr.h[l] if kind == "state" else getattr(tr.h[l], kind)

# 1. clean run: save every row
clean, hooks = {}, []
for kind in ["state", "mlp", "attn"]:
    for l in range(L):
        def save(m, i, o, kind=kind, l=l):
            clean[kind, l] = (o[0] if isinstance(o, tuple) else o).detach().clone()
        hooks.append(part(kind, l).register_forward_hook(save))
with torch.no_grad(): p_clean = p_answer(model(ids).logits).item()
for h in hooks: h.remove()

# 2. broken run: noise 3x the spread of all word embeddings, on the subject only
noise_sd = 3 * tr.wte.weight.std().item()
noise = torch.randn(NOISE_DRAWS, T, C) * noise_sd
on_subject = torch.zeros(1, T, 1); on_subject[0, subject] = 1
def broken_input(): return tr.wte(ids).expand(NOISE_DRAWS, -1, -1) + noise * on_subject

with torch.no_grad(): p_broken = p_answer(model(inputs_embeds=broken_input()).logits).mean().item()

# 3. patched runs: put back one clean row (or one part over a window of blocks) at one word
def patched(kind, blocks, w):
    hooks = []
    for l in blocks:
        def put_back(m, i, o, l=l):
            x = (o[0] if isinstance(o, tuple) else o).clone()
            x[:, w] = clean[kind, l][0, w]
            return (x,) + tuple(o[1:]) if isinstance(o, tuple) else x
        hooks.append(part(kind, l).register_forward_hook(put_back))
    with torch.no_grad(): p = p_answer(model(inputs_embeds=broken_input()).logits).mean().item()
    for h in hooks: h.remove()
    return p

grids = {}
for kind in ["state", "mlp", "attn"]:
    grids[kind] = []
    for l in range(L):
        blocks = [l] if kind == "state" else [b for b in range(l - WIN // 2, l - WIN // 2 + WIN) if 0 <= b < L]
        grids[kind].append([round(patched(kind, blocks, w), 4) for w in range(T)])
    print(kind, "done")

score = lambda p: (p - p_broken) / (p_clean - p_broken)
print(f"clean {p_clean:.4f}  broken {p_broken:.4f}")
for kind, g in grids.items():
    print(kind)
    for w in range(T):
        print(f"{words[w]:>10}", " ".join(f"{round(100 * score(g[l][w])):3d}" for l in range(L)))

json.dump(dict(model=NAME, layers=L, width=C, prompt=PROMPT, words=words, subject=subject, answer=ANSWER,
               p_clean=round(p_clean, 4), p_broken=round(p_broken, 4), noise_sd=round(noise_sd, 4),
               noise_draws=NOISE_DRAWS, window=WIN, grids=grids), open(OUT, "w"))

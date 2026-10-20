"""Function vectors, from Todd et al. 2023, "Function Vectors in Large Language Models".

Task: opposites, "hot -> cold, big -> small, up -> down, fast ->" -> " slow". Steps 1 to 3 are activation
patching (as in ../activation-patching/causal_tracing.py), used to find the heads; steps 4 and 5 are steering,
building a new row and adding it to the stream.
  1. clean prompts (three examples, then a question word); save each head's output at the last word,
     averaged over the prompts
  2. broken prompts: the same, with the examples' answers shuffled, so the task cannot be worked out
  3. for each head, one at a time: put its averaged clean output into the broken prompts at the last
     word, and record how much the right answer's chance goes up
  4. the function vector: the top heads' averaged outputs, each passed through its own slice of the
     output layer and added together, giving one row as wide as the stream
  5. add that row to the stream at the last word of prompts with no examples at all ("fast ->"),
     after one block, and see whether the model now answers with the opposite

Usage: python function_vectors.py [model, default gpt2-xl] [output.json]
"""
import sys, json, random, torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

NAME = sys.argv[1] if len(sys.argv) > 1 else "gpt2-xl"
OUT = sys.argv[2] if len(sys.argv) > 2 else "function_vectors.json"
EXAMPLES, PROMPTS, TOP = 3, 30, 10

torch.manual_seed(0); random.seed(0)
tok = GPT2TokenizerFast.from_pretrained(NAME)
model = GPT2LMHeadModel.from_pretrained(NAME).eval()
tr = model.transformer
L, H, C = model.config.n_layer, model.config.n_head, model.config.n_embd
HS = C // H

PAIRS = [("hot", "cold"), ("big", "small"), ("up", "down"), ("fast", "slow"), ("good", "bad"), ("happy", "sad"),
         ("high", "low"), ("old", "new"), ("rich", "poor"), ("dark", "light"), ("open", "closed"), ("hard", "soft"),
         ("long", "short"), ("early", "late"), ("full", "empty"), ("wet", "dry"), ("strong", "weak"), ("in", "out"),
         ("left", "right"), ("young", "old"), ("true", "false"), ("near", "far"), ("thick", "thin"), ("win", "lose"),
         ("love", "hate"), ("day", "night"), ("black", "white"), ("buy", "sell"), ("start", "end"), ("heavy", "light"),
         ("loud", "quiet"), ("clean", "dirty"), ("wide", "narrow"), ("cheap", "expensive"), ("first", "last"), ("top", "bottom")]
one = lambda s: len(tok(s).input_ids) == 1
PAIRS = [(a, b) for a, b in PAIRS if one(a) and one(" " + a) and one(" " + b)]  # one piece each, so all prompts are the same length
random.shuffle(PAIRS)
train, test = PAIRS[:24], PAIRS[24:]

def prompt(examples, q): return "".join(f"{a} -> {b}, " for a, b in examples) + f"{q} ->"
def make(shuffle):
    ex = random.sample(train, EXAMPLES + 1); q, a = ex.pop()
    if shuffle:
        outs = [b for _, b in ex]
        while any(o == b for o, (_, b) in zip(outs, ex)): random.shuffle(outs)
        ex = [(x, o) for (x, _), o in zip(ex, outs)]
    return prompt(ex, q), tok(" " + a).input_ids[0]

def batch(ps): return torch.cat([tok(p, return_tensors="pt").input_ids for p, _ in ps]), torch.tensor([a for _, a in ps])

# 1. clean prompts: every head's output at the last word, averaged
clean_ids, clean_ans = batch([make(False) for _ in range(PROMPTS)])
heads_out, hooks = {}, []
for l in range(L):  # the input to the output layer c_proj is the heads' outputs side by side, head h in columns h*HS:(h+1)*HS
    hooks.append(tr.h[l].attn.c_proj.register_forward_pre_hook(lambda m, i, l=l: heads_out.__setitem__(l, i[0][:, -1].mean(0).detach().clone())))
with torch.no_grad(): p_clean = model(clean_ids).logits[:, -1].softmax(-1)[torch.arange(PROMPTS), clean_ans].mean().item()
for h in hooks: h.remove()

# 2. broken prompts: answers shuffled
broken = [make(True) for _ in range(PROMPTS)]
broken_ids, broken_ans = batch(broken)

def p_right(l=None, h=None):
    hook = None
    if l is not None:
        def put_back(m, i):
            x = i[0].clone(); x[:, -1, h * HS:(h + 1) * HS] = heads_out[l][h * HS:(h + 1) * HS]; return (x,)
        hook = tr.h[l].attn.c_proj.register_forward_pre_hook(put_back)
    with torch.no_grad(): p = model(broken_ids).logits[:, -1].softmax(-1)[torch.arange(PROMPTS), broken_ans]
    if hook: hook.remove()
    return p

base = p_right()
p_broken = base.mean().item()

# 3. one head at a time
gain = [[round((p_right(l, h) - base).mean().item(), 4) for h in range(H)] for l in range(L)]
top = sorted(((gain[l][h], l, h) for l in range(L) for h in range(H)), reverse=True)[:TOP]
print(f"clean {p_clean:.4f}  broken {p_broken:.4f}")
for g, l, h in top: print(f"  block {l} head {h}: +{g:.4f}")

# 4. the function vector (from here on, steering: nothing is swapped between runs, a new row is built and added)
W = lambda l: tr.h[l].attn.c_proj.weight  # GPT-2 stores it as [in, out]
fv = sum(heads_out[l][h * HS:(h + 1) * HS] @ W(l)[h * HS:(h + 1) * HS] for _, l, h in top)

# 5. no examples: "fast ->" with and without the vector added after block `at`
def answer(q, at=None):
    hook = None
    if at is not None:
        def add(m, i, o):
            x = o[0].clone(); x[0, -1] += fv; return (x,) + tuple(o[1:])
        hook = tr.h[at].register_forward_hook(add)
    with torch.no_grad(): w = model(tok(prompt([], q), return_tensors="pt").input_ids).logits[0, -1].argmax()
    if hook: hook.remove()
    return tok.decode(w).strip()

right_by_block = [round(sum(answer(q, at) == a for q, a in test) / len(test), 3) for at in range(L)]
at = max(range(L), key=lambda b: right_by_block[b])
right_without = sum(answer(q) == a for q, a in test) / len(test)
zero_shot = [dict(q=q, right=a, without=answer(q), with_fv=answer(q, at)) for q, a in test]
print("right without the vector", right_without, " by block it is added after", right_by_block)
for z in zero_shot: print("  ", z)

json.dump(dict(model=NAME, layers=L, heads=H, width=C, task="opposites", examples=EXAMPLES, prompts=PROMPTS,
               sample_clean=tok.decode(clean_ids[0]), sample_broken=broken[0][0],
               p_clean=round(p_clean, 4), p_broken=round(p_broken, 4), gain=gain,
               top=[dict(block=l, head=h, gain=g) for g, l, h in top], right_without=round(right_without, 3),
               right_by_block=right_by_block, added_after=at, zero_shot=zero_shot), open(OUT, "w"))

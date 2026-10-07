"""Export a small window of real GPT-2 small numbers for the interactive drawing.

Runs GPT-2 small on three short questions, lets it write a few words, and saves, for every step of
the first block and for the final steps, the first 8 columns of each grid of numbers together with
the matching 8x8 corner of each weight matrix. Sizes in the drawing are the real ones (768, 3072,
50257); only the window is drawn in detail. The forward pass is written out by hand so every
intermediate value is available, and is checked against the library's own output.

Usage: python export.py <output.json>
"""
import sys, json, math, torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

WIN, NEW = 8, 8
PROMPTS = ["Q: Why is the sky blue?\nA:", "Q: Why is the sea blue?\nA:", "Q: Why is the grass green?\nA:"]
tok = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
tr = model.transformer
C, H = 768, 12
HS = C // H

def gelu(x): return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))
def ln(x, m):
    mean = x.mean(-1, keepdim=True); sd = ((x - mean) ** 2).mean(-1, keepdim=True).add(m.eps).sqrt()
    return (x - mean) / sd * m.weight + m.bias, mean.squeeze(-1), sd.squeeze(-1)

def block(x, b, keep=False):
    T = x.shape[0]; d = {}
    l1, m1, s1 = ln(x, b.ln_1)
    qkv = l1 @ b.attn.c_attn.weight + b.attn.c_attn.bias
    q, k, v = qkv.split(C, dim=-1)
    heads = lambda z: z.view(T, H, HS).transpose(0, 1)
    sc = heads(q) @ heads(k).transpose(-2, -1) / math.sqrt(HS)
    mask = torch.tril(torch.ones(T, T)) == 0
    att = sc.masked_fill(mask, float("-inf")).softmax(-1)
    ho = (att @ heads(v)).transpose(0, 1).contiguous().view(T, C)
    ao = ho @ b.attn.c_proj.weight + b.attn.c_proj.bias
    mid = x + ao
    l2, m2, s2 = ln(mid, b.ln_2)
    fc = l2 @ b.mlp.c_fc.weight + b.mlp.c_fc.bias
    act = gelu(fc)
    mo = act @ b.mlp.c_proj.weight + b.mlp.c_proj.bias
    out = mid + mo
    if keep:
        d = dict(ln1=l1, ln1_mean=m1, ln1_sd=s1, q=q, k=k, v=v, scores=sc[0], att=att[0], headsOut=ho, attnOut=ao,
                 residMid=mid, ln2=l2, ln2_mean=m2, ln2_sd=s2, fc=fc, act=act, mlpOut=mo, residOut=out)
    return out, d

R = lambda t: [[round(float(v), 4) for v in row] for row in t]
R1 = lambda t: [round(float(v), 4) for v in t]
def word(i):
    s = tok.decode([i])
    return s.replace("\n", "\\n").strip() or repr(s)

runs = []
with torch.no_grad():
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids
        qlen = ids.shape[1]
        full = model.generate(ids, max_new_tokens=NEW, do_sample=False, pad_token_id=50256)[0]
        T = full.shape[0]
        wte, wpe = tr.wte.weight[full], tr.wpe.weight[:T]
        x = wte + wpe; emb = x
        x, d = block(x, tr.h[0], keep=True)
        for b in tr.h[1:]: x, _ = block(x, b)
        lf, mf, sf = ln(x, tr.ln_f)
        logits = lf @ tr.wte.weight.T
        check = (logits - model(full[None]).logits[0]).abs().max().item()
        assert check < 2e-2, check
        probs = logits.softmax(-1)
        vocab = []
        for r in range(T):
            top = probs[r].topk(WIN).indices
            vocab.append(dict(ids=top.tolist(), strs=[word(i) for i in top.tolist()], logits=R(logits[:, top]), probs=R(probs[:, top]),
                              W=R(tr.wte.weight[top][:, :WIN])))
        w = lambda t: R(t[:, :WIN])
        sc = d["scores"].clone(); sc[torch.triu(torch.ones(T, T), 1) == 1] = -1e9
        runs.append(dict(prompt=p.replace("\n", " "), qlen=qlen, tokens=full.tolist(), strs=[word(i) for i in full.tolist()], check=round(check, 5),
            embed=dict(wte=w(wte), wpe=w(wpe), out=w(emb)),
            b0=dict(ln1=dict(out=w(d["ln1"]), mean=R1(d["ln1_mean"]), sd=R1(d["ln1_sd"])), q=w(d["q"]), k=w(d["k"]), v=w(d["v"]),
                    scores=R(sc), att=R(d["att"]), headsOut=w(d["headsOut"]), attnOut=w(d["attnOut"]), residMid=w(d["residMid"]),
                    ln2=dict(out=w(d["ln2"]), mean=R1(d["ln2_mean"]), sd=R1(d["ln2_sd"])), fc=w(d["fc"]), act=w(d["act"]), mlpOut=w(d["mlpOut"]), residOut=w(d["residOut"])),
            fin=dict(residIn=w(x), lnF=dict(out=w(lf), mean=R1(mf), sd=R1(sf)), vocab=vocab)))
        print(f"{p!r}: {qlen} tokens in, wrote {tok.decode(full[qlen:])!r}; hand-written forward pass differs from the library by {check:.1e}")

b = tr.h[0]
W = lambda m, r0=0: R(m.weight[:WIN, r0:r0 + WIN].T)     # Conv1D stores (in, out); row j here = neuron j's slopes
weights = dict(
    ln1=dict(g=R1(b.ln_1.weight[:WIN]), b=R1(b.ln_1.bias[:WIN])),
    q=dict(W=W(b.attn.c_attn, 0), b=R1(b.attn.c_attn.bias[0:WIN])),
    k=dict(W=W(b.attn.c_attn, C), b=R1(b.attn.c_attn.bias[C:C + WIN])),
    v=dict(W=W(b.attn.c_attn, 2 * C), b=R1(b.attn.c_attn.bias[2 * C:2 * C + WIN])),
    attnOut=dict(W=W(b.attn.c_proj), b=R1(b.attn.c_proj.bias[:WIN])),
    ln2=dict(g=R1(b.ln_2.weight[:WIN]), b=R1(b.ln_2.bias[:WIN])),
    fc=dict(W=W(b.mlp.c_fc), b=R1(b.mlp.c_fc.bias[:WIN])),
    mlpOut=dict(W=W(b.mlp.c_proj), b=R1(b.mlp.c_proj.bias[:WIN])),
    lnF=dict(g=R1(tr.ln_f.weight[:WIN]), b=R1(tr.ln_f.bias[:WIN])))
meta = dict(model="GPT-2 small", C=C, H=H, HS=HS, M=4 * C, V=tr.wte.weight.shape[0], nBlocks=len(tr.h), blockSize=tr.wpe.weight.shape[0], win=WIN,
            nWeights=sum(p.numel() for p in model.parameters()))
json.dump(dict(meta=meta, weights=weights, runs=runs), open(sys.argv[1], "w"), separators=(",", ":"))
print("meta:", meta)

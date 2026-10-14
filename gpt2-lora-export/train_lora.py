"""Train a small LoRA on GPT-2 small, for the fine-tuning drawing in the vault.

LoRA pairs (r = 8, alpha = 16) beside the query and value parts of every block's attention input layer;
every original weight frozen. The examples are made-up science questions in the drawing's "Q: ... A:" format,
each with a short answer starting "Because"; only the answer and the end token are scored (prompt at -100).
The three questions the drawing shows are not in the training set.

Usage: python train_lora.py <out.pt>
"""
import sys, math, torch, random
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

R, ALPHA, STEPS, LR = 8, 16, 300, 1e-3
torch.manual_seed(0); random.seed(0)
tok = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
for p in model.parameters(): p.requires_grad_(False)
C = 768

class LoRAqv(torch.nn.Module):
    """out = c_attn(x) + (alpha / r) * [B_q A_q x, 0, B_v A_v x]; B starts at zero, so at step 0 nothing changes."""
    def __init__(self, base):
        super().__init__(); self.base = base; self.s = ALPHA / R
        self.Aq = torch.nn.Parameter(torch.randn(R, C) / math.sqrt(C)); self.Bq = torch.nn.Parameter(torch.zeros(C, R))
        self.Av = torch.nn.Parameter(torch.randn(R, C) / math.sqrt(C)); self.Bv = torch.nn.Parameter(torch.zeros(C, R))
    def forward(self, x):
        out = self.base(x); z = torch.zeros(*x.shape[:-1], C, device=x.device)
        return out + self.s * torch.cat([x @ self.Aq.T @ self.Bq.T, z, x @ self.Av.T @ self.Bv.T], -1)

for b in model.transformer.h: b.attn.c_attn = LoRAqv(b.attn.c_attn)

PAIRS = [("Why is the sun hot?", "Because of nuclear fusion in its core."),
 ("Why is snow white?", "Because its ice crystals scatter all colours of light."),
 ("Why is the night dark?", "Because our side of the Earth faces away from the sun."),
 ("Why is blood red?", "Because of the iron in its haemoglobin."),
 ("Why is the moon bright?", "Because it reflects light from the sun."),
 ("Why is ice slippery?", "Because a thin layer of water forms on its surface."),
 ("Why is the desert dry?", "Because very little rain falls there."),
 ("Why is the ocean salty?", "Because rivers carry dissolved minerals into it."),
 ("Why is fire hot?", "Because burning releases energy as heat."),
 ("Why is gold yellow?", "Because it absorbs blue light."),
 ("Why is coal black?", "Because it absorbs almost all light."),
 ("Why is the rainbow curved?", "Because light bends at the same angle in each raindrop."),
 ("Why is lava red?", "Because hot rock glows."),
 ("Why is the sunset orange?", "Because sunlight travels through more air at dusk."),
 ("Why is milk white?", "Because its fat droplets scatter light."),
 ("Why is the leaf green?", "Because of the chlorophyll in its cells."),
 ("Why is copper green when old?", "Because it reacts with air and water."),
 ("Why is the winter cold?", "Because sunlight hits the ground at a low angle."),
 ("Why is the ground wet?", "Because it rained."),
 ("Why is the cloud grey?", "Because it is thick enough to block sunlight."),
 ("Why is a flame blue?", "Because the gas burns completely."),
 ("Why is the Earth round?", "Because gravity pulls matter evenly towards the centre."),
 ("Why is the river muddy?", "Because it carries soil from upstream."),
 ("Why is the glacier blue?", "Because dense ice absorbs red light."),
]
data = []
for q, a in PAIRS:
    p = tok.encode(f"Q: {q}\nA:"); ans = tok.encode(" " + a) + [tok.eos_token_id]
    data.append((p + ans, [-100] * len(p) + ans))
params = [p for p in model.parameters() if p.requires_grad]
print("moving weights:", sum(p.numel() for p in params), "of", sum(p.numel() for p in model.parameters()))
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
model.train()
for step in range(STEPS):
    batch = random.sample(data, 8); L = max(len(x) for x, _ in batch)
    ids = torch.tensor([x + [tok.eos_token_id] * (L - len(x)) for x, _ in batch])
    lab = torch.tensor([y + [-100] * (L - len(y)) for _, y in batch])
    loss = model(ids, labels=lab).loss
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 50 == 0 or step == STEPS - 1: print(step, round(loss.item(), 3))
model.eval()
lo = {f"{i}.{n}": getattr(b.attn.c_attn, n).detach().clone() for i, b in enumerate(model.transformer.h) for n in ("Aq", "Bq", "Av", "Bv")}
torch.save(dict(r=R, alpha=ALPHA, lora=lo, steps=STEPS, lr=LR, pairs=PAIRS), sys.argv[1])
for q in ["Why is the sky blue?", "Why is the sea blue?", "Why is the grass green?"]:
    ids = tok(f"Q: {q}\nA:", return_tensors="pt").input_ids
    out = model.generate(ids, max_new_tokens=16, do_sample=False, pad_token_id=50256)
    print(repr(q), "->", repr(tok.decode(out[0, ids.shape[1]:])))

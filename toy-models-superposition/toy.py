"""The experiment at the centre of Toy Models of Superposition (Anthropic, 2022), at its smallest.

There are n things. In each example a thing is present with chance p, and when present has a size
between 0 and 1; otherwise it is 0. The network squeezes the n numbers into m neurons, fewer than n,
and must give the n numbers back:

  h     = W x                 n things  -> m neurons
  x_hat = ReLU(W^T h + b)     m neurons -> n things again
  loss  = sum over things of importance * (x - x_hat)^2

The same table W is used on the way in and, turned over, on the way out. Column i of W is where
thing i is put among the m neurons: an arrow with m parts.

Each run is recorded with tracker.py (a copy of the run tracker's writer) so it can be watched in the
viewer: the loss, how many things were kept, each thing's arrow length, and the arrows themselves.

Usage: python toy.py --things 10 --neurons 5 --present 0.9
       python toy.py --things 5 --neurons 2 --present 0.1 --importance 0.7 --out ~/run-tracker-data/runs/toy-superposition/n5_m2_p0.1
"""
import argparse, os, torch
import tracker

p = argparse.ArgumentParser()
p.add_argument("--things", type=int, default=10)
p.add_argument("--neurons", type=int, default=5)
p.add_argument("--present", type=float, default=0.9, help="chance that each thing is present in an example")
p.add_argument("--importance", type=float, default=1.0, help="thing i counts importance**i in the loss; 1 means all equal")
p.add_argument("--steps", type=int, default=20000)
p.add_argument("--batch", type=int, default=1024)
p.add_argument("--no-relu", action="store_true")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default=None, help="folder to record the run in; nothing is recorded if left out")
p.add_argument("--log-every", type=int, default=100)
args = p.parse_args()
torch.manual_seed(args.seed)
n, m = args.things, args.neurons

W = torch.nn.Parameter(torch.randn(m, n) * 0.3)
b = torch.nn.Parameter(torch.zeros(n))
weight = args.importance ** torch.arange(n)
opt = torch.optim.Adam([W, b], lr=1e-3)

def examples(count):
    return torch.rand(count, n) * (torch.rand(count, n) < args.present)

def forward(x):
    out = x @ W.T @ W + b
    return out if args.no_relu else torch.relu(out)

ABOUT = {
    "loss": "What is minimised: how far each thing that comes back is from what went in, squared, times its importance, added up.",
    "things_kept": "How many things have an arrow longer than 0.5. A thing with a short arrow has been dropped.",
    "in_superposition": "How many kept things share their direction with another thing (an overlap beyond 0.3 either way).",
    "arrow_length": "The length of each thing's arrow among the neurons, thing 1 first.",
    "arrows": "The table W: one row per neuron, one column per thing. Column i is where thing i is put.",
    "overlaps": "The dot product of every pair of arrows (W turned over, times W). The diagonal is each arrow's length squared.",
}
run = None
if args.out:
    out = os.path.expanduser(args.out)
    run = tracker.start(out, vars(args), total=args.steps, project=os.path.basename(os.path.dirname(os.path.normpath(out))),
                        group="n%d_m%d" % (n, m))
    run.describe(ABOUT)

def state():
    """What the network looks like now, as plain numbers."""
    with torch.no_grad():
        length = W.norm(dim=0); overlap = W.T @ W
        off = (overlap - torch.diag(overlap.diagonal())).abs().max(0).values
        r = lambda t: [round(float(v), 3) for v in t]
        return {"things_kept": int((length > 0.5).sum()), "in_superposition": int(((length > 0.5) & (off > 0.3)).sum()),
                "arrow_length": r(length), "arrows": [r(row) for row in W], "overlaps": [r(row) for row in overlap], "bias": r(b)}

for step in range(args.steps + 1):
    x = examples(args.batch)
    loss = (weight * (x - forward(x)) ** 2).sum(-1).mean()
    if run and (step % args.log_every == 0 or step == args.steps):
        run.log(step, loss=round(loss.item(), 5), **state())
    if step == args.steps:
        break
    opt.zero_grad(); loss.backward(); opt.step()
if run:
    run.finish()

with torch.no_grad():
    length = W.norm(dim=0)                                  # how long each thing's arrow is; near 0 means it was dropped
    alone = forward(torch.eye(n)).diagonal()                # each thing put in alone at full size: what comes back for it
    overlap = (W.T @ W)                                     # dot product of every pair of arrows
    others = (overlap - torch.diag(overlap.diagonal())).abs().max(0).values
    seen = args.steps * args.batch * args.present
    print(f"{n} things, {m} neurons, each present {args.present:.0%} of the time; each thing seen about {seen:,.0f} times")
    print(f"things present per example, on average: {n * args.present:.1f}")
    print("thing   arrow length   comes back as (put in alone at 1.0)   largest overlap with another thing")
    for i in range(n):
        print(f"{i + 1:>4}    {length[i]:>8.2f}      {alone[i]:>8.2f}                          {others[i]:>8.2f}")
    print(f"things kept (arrow longer than 0.5): {int((length > 0.5).sum())} of {n}")
    print(f"final loss: {loss.item():.4f}")

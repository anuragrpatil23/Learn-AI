"""Give every unit of a step a fixed place on a flat map, so that units with alike patterns sit near each other.

A network's units have no positions of their own; a unit's number is only the order it was made in.
This works some out. Each unit is described by a list of numbers (for a feature, its pattern across
the MLP neurons). Units whose lists point the same way are neighbours, and the map is arranged so
that neighbours end up close together on the page.

The method is the idea behind UMAP, written out with PyTorch so that nothing new needs installing:
find each unit's nearest neighbours, then move the dots so that neighbours pull together and
randomly chosen pairs push apart. Two dimensions cannot hold everything about thousands of
dimensions, so the map says how well it kept the neighbours: see `kept` below.
"""
import torch

def neighbours(rows, k=15, chunk=2048):
    """For each row, the k rows that point most nearly the same way. rows: (n, d). Returns (n, k) indices and likeness."""
    x = rows / rows.norm(dim=1, keepdim=True).clamp_min(1e-8)
    idx, sim = [], []
    for start in range(0, x.shape[0], chunk):
        s = x[start:start + chunk] @ x.T
        s[torch.arange(s.shape[0]), torch.arange(start, start + s.shape[0])] = -2       # not its own neighbour
        best = s.topk(k, dim=1); idx.append(best.indices); sim.append(best.values)
    return torch.cat(idx), torch.cat(sim)

def place(rows, k=15, epochs=300, seed=0):
    """A place for every row: (n, 2), roughly within -1 to 1. Also returns the neighbours it used."""
    g = torch.Generator().manual_seed(seed)
    n = rows.shape[0]
    idx, sim = neighbours(rows, k)
    # start from the two directions along which the rows differ most, so the broad arrangement is not left to chance
    centred = rows - rows.mean(0)
    _, _, v = torch.pca_lowrank(centred, q=2, niter=4)
    y = centred @ v; y = (y / y.abs().max() * 10).clone().requires_grad_()
    a, b = 1.58, 0.90                                           # how tightly neighbours sit; UMAP's values for a small gap
    src = torch.arange(n).repeat_interleave(k); dst = idx.flatten()
    opt = torch.optim.Adam([y], lr=1.0)
    for epoch in range(epochs):
        for group in opt.param_groups:
            group["lr"] = 1.0 * (1 - epoch / epochs) + 0.01
        d_pull = ((y[src] - y[dst]) ** 2).sum(1)
        far = torch.randint(0, n, (src.shape[0] * 3,), generator=g)                    # three random others for each neighbour
        d_push = ((y[src.repeat(3)] - y[far]) ** 2).sum(1)
        loss = torch.log1p(a * d_pull.clamp_min(1e-9) ** b).sum() + torch.log1p(1 / (a * d_push.clamp_min(1e-4) ** b)).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    y = y.detach(); y = y - y.mean(0)
    return y / y.abs().quantile(0.995), idx

def kept(y, idx, near=10, within=50, sample=2000, seed=0):
    """Of each unit's `near` nearest neighbours by pattern, the share found among its `within` nearest on the map. 1 is perfect."""
    g = torch.Generator().manual_seed(seed)
    pick = torch.randperm(y.shape[0], generator=g)[:sample]
    d = torch.cdist(y[pick], y); d[torch.arange(len(pick)), pick] = float("inf")
    on_map = d.topk(within, dim=1, largest=False).indices
    return float((idx[pick, :near, None] == on_map[:, None, :]).any(2).float().mean())

def regions(y, count=36, rounds=25, seed=0):
    """Split the map into patches of nearby dots. Returns each dot's patch, and each patch's centre."""
    g = torch.Generator().manual_seed(seed)
    centres = y[torch.randperm(y.shape[0], generator=g)[:count]].clone()
    for _ in range(rounds):
        which = torch.cdist(y, centres).argmin(1)
        for c in range(count):
            if (which == c).any():
                centres[c] = y[which == c].mean(0)
    return which, centres

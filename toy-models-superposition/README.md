# Toy Models of Superposition, at its smallest

The central experiment of Toy Models of Superposition (Anthropic, 2022): n things squeezed into m neurons, fewer than n, and asked for back.

```
h     = W x                 n things  -> m neurons
x_hat = ReLU(W^T h + b)     m neurons -> n things again
loss  = sum over things of importance * (x - x_hat)^2
```

- `toy.py` is the whole experiment: the data, the network, the loss, and what is recorded.
- `run_paper.sh` runs the paper's two demonstrations: 5 things into 2 neurons, and 20 things into 5, from always present to rarely present, and once without the ReLU.
- `tracker.py` is a copy of the run tracker's writer (github.com/anuragrpatil23/run-tracker), so each run can be watched there.

What it shows: when things are always present the network keeps as many as it has neurons, the most important ones. As they get rarer it packs in more, by letting things share directions. Without the ReLU it never packs in more than it has neurons.

```
python toy.py --things 10 --neurons 5 --present 0.05
```

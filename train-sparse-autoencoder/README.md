# train-sparse-autoencoder

Trains a sparse autoencoder on the 3,072 MLP neurons of GPT-2 small's first block, following the
method of Towards Monosemanticity (Anthropic, 2023), and logs how it forms.

- `train.py` is the whole experiment: making rows from text, the network, the loss, and the log.
- `make_tokens.py` turns the fineweb-edu text into token files for `train.py`.
- `submit_minerva.sh` submits one run to Minerva's GPU queue.

Each run writes to `runs/<name>/`: the record of the run, and snapshots of the weights at the start,
at 1%, at 10% and at the end.

The record is written with the Weights & Biases client if it is installed (`pip install wandb`),
always in offline mode: nothing is sent anywhere, and no account is needed. Without the client,
`tracker.py` writes the same things as plain files; `--logger files` or `--logger wandb` chooses one
outright. Either way the runs are copied to the laptop and looked at with
[run-tracker](https://github.com/anuragrpatil23/run-tracker) (`rt sync`, `rt view`), filed under the
project given by `--project`.

What is logged, by the usual names: `train/loss` and its two terms, the same on a held-out batch
as `val/…`, `lr`, `grad_norm`, `rows_per_second`, the state of the dictionary as `features/…`, and
what the strongest feature for " sky" and " the" responds to. With the W&B client a run still going
reaches disk a block at a time, so it can be some minutes behind; with the plain files each line is
on disk as it is logged.

Known limits. Features that stop firing are not restarted, so their number can be watched as it
grows. The encoder starts as the decoder turned over, which later work found helps and the 2023
paper does not do. On Apple GPUs with PyTorch 2.1 training blows up after about a hundred steps;
the same run is steady on CPU, so use `--device cpu` on a Mac or run on CUDA.

## Scanner

`scan_server.py` serves a trained network to the Train Run Tracker's Scan view: type text, see what
each step of GPT-2's first block and the sparse autoencoder produced, open a unit, contrast two texts.

    python scan_server.py --runs ~/run-tracker-data/runs --port 8790

It is written on `scankit.py`, a copy of the scanner kit from the run-tracker repo, which carries
everything that is the same for any network; what is in `scan_server.py` is GPT-2's own. After the
kit changes, copy it across again. It finds `step_*.pt` snapshots at any depth under `--runs`; fetch one first with `rt fetch`.
`--openai <file>` adds the sparse autoencoder OpenAI published for the same neurons. `scan_fixtures/`
holds recorded replies for every route, written by `--write-fixtures`, for building against.


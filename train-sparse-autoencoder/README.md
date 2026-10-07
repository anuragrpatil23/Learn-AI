# train-sparse-autoencoder

Trains a sparse autoencoder on the 3,072 MLP neurons of GPT-2 small's first block, following the
method of Towards Monosemanticity (Anthropic, 2023), and logs how it forms.

- `train.py` is the whole experiment: making rows from text, the network, the loss, and the log.
- `make_tokens.py` turns the fineweb-edu text into token files for `train.py`.
- `submit_minerva.sh` submits one run to Minerva's GPU queue.

Each run writes to `runs/<name>/`: `log.jsonl` (one line per log step), `config.json`, and
snapshots of the weights at the start, at 1%, at 10% and at the end.

Known limits. Features that stop firing are not restarted, so their number can be watched as it
grows. The encoder starts as the decoder turned over, which later work found helps and the 2023
paper does not do. On Apple GPUs with PyTorch 2.1 training blows up after about a hundred steps;
the same run is steady on CPU, so use `--device cpu` on a Mac or run on CUDA.

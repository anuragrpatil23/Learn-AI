# Function vectors

A rerun of Todd et al. 2023, "Function Vectors in Large Language Models", one of the two methods papers William Rudman sent after the first meeting (2026-10-14).

It is two techniques in a row, and the script keeps them apart:

1. **Finding the heads: activation patching.** Opposites prompts ("hot -> cold, big -> small, up -> down, fast ->") are broken by shuffling the examples' answers. Each attention head's output at the last word, averaged over clean prompts, is put into the broken prompts one head at a time, and the gain in the right answer's chance is recorded. This is the same tool as `../activation-patching/causal_tracing.py`, pointed at single heads.
2. **Using what was found: steering (activation addition).** The top heads' outputs are turned into rows as wide as the stream and added up into one row, the function vector. It is added to the stream of prompts with no examples ("fast ->"), to see whether the model now gives the opposite. Nothing is swapped from one run into another here; a new row is built and added.

It writes a JSON file that the drawing in the vault reads (`sfresearch/AI Synthesis/AI Exploration/Concepts/Interpretability/function-vectors.html`).

## Running

```
python function_vectors.py gpt2-xl function_vectors.json
```

Defaults to `gpt2-xl` (6 GB download); `gpt2` runs in about a minute. Environment: the `build_sparse_autoencoder_venv_310` pyenv environment (torch 2.1, transformers 4.43).

## What it shows (2026-10-20)

On GPT-2 XL: block 20 head 5 carries the most (+13 points of chance). With the top 10 heads' vector added after blocks 4 to 17, 3 of 8 held-out words get their opposite with no examples, none without it. On GPT-2 small it barely works.

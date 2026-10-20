# Activation patching

A rerun of causal tracing, the activation patching method of the ROME paper, one of the two methods papers William Rudman sent after the first meeting (2026-10-14). The other, Function Vectors, uses this tool for one step and lives in `../function-vectors/`.

| Script | Paper | What it does |
|---|---|---|
| `causal_tracing.py` | Meng et al. 2022, "Locating and Editing Factual Associations in GPT", section 2 | Breaks the subject of "The Space Needle is in downtown" with noise, then puts back one clean row at a time and measures how much of " Seattle" returns. Also the MLP alone and attention alone. |

It prints its result and writes a JSON file, which the drawing in the vault reads (`sfresearch/AI Synthesis/AI Exploration/Concepts/Interpretability/activation-patching.html`).

## Running

```
python causal_tracing.py gpt2-xl causal_tracing.json
```

The model defaults to `gpt2-xl` (48 blocks, the one the ROME paper used, a 6 GB download). `gpt2` runs in about a minute and is useful for trying changes. Environment: the `build_sparse_autoencoder_venv_310` pyenv environment (torch 2.1, transformers 4.43).

## What GPT-2 small shows

It runs on GPT-2 small, but weakly. Its clean chance of " Seattle" is only 8 percent, and the early site shows up at " Space" rather than at the last subject piece "le". This is why the drawing uses GPT-2 XL.

## What GPT-2 XL shows (2026-10-20)

Causal tracing reproduces the paper's Figure 1: the clean chance of " Seattle" is 97.6 percent and the broken one 2.3 percent. Putting back the row at "le" restores up to 99 percent around blocks 12 to 30; the MLP alone does so at "le" around blocks 12 to 20; attention alone at "downtown" around blocks 26 to 40. Putting back one subject piece in the first block restores almost nothing, since the other pieces stay noisy.


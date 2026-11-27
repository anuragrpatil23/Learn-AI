# model-diagrams

`modeldiagram` writes a model in the vault's model-diagram format: its structure (`<model>.model.yaml`) and the real numbers for each input (`values/<input>.json`). The Textbooks tab in Thinking Space draws them in 3D. The format itself is described in the vault, at `kai-workspace/universal/model-diagrams.md`. The models it has written live at `lifeblood_systems/sfresearch/AI Synthesis/AI Exploration/Concepts/model-diagrams/`.

## Setup

```
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e . pytest
```

It needs transformers 5.20 or newer, since Qwen3.5 was added there. Models and tokenizers come from the Hugging Face cache (`HF_HOME`).

## Commands

```
# structure: one pass on the meta device, so no weights and no GPU
modeldiagram spec gpt2 --out <vault>/model-diagrams/llm/gpt-2 --batch 3 --tokens 11
modeldiagram spec Qwen/Qwen3.5-9B --out <vault>/model-diagrams/vlm/qwen3.5-9b --grid 1,60,48 --tokens 825 --image-at 61

# values for a language model, run on the CPU: the raw dump goes to runs/<run id>/ (git-ignored), the trim to values/
modeldiagram values gpt2 --input <model folder>/inputs/sky.txt --out <model folder>

# values for a vision-language model: the labbook job on Minerva (experiments/01-vlm-values/README.md), then the trim
modeldiagram trim-vlm-dump runs/<run>/dump.pt --out <model folder>

# values for Qwen3.5-9B, trimmed from the vlm-in-one-drawing dumps (Minerva jobs 283648526 and 283648536)
modeldiagram trim-qwen-dump --acts <.html-parts>/vlm-in-one-drawing/vlm-activations.json \
    --weights <.html-parts>/vlm-in-one-drawing/weights.json --acts-job 283648526 --weights-job 283648536 --out <model folder>

# does a folder connect: every op reads something made before it, shapes agree, steps name real ids, values fit
modeldiagram check <model folder> ...
```

## How it works

- `kinds.py` says what each module class is: one of the format's op kinds, a container, a repeated block, a part read by its op, or something to skip. A class missing from it stops the run and names the class, so nothing is guessed.
- `trace.py` runs one pass with a hook on every module, which gives a tree of calls with their input and output tensors. It also wraps the attention function (for the queries and keys after rotation, and the weights) and the running-memory function (for the state at the end). On the meta device a small dispatch mode moves the model's CPU helper tensors to meta.
- `build.py` walks the tree. Each kind has a reader that names the axes of what it makes, and an axis given two sizes stops the run. A run of blocks is grouped by layout, so Qwen's language model becomes a running-memory block (24) and an attention block (8, every fourth). Adds, and joins where rows are re-cut, are worked out from the shapes. Names follow what the engine reads: `<op>.out`, `<op>.<part>`, `<op>[.<part>].weight` stored (out, in).
- `texts.py` writes each op's plain name and each step's sentence from its kind and the sizes.
- `values.py` reruns the same trace on the real model. Because the same code names the tensors, the names match the spec. It keeps the columns the engine draws (8 of 768, 16 of 3,072) at 3 significant digits.

## Adding a model

Run `spec` on it. If it stops on a class, add the class to `kinds.py`. If the class is a new kind of attention, memory or MLP, also give it a reader in `build.py`. Then run `check`, and write the model's page beside its spec.

A model folder holds `weights.json` once (the weights are the same for every input) and `values/<input>.json` per input, with activations only.

## Vision-language models on Minerva

A VLM's values need a GPU, so they come from a labbook job. This folder is its own labbook project: `project.env`, `model.env.example` (copy to `.model.env`), `requirements.txt` for the job's venv, and `experiments/01-vlm-values/`, whose README has the submit line. The job is `modeldiagram.vlm_job`. It runs the same trace as `spec` on the real model, keeping tensors only for the blocks the spec draws, and writes a generous `dump.pt` (64 columns, all heads, the scores before the softmax). It then trims that into a model folder inside the run folder. `trim-vlm-dump` does the same trim later, on the laptop. A test runs the whole path on a tiny Qwen3.5 with random weights.

labbook refuses to submit from a checkout with uncommitted changes anywhere in Learn-AI, since it checks with git status.

## Tests

`pytest tests` runs in a few seconds. One Qwen test builds the 9B model on the meta device. Another runs the Minerva job's trace, dump and trim on a tiny random Qwen3.5. `tests/compare_reference.py` checks GPT-2's values against the hand-built gpt-with-sparse-network page's data.

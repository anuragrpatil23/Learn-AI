# 01 vlm values: Qwen3.5-9B's numbers for the model drawing

## Question

What numbers does the drawing of Qwen3.5-9B need from a real pass, and can one job make all of them, with the same names as its spec? The dumps of 2026-11-26 (jobs 283648526 and 283648536) were made for one hand-built page. They have attention for head 1 only and no scores before the softmax. They also have no weights for the vision keys and values, so the drawing's attention steps are partly empty.

## Prediction

One pass per picture, traced by the generator's own code, gives every tensor of the spec. That includes all 16 heads in vision block 1 and in language block 4, their scores, the running memory's state, and every weight's corner. The replies match the run records (dots: D with the hint, A without; squares: D and D). The trimmed values stay under 2 MB per picture.

## Setup

```
counting.json --> for each picture: processor --> generate (reply, no hooks)
                                             \-> one traced pass (trace.run_vlm: tensors kept only for the drawn blocks)
                                                   --> build.build_vlm (spec names) --> dump_one --> runs/<run>/dump.pt
dump.pt --> vlm_job.trim --> runs/<run>/model-folder/values/<picture>.json and weights.json
```

- Model: the shared copy of Qwen3.5-9B, untrained (mdl-ce4a4b, revision c2022362), bfloat16, eager attention, one 80 GB GPU.
- Inputs (`inputs/counting.json`): the four pictures of the model drawing, dots and squares, each with the hint box drawn and empty, with the run's recorded prompt (`prompt-dots.txt`, and `prompt-squares.txt` with the squares question). The pictures here are the copies in vlm-seeing-a-picture's data. Before trusting a number-by-number comparison with the old dumps, check their pixel hashes against the exp-cd960d run's `inputs.json`.
- Rows: `ROWS=hint-pairs` chooses the same rows as dump_activations.py. In the vision part, 4 letter and 4 object tokens, 4 patches each. In the language part, 8 words before the picture, 8 letter and 8 object tokens, and the last 8 words.
- The dump keeps 64 columns, all heads, and the scores before the softmax. The values keep the columns the drawing shows.

## What changed since the last experiment

This is the first in this folder. Against the old dump scripts, everything comes from the generator's trace (so the names are the spec's), all heads and the scores are kept, and the weights come from the same pass, keys and values included.

## Command

Not submitted: the run hold (2026-11-26). When the hold is lifted, run from `Learn-AI/model-diagrams` on the login node, with the venv from `jobs/requirements.txt` active and `.model.env` copied from `model.env.example`:

```
KIND=full CONDITION="model=untrained" LABBOOK_WHY="the model drawing's Qwen3.5-9B values: all heads, scores, every weight" \
LABBOOK_SHOWN="the four counting pictures (dots and squares, hint drawn and empty) with the recorded letter-only prompt" \
LABBOOK_REPLY="the answer letter after Answer:" \
    labbook submit 01-vlm-values modeldiagram.vlm_job INPUTS=experiments/01-vlm-values/inputs/counting.json NEW_TOKENS=12 -W 60
```

Try `--dry-run` first. After the run syncs back, run `modeldiagram trim-vlm-dump runs/<run>/dump.pt --out <vault>/model-diagrams/vlm/qwen3.5-9b`, then `modeldiagram check`. This replaces the files from the old dumps.

## Runs

None yet.

## Analysis

To be written after the run.

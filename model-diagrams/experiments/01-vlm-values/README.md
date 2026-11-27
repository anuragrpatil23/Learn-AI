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

Anurag lifted the hold for this one run (2026-11-26). It goes into the cluster queue with `labbook request` on datapilot, never straight to LSF with `labbook submit`, `jobs/submit.sh` or `bsub` (the standing rule in sfresearch-workflow-conventions.md). The queue project is `model-diagrams`. The checkout is in scratch (`/sc/arion/scratch/patila06/Learn-AI`; nothing new goes directly in `/sc/arion/work/patila06` or above) and must have no local changes. The venv is in scratch too (`ENV` in `project.env`), built from `jobs/requirements.txt` and rebuilt from it if scratch purges it and `.model.env` copied from `model.env.example`:

```
KIND=full CONDITION="model=untrained" LABBOOK_WHY="the model drawing's Qwen3.5-9B values: all heads, scores, every weight" \
LABBOOK_SHOWN="the four counting pictures (dots and squares, hint drawn and empty) with the recorded letter-only prompt" \
LABBOOK_REPLY="the answer letter after Answer:" \
/data/MountSinaiGit/cc-minerva/labbook-venv/bin/labbook request --project model-diagrams --hours 1 \
    01-vlm-values modeldiagram.vlm_job INPUTS=experiments/01-vlm-values/inputs/counting.json NEW_TOKENS=12 \
    -q gpuexpress -W 60
```

The four pictures match exp-cd960d (2026-11-26_0558-untrained-letter) inputs.json by pixel digest: the dots are shape-0052 and the squares shape-0001. After the run syncs back, run `modeldiagram trim-vlm-dump runs/<run>/dump.pt --out <vault>/model-diagrams/vlm/qwen3.5-9b`, then `modeldiagram check`. This replaces the files from the old dumps.

## Runs

- run-112c39 (full, req-de0159, LSF 283716783, code f3d5d79): done. Replies D, A, D, D, as predicted. `runs/2026-11-27_0234-untrained/`, dump.pt sha256 fd13aee95fc773b3914040601ee82161d0bb61cfc71e2a91ff1a80994fdcb20b. Trimmed into the vault (commit 2f3ee5eba).
- run-587d9d (trial, req-f968b3, LSF 283716716, code ad58e0d): done, byte-identical dump.
- run-8b45ff (trial, req-42e7b6, LSF 283716534): failed, a CPU mask on GPU scores; fixed in 06a797a.
- run-916a26 (full, req-3a9c27, LSF 283707672): failed at start, no torchvision; fixed in 9362dd0.

## Analysis

The prediction held. One pass per picture gave every tensor of the spec with its names: all 16 heads in vision block 1 and language block 4 with their scores before the softmax, the vision and language queries and keys, the gate, the memory's keys, write, fade and state, the place rows, and the weights the old dumps lacked (vision keys and values, gate, memory keys and values, token table). The replies match the run records, and the values are 1.23 MB per picture. `modeldiagram check` passes on the vault folder. What was run and learnt is in `notes.md`.

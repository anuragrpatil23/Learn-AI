# 01 vlm values: notes

What was decided, run and learnt, with commits, run ids and hashes. Both workers of a pair read this first and add to it as they go.

## 2026-11-27, pair w-qwen-values / w-qwen-values-minerva

Brief: `kai-workspace/sfresearch/briefs/2026-11-27-pair-qwen-values.md` (from interpretability-6c).

Goal: the full run finishes in the queue, is trimmed into the vault's `model-diagrams/vlm/qwen3.5-9b/` (values and weights.json), and `modeldiagram check` passes.

### Before this pair

- run-916a26 (req-3a9c27, LSF 283707672, code da45b1c) failed at start on lg05e14: `Qwen3VLVideoProcessor requires the Torchvision library`. Fixed in 9362dd0 (torchvision 0.29.1+cu130 in `jobs/requirements.txt`). Its run folder is on the laptop, untracked, under `runs/2026-11-27_0018-untrained/`.
- The earlier scratch checkout and venv were removed; both are rebuilt by this pair.
- Lesson from last night: with `python/3.12.5` loaded, the module's `PYTHONPATH` leaks in. Unset it when installing and checking, or pip skips packages and the wrong torch loads.

### Plan

1. Fresh clone at `/sc/arion/scratch/patila06/Learn-AI` (origin/main 9362dd0), venv at `/sc/arion/scratch/patila06/venvs/model-diagrams-venv`, `.model.env` from the example.
2. Trial: `KIND=trial`, same four inputs (the job is four passes, a few minutes), `-W 30`.
3. Full run with the README's command. Commit the run's files on Minerva and push; `dump.pt` stays out of git, listed in run.json `big_files` with its sha256.
4. Copy `dump.pt` to the laptop over ssh, check its sha256, `modeldiagram trim-vlm-dump`, `modeldiagram check`, commit in the vault.

### Log

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

- 2026-11-27 02:20 (minerva). Fresh clone at `/sc/arion/scratch/patila06/Learn-AI`, origin/main cdc229e. Over ssh the GitHub key on the login node is refused; the clone and push go over https (stored credentials). Hook `run-size-check.sh` installed as `.git/hooks/pre-commit` (sha256 2ffaff71...5630, same as cc-minerva bin). Venv `/sc/arion/scratch/patila06/venvs/model-diagrams-venv` built with `python/3.12.5`, PYTHONPATH unset; checked from the venv: torch 2.14.1+cu130 (cuda 13.0), torchvision 0.29.1+cu130, transformers 5.24.1. `.model.env` copied from the example: MODEL_ID mdl-ce4a4b, the shared copy already in `/sc/arion/scratch/patila06/models/`, so no download.
- 2026-11-27 02:21 (minerva). Trial req-42e7b6, run-8b45ff, LSF 283716534, code cdc229e, lg07c09 (A100 80 GB): FAILED after 177 s. The model loaded, then `build.read_attention` (build.py:421) raised `expected self and mask to be on the same device, but got mask on cpu and self on cuda:0`: the causal mask was made on the CPU. The tiny-model test runs on the CPU, so it never saw this. Fix: the mask is made on `scores.device` (next commit). Run folder `runs/2026-11-27_0222-untrained/` committed as a record.
- 2026-11-27 02:28 (minerva). Trial 2 req-f968b3, run of `runs/2026-11-27_0228-untrained/`, LSF 283716716, code ad58e0d: DONE in 1.2 min of job time. Replies: dots-hint D, dots-empty A, squares-hint D, squares-empty D, as predicted. 825 tokens, picture from 61, 124 tensors per picture. Values 1.23 MB per picture (under 2 MB), weights.json 80 KB. dump.pt 5,210,069 bytes, sha256 fd13aee9...b20b (matches run.json results), kept out of git and listed under big_files. The slow-kernel warnings (no causal_conv1d / flash-linear-attention) are harmless here.
- 2026-11-27 (laptop). Trial run-587d9d's dump.pt copied over ssh, sha256 fd13aee9...cb20b matches. Trimmed into a scratch folder with the vault's spec: `check` passes (124 tensors, 4 values at about 1.23 MB, weights 80 KB). Against the old vault values, new are: vision q and k, scores before softmax and all 16 heads in both vision block 1 and language block 4, the gate, the memory keys, write and fade, and the place rows (vis_position.rows). New weights: vision k and v, gate, memory k and v, embed. No slice is flat or NaN. Replies match the prediction (D, A, D, D).
- 2026-11-27 (laptop). Decision: dump.pt is 5.2 MB, not over 50 MB as the brief assumed, so by the conventions it is committed with the run. The `dump.pt` line in .gitignore (cdc229e) is removed; the 50 MB pre-commit hook already stops anything too big.
- 2026-11-27 02:34 (minerva). Trial 2 above is run-587d9d. Full run req-de0159, run-112c39, LSF 283716783, code f3d5d79 (clean), lg07c09: DONE, 1.2 min of job time. Replies D, A, D, D, the same as the trial. dump.pt 5,210,069 bytes, sha256 fd13aee95fc773b3914040601ee82161d0bb61cfc71e2a91ff1a80994fdcb20b, byte-identical to the trial dump, so the job is deterministic. From 543cf87 on, a dump under 50 MB goes into git with its run, so this runs dump.pt is committed and has no big_files entry. The trials dump.pt stays on Minerva uncommitted, still listed in its run.json big_files.

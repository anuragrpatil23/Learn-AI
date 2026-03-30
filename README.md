# Learn AI

This repository is a working collection of machine learning study projects, experiments, and research code. It spans transformer training, sparse autoencoders, reinforcement learning scaffolding, optimization notebooks, and CNN fundamentals.

## Repository Layout

### `build-nanogpt/`

A from-scratch NanoGPT build focused on transformer training, GPT-2 weight loading, distributed training, and performance improvements.

### `NanoGPT/`

Additional NanoGPT lecture and notebook material, including exploratory development notebooks.

### `build-sparse-autoencoder/`

Sparse autoencoder experiments and training code, with notebooks and package scaffolding for interpretability work.

### `autoresearch/`

Autonomous LLM pretraining experiments based on `karpathy/autoresearch`, designed for rapid research loops on HPC hardware.

### `deep_gradients/`

CNN-focused learning code for understanding convolutional networks, residual architectures, and training behavior on small datasets.

### `Large-Scale-Optimization/`

Notebook-heavy optimization work covering logistic regression, second-order methods, SVRG, and subgradient-based methods.

### `learn-rl/`

An early reinforcement learning package scaffold reserved for future RL experiments.

## Environment Setup

There is no single root environment for every subproject. Most directories are intended to be used independently.

- Poetry-based projects: `deep_gradients/`, `build-sparse-autoencoder/`, and `learn-rl/`
- `uv`-based project: `autoresearch/`
- Notebook-first projects: `Large-Scale-Optimization/` and parts of `NanoGPT/`

Typical setup inside a project directory:

```bash
cd <project-directory>
poetry install
```

For `autoresearch/`:

```bash
cd autoresearch
uv sync
```

## What To Read First

- Start with `build-nanogpt/README.md` for the most complete transformer project overview.
- Open `autoresearch/README.md` for the autonomous pretraining workflow.
- Review `Large-Scale-Optimization/README.md` and notebooks for optimization experiments.
- Use the package directories in `deep_gradients/`, `build-sparse-autoencoder/`, and `learn-rl/` as code-first project entry points.

## Notes

- Large model caches and generated artifacts should stay out of Git.
- Several subprojects are still exploratory, so some directories currently have minimal README coverage.
- Contributor attribution is normalized with `.mailmap` to reduce duplicate identities in Git tooling and GitHub.

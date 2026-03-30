# Learn AI

This repository contains multiple machine learning projects spanning language modeling, optimization, interpretability, computer vision, reinforcement learning, and autonomous research workflows.

## Table of Contents

- [Repository Layout](#repository-layout)
- [NanoGPT](#nanogpt)
- [Build NanoGPT](#build-nanogpt)
- [Build Sparse Autoencoder](#build-sparse-autoencoder)
- [Deep Gradients](#deep-gradients)
- [Large-Scale Optimization](#large-scale-optimization)
- [Autoresearch](#autoresearch)
- [Learn RL](#learn-rl)
- [Dataset Availability](#dataset-availability)
- [Environment Setup](#environment-setup)
- [Usage](#usage)
- [Contributing](#contributing)
- [Notes](#notes)

## Repository Layout

- `NanoGPT/`: lecture notebooks and exploratory GPT development material
- `build-nanogpt/`: from-scratch NanoGPT build and training work
- `build-sparse-autoencoder/`: sparse autoencoder experiments and training code
- `deep_gradients/`: CNN and gradient-based learning experiments
- `Large-Scale-Optimization/`: optimization notebooks and supporting material
- `autoresearch/`: autonomous LLM pretraining research experiments
- `learn-rl/`: reinforcement learning package scaffold for future work

## NanoGPT

NanoGPT is a minimal implementation of a Generative Pre-trained Transformer model in PyTorch. This part of the repository is used to understand transformer training, fine-tuning, and model development through lecture material and exploratory notebooks.

### Features

- Basic transformer architecture with the ability to load pre-trained GPT-2 weights
- Distributed Data Parallelism for utilizing multiple GPUs
- Model checkpointing for saving and resuming training progress
- Performance optimizations including mixed precision and flash attention
- Configurable hyperparameters aligned with GPT-style training settings
- Training and validation work beyond Tiny Shakespeare-scale experiments

### Distributed Data Parallelism

NanoGPT supports distributed data parallelism to leverage multiple GPUs for training. Use the appropriate `torch.distributed` setup and launch configuration in the training scripts when running multi-GPU jobs.

### Model Checkpoints

Model and optimizer checkpoints are saved during training so work can be resumed without restarting from scratch.

### Performance Improvements

- Mixed precision training for faster computation and reduced memory usage
- Integration of `torch.compile` for optimized model execution
- Flash attention for faster attention computation

### Development

Key development work includes:

- Initial transformer implementation
- Loading and fine-tuning GPT-2 weights
- Iterative improvements in distributed training, checkpointing, and performance

## Build NanoGPT

This project contains the build process and practical improvements for NanoGPT. It is the most complete code-first transformer project in the repository and is the best starting point for hands-on GPT training work.

### Highlights

- Initial build of the NanoGPT model
- Loading and fine-tuning pre-trained GPT-2 weights
- Distributed training and performance optimizations
- Experimentation around training datasets and hyperparameters

## Build Sparse Autoencoder

This project contains sparse autoencoder experiments focused on representation learning and interpretability workflows. The directory includes training code, notebooks, and package scaffolding for future SAE iterations.

### Highlights

- Sparse autoencoder training code in PyTorch
- Notebook-based experimentation and playbooks
- Local model cache support kept out of Git

## Deep Gradients

The goal of this project is to implement and study convolutional neural networks from first principles on smaller datasets. It is aimed at understanding how training dynamics and architectural choices affect learning.

### Objectives

- Implement CNN and ResNet-style ideas from scratch
- Compare execution across CPU and GPU environments
- Understand CUDA-related training speedups
- Observe how weights evolve across layers during training
- Study batch normalization and residual connections

This project is meant to build intuition by deconstructing the framework rather than only using high-level abstractions.

## Large-Scale Optimization

This project is divided into two broad areas of optimization work.

### Part 1

Implementation of first-order and second-order methods for multiclass logistic regression, including Hessian-based methods. Covered methods include:

- Gradient Descent
- Newton-Raphson
- Stochastic Gradient Descent
- Minibatch SGD
- Minibatch Gradient Descent
- SVRG

For Hessian calculations and derivation, the earlier README referenced this helpful [blog](http://fourier.eng.hmc.edu/e176/lectures/ch7/node14.html).

### Part 2

Implementation of subgradient and proximal gradient methods for a data denoising task.

## Autoresearch

`autoresearch/` contains autonomous LLM pretraining experiments based on `karpathy/autoresearch`. The idea is to let an agent iteratively modify training code, run short bounded experiments, and keep changes that improve validation performance.

### Setup Summary

- Platform: Minerva HPC with NVIDIA H100 80GB GPUs
- Model: GPT with Muon plus AdamW, Flash Attention 3, and `torch.compile`
- Metric: validation bits per byte
- Workflow: repeated short experiments with incremental code changes

## Learn RL

`learn-rl/` is currently an early reinforcement learning scaffold. It is intended as a place to grow future RL experiments, package structure, and tests without mixing that work into the older projects.

## Dataset Availability

Current Minerva dataset state inside the Learn-AI workspace:

- Ready to use: `NanoGPT/nanoGPT/data/shakespeare_char/` with `train.bin`, `val.bin`, `meta.pkl`, and `input.txt`
- Small text inputs for demos and lecture work: `NanoGPT/nanoGPT/data/input.txt` and `NanoGPT/nanoGPT_lecture/dataset/input.txt`
- Prep scaffolds present but not populated as local corpora: `NanoGPT/nanoGPT/data/openwebtext/`, `NanoGPT/nanoGPT/data/shakespeare/`, and `build-nanogpt/build_nanogpt/edu_fineweb10B/`
- Local model cache area exists at `model_cache_dir/`, but it is currently small and should be treated as a cache, not a durable dataset store
- Checkpoints such as `build-nanogpt/checkpoint.pth` and related files are training artifacts, not datasets

Practical default: if you want a dataset that is already prepared and usable on Minerva, start with `NanoGPT/nanoGPT/data/shakespeare_char/`.

## Environment Setup

There is no single root environment for the entire repository. Most directories are intended to be used independently.

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

## Usage

Usage varies by project. In general:

- For package-based projects, change into the project directory and install dependencies there
- For notebook-heavy projects, open the relevant notebook and configure the environment locally
- For transformer training work, start with `build-nanogpt/README.md`
- For autonomous research experiments, start with `autoresearch/README.md`

## Contributing

Contributions are welcome through normal GitHub workflow.

1. Fork the repository.
2. Create a branch for your work.
3. Make and test your changes.
4. Commit with a clear message.
5. Push the branch and open a pull request.

## Notes

- Large model caches and generated artifacts should stay out of Git.
- Several subprojects are still exploratory, so some directories currently have minimal README coverage.
- Contributor attribution is normalized with `.mailmap` to reduce duplicate identities in Git tooling and GitHub.

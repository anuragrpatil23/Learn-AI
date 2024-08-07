# Combined Projects Repository

This repository contains multiple projects focused on different aspects of deep learning and optimization. Below is a brief description of each project and its goals.

## Table of Contents

- [NanoGPT](#nanogpt)
- [Deep Gradients](#deep-gradients)
- [Large-Scale Optimization](#large-scale-optimization)
- [Build NanoGPT](#build-nanogpt)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## NanoGPT

NanoGPT is a minimal implementation of a Generative Pre-trained Transformer (GPT) model using PyTorch. This project aims to provide a clear and straightforward example of training and fine-tuning a transformer-based language model, leveraging modern deep learning techniques for efficient and scalable training.

### Features

- Basic transformer architecture with the ability to load pre-trained GPT-2 weights.
- Distributed Data Parallelism for utilizing multiple GPUs.
- Model checkpointing for saving and resuming training progress.
- Performance optimizations including mixed precision and flash attention.
- Configurable hyperparameters aligned with GPT-3 settings.
- New training and validation datasets beyond the Tiny Shakespeare dataset.

### Installation

1. Clone the repository:


2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

To train the model, use the following command:
    ```bash
    python train.py
    ```

### Distributed Data Parallelism

NanoGPT supports distributed data parallelism to leverage multiple GPUs for training. To enable this, ensure you have the `torch.distributed` module set up and use the appropriate configuration in your training script.

### Model Checkpoints

The model and optimizer checkpoints are saved at the end of specified epochs. This allows you to resume training from a saved state, ensuring efficient use of computational resources. Checkpoints are saved at the end of epoch 1 and epoch 5 by default.

### Performance Improvements

NanoGPT includes several performance optimizations:
- Mixed precision training for faster computation and reduced memory usage.
- Integration of `torch.compile` for optimized model execution.
- Flash attention mechanisms for improved attention computations.

### Hyperparameter Tweaks

The default hyperparameters are set to align with those used in GPT-3, including:
- AdamW optimizer betas.
- Cosine decay learning rate schedule.

These tweaks ensure that the model performs optimally and aligns with state-of-the-art practices.

### Training and Validation Datasets

NanoGPT moves beyond the limitations of the Tiny Shakespeare dataset by introducing new, more comprehensive training and validation datasets. This allows for more robust and diverse model training.

### Results

Training results from various epochs are logged and saved:
- Initial training epoch results.
- Results from a comprehensive 5-epoch run.

These logs provide insights into the model's performance and improvements over time.

### Development

Key development milestones and improvements:
- Initial build of the NanoGPT model with basic transformer architecture.
- Loading and fine-tuning pre-trained GPT-2 weights.
- Incremental enhancements in distributed training, checkpointing, and performance optimizations.

## Deep Gradients

The goal of this project is to implement a Convolutional Neural Network (CNN) from scratch for a basic toy dataset, MNIST. The main objectives are:

- Implement a modern ResNet architecture running across 3 GPU nodes, 1 GPU, and 1 CPU.
- Compare speeds, GPU utilization, and benefits of using multiple GPUs.
- Understand how CUDA optimizations help train bigger networks.
- Observe how weights evolve across layers during training.
- Understand batch normalization and residual blocks.

This project aims to provide a good understanding of CNNs by deconstructing the framework and visualizing the features detected across layers.

## Large-Scale Optimization

This project is divided into two parts:

### Part 1

Implementation of first-order and second-order methods, including Hessian calculations, for multiclass logistic regression. Methods include:

- Gradient Descent
- Newton-Raphson
- Stochastic Gradient Descent (SGD)
- Minibatch SGD
- Minibatch Gradient Descent
- SVRG

For Hessian calculations and derivation, refer to this excellent [blog](http://fourier.eng.hmc.edu/e176/lectures/ch7/node14.html).

### Part 2

Implementation of Subgradients and Proximal gradients for a data denoising task.

## Build NanoGPT

This project contains the build process and improvements for NanoGPT. Key features and milestones include:

- Initial build of the NanoGPT model.
- Loading and fine-tuning pre-trained GPT-2 weights.
- Distributed training and performance optimizations.

## Installation

For each project, navigate to the respective directory and follow the installation instructions provided in the individual README files or documentation.

## Usage

Refer to the specific usage instructions provided in each project's README file or documentation.

## Contributing

Contributions are welcome! Please fork the repository and submit pull requests for any enhancements or bug fixes.

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a new Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

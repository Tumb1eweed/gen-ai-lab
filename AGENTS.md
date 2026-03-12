# AGENTS.md

## Project Purpose

This repository is a hands-on learning lab for generative AI. The learning and implementation order is:

1. AE
2. VAE
3. CVAE
4. VQ-VAE
5. Diffusion by U-Net
6. DiT

The repository should evolve stage by stage. Do not skip ahead to later models before the current stage is stable.

## Current Active Stage

The active stage is AE.

The current goal is not image quality or advanced optimization. The current goal is to make the full autoencoder training loop on CIFAR-10 understandable and runnable.

Agents working in this repository should prioritize these questions:

1. How the encoder, latent vector, and decoder fit together in a deterministic autoencoder
2. Why reconstruction-only training is a useful baseline before introducing probabilistic latents and KL regularization in VAE
3. How to inspect reconstructions and validation behavior during training

## Current Status

- AE: in progress
- VAE: in progress after AE is stable
- CVAE: not started
- VQ-VAE: not started
- Diffusion by U-Net: not started
- DiT: not started

## Implementation Priorities

When working on the AE stage, prefer minimal and readable solutions.

1. Keep the implementation focused on CIFAR-10.
2. Prefer a small convolutional autoencoder over a larger or more abstract architecture.
3. Keep the objective focused on reconstruction loss and report train and validation loss clearly.
4. Save checkpoints, reconstructions, and loss curves during training.
5. Favor simple scripts and explicit code over premature abstractions.

## What To Avoid Right Now

Until the AE stage is stable, avoid introducing:

1. multi-model frameworks
2. large-scale refactors
3. heavy configuration systems
4. mixed precision or distributed training complexity
5. optimization for benchmark quality over clarity

## Expected Workflow

For AE work, the recommended sequence is:

1. Read the local AE documentation.
2. Run a small training job.
3. Inspect validation reconstructions and loss curves.
4. Adjust a small number of core hyperparameters such as latent dimension and learning rate.
5. Compare the deterministic autoencoder behavior with the later VAE stage before moving on.

## Repository Convention

Changes should be incremental and easy to inspect. Keep code and documentation aligned. If a behavior changes, update the relevant README in the same pass.
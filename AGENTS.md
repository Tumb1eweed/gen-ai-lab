# AGENTS.md

## Project Purpose

This repository is a hands-on learning lab for generative AI. The learning and implementation order is:

1. VAE
2. CVAE
3. VQ-VAE
4. Diffusion by U-Net
5. DiT

The repository should evolve stage by stage. Do not skip ahead to later models before the current stage is stable.

## Current Active Stage

The active stage is VAE.

The current goal is not image quality or advanced optimization. The current goal is to make the full VAE training loop on CIFAR-10 understandable and runnable.

Agents working in this repository should prioritize these questions:

1. How the encoder, latent distribution, reparameterization, and decoder fit together
2. Why the VAE objective contains both reconstruction loss and KL divergence
3. How to inspect reconstructions and random samples during training

## Current Status

- VAE: in progress
- CVAE: not started
- VQ-VAE: not started
- Diffusion by U-Net: not started
- DiT: not started

## Implementation Priorities

When working on the VAE stage, prefer minimal and readable solutions.

1. Keep the implementation focused on CIFAR-10.
2. Prefer a small convolutional VAE over a larger or more abstract architecture.
3. Log total loss, reconstruction loss, and KL loss separately.
4. Save checkpoints, reconstructions, and random samples during training.
5. Favor simple scripts and explicit code over premature abstractions.

## What To Avoid Right Now

Until the VAE stage is stable, avoid introducing:

1. multi-model frameworks
2. large-scale refactors
3. heavy configuration systems
4. mixed precision or distributed training complexity
5. optimization for benchmark quality over clarity

## Expected Workflow

For VAE work, the recommended sequence is:

1. Read the local VAE documentation.
2. Run a small training job.
3. Inspect reconstructions and random samples.
4. Adjust a small number of core hyperparameters such as latent dimension, beta, and learning rate.
5. Only move to the next model family after the VAE pipeline is stable and understandable.

## Repository Convention

Changes should be incremental and easy to inspect. Keep code and documentation aligned. If a behavior changes, update the relevant README in the same pass.
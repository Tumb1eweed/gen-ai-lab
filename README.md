# gen-ai-lab

This repository is a hands-on learning lab for generative AI. The implementation order is:

1. AE
2. VAE
3. CVAE
4. VQ-VAE
5. Diffusion by U-Net
6. DiT

The current active stage is AE. The first goal is not image quality. It is to build a complete deterministic autoencoder training loop on CIFAR-10 and understand:

1. How the encoder, latent vector, and decoder work together
2. Why reconstruction-only training is a useful baseline before introducing the VAE KL term
3. How to inspect reconstructions and validation behavior during training

## Current Status

- AE: in progress
- VAE: in progress after AE is stable
- CVAE: not started
- VQ-VAE: not started
- Diffusion by U-Net: not started
- DiT: not started

## First Running Target

The first runnable target lives in [AE](AE). It is a small convolutional autoencoder for CIFAR-10 with:

- train, validation, and test splits
- training checkpoints
- reconstruction image dumps
- train and validation loss curves
- a simple deterministic latent bottleneck for baseline comparison with VAE

## Suggested Learning Rhythm

1. Read the AE README and understand the reconstruction objective.
2. Run one epoch and inspect the saved reconstructions and loss curves.
3. Modify latent dimension and learning rate to see how training changes.
4. After the AE loop is stable, move to VAE to add probabilistic latents and KL regularization.
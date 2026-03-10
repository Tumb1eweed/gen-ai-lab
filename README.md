# gen-ai-lab

This repository is a hands-on learning lab for generative AI. The implementation order is:

1. VAE
2. CVAE
3. VQ-VAE
4. Diffusion by U-Net
5. DiT

The current active stage is VAE. The first goal is not image quality. It is to build a complete training loop on CIFAR-10 and understand:

1. How the encoder, latent distribution, reparameterization, and decoder work together
2. Why the VAE loss is a sum of reconstruction loss and KL divergence
3. How to inspect reconstructions and random samples during training

## Current Status

- VAE: in progress
- CVAE: not started
- VQ-VAE: not started
- Diffusion by U-Net: not started
- DiT: not started

## First Running Target

The first runnable target lives in [VAE](VAE). It is a small convolutional VAE for CIFAR-10 with:

- automatic CIFAR-10 download
- training checkpoints
- reconstruction image dumps
- random latent sampling
- separate logging for total loss, reconstruction loss, and KL loss

## Suggested Learning Rhythm

1. Read the VAE README and understand the training objective.
2. Run one epoch and inspect the saved reconstructions and samples.
3. Modify latent dimension, beta, and learning rate to see how training changes.
4. After the VAE loop is stable, move to conditional generation with CVAE.
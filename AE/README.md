# Autoencoder on CIFAR-10

This directory contains a simple convolutional autoencoder baseline on CIFAR-10.

Unlike the VAE implementation, this model uses a deterministic latent vector and only optimizes reconstruction loss. It is useful as a comparison point when learning why VAE reconstructions are often blurrier and why the KL term changes the behavior of the latent space.

## Goal

Build a minimal and readable autoencoder pipeline with:

1. train split
2. validation split
3. test evaluation
4. checkpoints
5. reconstruction image dumps
6. train and validation loss curves

## Files

- `config.py`: default hyperparameters and paths
- `dataset.py`: CIFAR-10 train, validation, and test dataloaders
- `model.py`: convolutional autoencoder
- `loss.py`: reconstruction loss
- `utils.py`: plotting, checkpointing, and image helpers
- `train.py`: train and validate the model
- `test.py`: evaluate a saved checkpoint on the test split

## Environment

This implementation can use the same conda environment as the VAE stage.

```bash
conda activate vae
```

If you want an environment file inside this folder as well:

```bash
conda env create -f environment.yml
conda activate ae
```

## Run Training

```bash
python train.py --epochs 50 --device cuda
```

The script trains on the training split and reports validation loss at the end of every epoch.

## Run Test

```bash
python test.py --checkpoint outputs/checkpoints/ae_best.pt --device cuda
```

The test script evaluates reconstruction loss on the test split, saves a reconstruction grid, and also decodes standard normal latent noise so you can compare what the AE decoder produces without an encoded input image.

## Outputs

- `outputs/checkpoints/`: last and best checkpoints
- `outputs/plots/`: train and validation loss curves
- `outputs/reconstructions/`: validation reconstruction grids by epoch
- `outputs/test/`: final test reconstruction grid and normal-noise decoder samples

## What To Compare With VAE

1. AE usually reconstructs more sharply than VAE.
2. AE does not learn a regularized probabilistic latent space.
3. AE has no KL term, so the loss is simpler but the latent representation is less structured for generation.
# VAE on CIFAR-10

This directory contains the first working stage of the learning lab: a compact convolutional variational autoencoder trained on CIFAR-10.

## Goal

The purpose of this implementation is to make the full VAE loop concrete:

1. Encode an image into a latent distribution parameterized by mean and log variance.
2. Sample a latent vector with the reparameterization trick.
3. Decode the latent vector back into an image.
4. Train with reconstruction loss plus KL divergence.

The first version is intentionally small. It prioritizes readability and a stable training loop over image quality.

## Files

- `config.py`: default hyperparameters and runtime paths
- `dataset.py`: CIFAR-10 dataloaders and normalization helpers
- `model.py`: convolutional VAE model
- `loss.py`: VAE objective split into total, reconstruction, and KL terms
- `utils.py`: checkpoint and image saving helpers
- `train.py`: training entrypoint
- `sample.py`: load a checkpoint and sample images from the latent prior

## Environment

This project uses a dedicated conda environment named `vae`.

Create it from this directory:

```bash
conda env create -f environment.yml
conda activate vae
```

If the environment already exists, update it with:

```bash
conda env update -f environment.yml --prune
conda activate vae
```

## Run Training

From this directory:

```bash
python train.py --epochs 100 --batch-size 128
```

By default the script uses `--device auto`, which means it prefers GPU when CUDA is available and otherwise falls back to CPU. You can force GPU explicitly with:

```bash
python train.py --device cuda
```

If a run is interrupted and `outputs/checkpoints/vae_last.pt` exists, resume it with:

```bash
python train.py --device cuda --resume-checkpoint outputs/checkpoints/vae_last.pt
```

Useful options:

```bash
python train.py --latent-dim 128 --beta 1.0 --learning-rate 1e-3
```

## Run Sampling

After training:

```bash
python sample.py --checkpoint outputs/checkpoints/vae_last.pt --num-samples 64
```

If you prefer not to activate the environment in the shell, you can run the interpreter directly:

```bash
/home/chenrui/miniconda3/envs/vae/bin/python train.py --epochs 100 --batch-size 128
```

## Outputs

Training artifacts are written to `outputs/`:

- `outputs/checkpoints/`: model checkpoints
- `outputs/plots/`: loss history JSON and loss curve figure
- `outputs/reconstructions/`: side-by-side input and reconstruction grids
- `outputs/samples/`: random samples from the latent prior

## What To Inspect First

1. Whether total loss, reconstruction loss, and KL loss all print during training
2. Whether reconstructions roughly preserve color and coarse object layout
3. Whether random samples are no longer pure noise after a few epochs

## Common First Adjustments

If training is unstable or collapses early, adjust these first:

1. `--beta`
2. `--latent-dim`
3. `--learning-rate`
4. decoder capacity in `model.py`
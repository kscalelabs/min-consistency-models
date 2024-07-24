<p align="center">
  <picture>
    <img alt="K-Scale Open Source Robotics" src="https://media.kscale.dev/kscale-open-source-header.png" style="max-width: 100%;">
  </picture>
</p>

<div align="center">

[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/kscalelabs/ksim/blob/main/LICENSE)
[![Discord](https://img.shields.io/discord/1224056091017478166)](https://discord.gg/k5mSvCkYQh)
[![Wiki](https://img.shields.io/badge/wiki-humanoids-black)](https://humanoids.wiki)
<br />
[![python](https://img.shields.io/badge/-Python_3.11-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![ruff](https://img.shields.io/badge/Linter-Ruff-red.svg?labelColor=gray)](https://github.com/charliermarsh/ruff)
<br />
[![Python Checks](https://github.com/kscalelabs/min-consistency-models/actions/workflows/test.yml/badge.svg)](https://github.com/kscalelabs/min-consistency-models/actions/workflows/test.yml)

</div>

# Consistency Models

Minimal implementation of consistency models in PyTorch.

This repository was the code created as a part of Nathan's talk on Flow Matching, which also leaked a bit into Consistency Modeling, a useful optimization for flow matching that reduces the number of sampling steps required in order to reach final model prediction.

The key idea is moving from noisy images to non-noisy images by following a flow "path" between Gaussian noise and the target image!

## Getting Started

Simply run `train.py`! Read through the code to understand annotations, interchange the dataset, experiment with different hyperparameters/loss functions, etc.

## Command Line Arguments

To customize the training of the consistency models, the following command line arguments can be used:

- `--prefix`: Prefix for checkpoint and output names. This is useful for organizing different experiments. Default is an empty string.
- `--n_epochs`: Number of epochs for which the model will be trained. Default is 100.
- `--output_dir`: The directory where checkpoints and generated images will be saved. Default is `./contents`.
- `--device`: The CUDA device to use for training. Default is `cuda:0`. If you're using a CPU, you can change this to `cpu`.
- `--loss_type`: The type of loss function to use. Can be either `mse` for Mean Squared Error or `huber` for Huber loss. Default is `mse`.
- `--partial_sampling`: Enables partial sampling, which can be useful for reducing the number of sampling steps required to reach the final model prediction. This is disabled by default and can be enabled by adding this flag without any value.

### Example Usage

To run the training with specific options, you can use the command line as follows:

```bash
python train.py \
  --prefix experiment1 \
  --n_epochs 200 \
  --output_dir ./experiment1_outputs \
  --device cuda:0 \
  --loss_type huber \
  --partial_sampling
```

To run inference from a model checkpoint, you can use the command line as follows:

```bash
python infer.py \
  --device cuda:0 \
  --prefix experiment1_ \
  --output_dir ./experiment1_outputs
```

### Contributing

See the [contributing guide](CONTRIBUTING.md) to get started.

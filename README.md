# min-consistency-models

Minimal implementation of consistency models in PyTorch.

This repository was the code created as a part of Nathan's talk on Flow Matching, which also leaked a bit into Consistency Modeling, and useful optimization for flow matching that reduces the number of sampling steps required in order to reach final model prediction

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
python train.py --prefix experiment1 --n_epochs 200 --output_dir ./experiment1_outputs --device cuda:0 --loss_type huber --partial_sampling
```

## Miscellaneous
TODO
- [ ] Latent Consistency Modeling with a VAE

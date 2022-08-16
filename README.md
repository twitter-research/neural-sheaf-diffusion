# Neural Sheaf Diffusion

This repository contains the official code for the paper 
[Neural Sheaf Diffusion: A Topological Perspective on Heterophily and Oversmoothing in GNNs](https://arxiv.org/abs/2202.04579).

## Getting started

We used `CUDA 10.2` for this project. To set up the environment, run the following command:

```commandline
conda env create --file=environment_gpu.yml
conda activate nsd
```

To make sure that everything is set up correctly, you can run all the tests using:
```commandline
pytest -v .
```

## Running experiments

First disable weights and biases by running `wandb disabled`. For an example run on
`texas` run from the main directory:

```commandline
sh ./exp/scripts/run_texas.sh
```

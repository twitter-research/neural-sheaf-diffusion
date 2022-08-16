# Neural Sheaf Diffusion

This repository contains the official code for the paper 
[Neural Sheaf Diffusion: A Topological Perspective on Heterophily and Oversmoothing in GNNs](https://arxiv.org/abs/2202.04579).

## Getting started

To set up the environment, run the following command:

```commandline
conda env create --name nsd --file=environment.yml
conda activate nsd
```

To make sure that everything is set up correctly, you can run all the tests using:
```commandline
pytest -v .
```
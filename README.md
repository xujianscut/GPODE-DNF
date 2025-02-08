

# Bayesian Gaussian Process ODEs via Double Normalizing Flows

## Overview

This repository implements a novel method for solving Ordinary Differential Equations (ODEs) using Bayesian Gaussian Processes (GPs) combined with Double Normalizing Flows. Our approach aims to model the uncertainty in ODE solutions in a probabilistic framework, leveraging the power of GPs to handle non-linearity and uncertainty propagation. The method also uses normalizing flows to efficiently approximate the posterior distribution, improving the flexibility and scalability of the model.

## Key Features

- **Bayesian Gaussian Process ODE Solver**: Use a Gaussian process to model the latent function in ODEs with uncertainty quantification.
- **Double Normalizing Flows**: Enhance the model's flexibility in approximating complex posterior distributions through the use of normalizing flows.
- **Scalability**: The approach is designed to scale to high-dimensional systems by leveraging efficient variational inference techniques.
- **Uncertainty Quantification**: Our model provides probabilistic solutions to ODEs, offering not only point estimates but also credible intervals, useful for real-world applications.



## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- Scikit-learn




## Citation

If you use this code in your research or projects, please cite the paper:

@inproceedings{
xu2025bayesian,
title={Bayesian Gaussian Process {ODE}s via Double Normalizing Flows},
author={JIAN XU and Shian Du and Junmei Yang and Xinghao Ding and Delu Zeng and John Paisley},
booktitle={The 28th International Conference on Artificial Intelligence and Statistics},
year={2025},
url={https://openreview.net/forum?id=v6hI4jxD1b}
}


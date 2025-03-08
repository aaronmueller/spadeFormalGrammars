# Projecting Assumptions: The Duality Between Sparse Autoencoders and Concept Geometry

Codebase for the paper ["Projecting Assumptions: The Duality Between Sparse Autoencoders and Concept Geometry"](https://arxiv.org/abs/2503.01822).

## Broad notes on usage
- The data-generating process is in folder *dgp*
- Train configs can be located in folder *config*

- To train a base model, use `python train_basemodel.py` (a pretrained base model is already provided in `trained_models/final/`)
    - Some basic evals to check grammaticality of generations have been implemented in *evals*

- To train SAEs, use `python train_saes.py` (pretrained SAEs of different types are provided in `trained_models/final/`)
    - The base model has to be specified in SAE training config
    - Default metrics track sparsity and fidelity metrics like reconstruction error

- To reproduce figures from the paper, refer to the `figures` directory
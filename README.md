Improved Training of Wasserstein GANs
=====================================

This is a project test Wasserstein GAN objectives on language. The code is built on a fork of [the popular project under the same title](https://github.com/igul222/improved_wgan_training).

We try to reproduce results from their [paper](https://arxiv.org/abs/1704.00028). We clean their code for language generation, try smaller datasets, standard preprocessing and slightly different architectures.

We striped a lot of unused code to better understand the code

## Datasets
You can download Download Google Billion Word at [http://www.statmt.org/lm-benchmark/] .


## Prerequisites

- Python, NumPy, TensorFlow, SciPy, Matplotlib
- A recent NVIDIA GPU

## Models
Most important is `python gan_language.py`: Character-level language model. It has help.

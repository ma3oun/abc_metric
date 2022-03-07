# Attribution-based Confidence metric for deep neural networks
## Decription

This repository provides a PyTorch implementation of the abc metric as decribed in this [paper](https://proceedings.neurips.cc/paper/2019/file/bc1ad6e8f86c42a371aff945535baebb-Paper.pdf)

## Installation

1. Download or clone the repository
2. Install the requirements

## Usage

You can specify the directory for dataset download by setting the DATASETS_ROOT environment variable.

Scripts for MNIST (with and without background noise) are provided:
```bash
export DATASETS_ROOT="/tmp"
python mnist_baseline.py
python mnist_noise.py
```

A script for Cifar10 is also provided:
```bash
export DATASETS_ROOT="/tmp"
python cifar10.py
```

The abc metric is tested using rotated data or alpha blending between two random samples.
The displayed metrics are the average abc score, the average abc score for correctly classified samples and the average abc score for misclassified samples.

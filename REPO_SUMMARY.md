# Repository summary

## Overview
This repository is a collection of beginner-friendly PyTorch scripts that demonstrate core deep-learning workflows: manual gradient descent, linear and logistic regression, multilayer perceptrons, convolutional networks for MNIST, dataset/dataloader usage, activation-function comparisons, and basic CUDA environment checks. It also includes a small tabular dataset (`diabetes.csv`) and a couple of PDF notes related to CUDA and multiprocessing.

## Contents at a glance
| File | Purpose |
| --- | --- |
| `Back Propagation.py` | Manual gradient updates for a quadratic regression toy example. |
| `SGD.py` | From-scratch SGD on a tiny linear regression task with loss plotting. |
| `PyTorch Linear Model.py` | Linear regression implemented with `torch.nn.Linear` and SGD, including loss visualization. |
| `Logistic regression and BCE loss.py` | Binary logistic regression with BCE loss on synthetic data plus probability plots. |
| `Multiple dimension input.py` | Multilayer network trained on `diabetes.csv` (batch gradient, BCE loss). |
| `Dataloader and dataset.py` | Custom `Dataset` + `DataLoader` pipeline for `diabetes.csv` with a deeper MLP and GPU support. |
| `three activate function comparison.py` | Compares Sigmoid/Tanh/ReLU activations on the diabetes task over repeated runs. |
| `MNIST training.py` | Fully connected classifier for MNIST with SGD and progress logging. |
| `MINST training2.py` | CNN-based MNIST classifier using Adam, with train/test loops. |
| `CUDAtest.py` | Quick CUDA availability, device, and architecture information. |
| `diabetes.csv` | Tabular dataset (8 features + label) used by the diabetes-related scripts. |
| `CUDA和PyTorch不匹配？.pdf`, `鱼与熊掌不可得兼！——多进程CPU与CUDA在传统操作系统的纠结.pdf` | Reference notes about CUDA/PyTorch compatibility and multiprocessing considerations. |

## Running the examples
- Python 3.x with the following libraries: `torch`, `torchvision` (for MNIST scripts), `numpy`, and `matplotlib`.
- Run any script directly, e.g.:
  - `python "CUDAtest.py"` to inspect CUDA support.
  - `python "MNIST training.py"` or `python "MINST training2.py"` to train MNIST models (downloads data to `./data`).
  - `python "Dataloader and dataset.py"` to train an MLP on `diabetes.csv` using a DataLoader and optional GPU acceleration.
- Most training scripts print periodic loss/accuracy and, where applicable, display matplotlib charts.

## Notes and testing
- No automated test suite is present; `pytest` is not installed in this repository. Scripts are intended to be run individually for experimentation and learning.

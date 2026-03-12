# DA-MANet: A Dynamic Feedback Multi-Scale Attention Network for Lithology Identification

## 1. Introduction & Motivation
This repository contains the official Python implementation of **DA-MANet** (Dynamic Feedback Multi-Scale Attention Network) alongside baseline models such as LMAFNet.

**Why use this code?**
In the field of geosciences, lithology identification using well logging data are critical yet challenging tasks due to the complexity of geological formations and the non-linear nature of well-log signals. DA-MANet addresses this by integrating a Multi-Scale Attention mechanism (`MA.py`) with a dynamic feedback network. This architecture effectively captures both multi-scale local features and global long-range dependencies in well-log curves, significantly improving identification accuracy and robustness in complex environments.

**How to use this code?**
Researchers and practitioners can use this repository to train DA-MANet on their own well-log datasets, evaluate its classification performance, or adapt the multi-scale attention modules for other deep learning applications in geological analysis.

## 2. Repository Structure
To comply with open-source security best practices, all source code files are provided directly in the repository. **No compressed files (e.g., .zip, .rar, .7z) are used**, ensuring maximum transparency and easy code review.

* `DA_MANet.py` / `DA_MANet_model.py`: Core model architecture and network definition for DA-MANet.
* `LMAFNet.py` / `LMAFNet_model.py`: Definition for the baseline comparison model (LMAFNet).
* `MA.py`: The core Multi-Scale Attention module.
* `train.py` / `LMAFNet_train.py`: Training scripts for the respective models.
* `load_data.py`: Data loading, cleaning, and preprocessing module for well-log datasets.
* `util.py`: Utility functions for metric calculation and logging.

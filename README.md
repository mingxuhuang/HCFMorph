# HCFMorph: Hybrid Contribution-Aware Fusion Network for Multi-temporal Medical Image Registration

[![Paper](https://img.shields.io/badge/Paper-BIBM2025-blue)]()
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)]()

Official implementation of **HCFMorph**, a hybrid contribution-aware fusion network for multi-temporal medical image registration, accepted at BIBM 2025.

## 📖 Abstract

Longitudinal deformable registration is crucial for precision radiotherapy but is challenged by large-scale organ deformations and anatomical coupling. While existing deep learning registration methods have shown promise, they often face two key limitations: (1) a trade-off in feature extraction between local details and long-range dependencies, and (2) feature distortion from the naive fusion of spatially misaligned features. 

To overcome these challenges, we propose HCFMorph, a novel, unified coarse-to-fine registration framework. Our framework introduces three key innovations: 
- **Synergistic Mamba-Conv module** to model coupled organ motion by capturing global-local features
- **Contribution-Aware Feature Fusion module** to adaptively fuse features from the fixed and moving images, thereby mitigating feature misalignment
- **Deformation-Guided Hierarchical Refinement strategy** in the decoder, which addresses large-scale deformations by recovering the field in a coarse-to-fine manner and leverages low-resolution fields to guide and pre-align high-resolution features

Experiments on a clinical dataset of 432 longitudinal CT scans from 97 cervical cancer patients show that HCFMorph significantly outperforms state-of-the-art methods. It achieves dominant anatomical accuracy while maintaining excellent topological integrity with a near-zero Jacobian folding rate, striking a superior balance between accuracy and plausibility.

## 🚀 Quick Start

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/mingxuhuang/HCFMorph.git
   cd HCFMorph
2. **Install dependencies**
conda env create -f environment.yml
conda activate hcfmorph
3. **Training**
HCFMorph/scripts/torch/train_cross.py
4. **Testing/Inference**
HCFMorph/scripts/torch/testVmambamorph.py

## 🤝 Citation
@inproceedings{huang2025hcfmorph,
  title={HCFMorph: Hybrid Contribution-Aware Fusion Network for Multi-temporal Medical Image Registration},
  author={Mingxu huang and Chaolu Feng, Yuhua Gao, Ming Cui, Dazhe Zhao and Deyu Sun},
  booktitle={IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  year={2025}
}

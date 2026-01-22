# Frequency-Aware Contrastive Learning and Spectral Disentanglement for Unsupervised Image Deraining

## Abstract
We reformulate unsupervised image deraining as a signal separation task, addressing the practical challenge of acquiring paired training data. Existing methods often neglect the intrinsic physical prior of the rain signal: its directional high-frequency energy in the Fourier amplitude spectrum. Based on this insight, our method is driven by two core innovations. First, a novel frequency amplitude loss acts as an unsupervised spectral separation constraint, penalizing any spectral overlap to ensure the recovered background is verifiably free of rain's unique signature. Second, a dual-domain contrastive learning strategy trains a rain streak extractor that accurately models the joint spatio-spectral priors of rain streaks and learns their consistent characteristics. This robust modeling allows our framework to surpass all unsupervised benchmarks and demonstrate superior generalization over state-of-the-art supervised methods on real-world datasets, validating the superiority of our signal prior-based approach. The source code is publicly available at https://github.com/Windrain7/DDCL.

## Requirements

### Environment Setup
```bash
# Create environment from file
conda env create -f environment.yml

# Activate the environment
conda activate ddcl
```
### Key Dependencies
- torch >= 2.4.0
- torchvision >= 0.19.0
- numpy >= 1.26.4
- scikit-image >= 0.23.2
- opencv-python >= 4.10.0
- tensorboard >= 2.17.0
- tqdm, pandas, pyyaml

**Note**: This project requires NVIDIA GPU with CUDA support for training and inference.

## Datasets

The following datasets are used and can be downloaded from the provided links:
- [Rain100L](https://pan.baidu.com/s/1gNSuK8-52gGXCeUPtJfBOg?pwd=t326): Light rain dataset with 200 training and 100 testing pairs
- [Rain200L](https://pan.baidu.com/s/1WX9ldHmgw1P3t-WMAvlmLQ?pwd=t326): Light rain dataset with 1800 training and 200 testing pairs
- [Rain200H](https://pan.baidu.com/s/1wcRNgUtb1as20rfKRykZmQ?pwd=t326): Heavy rain dataset with 1800 training and 200 testing pairs
- [Rain800](https://pan.baidu.com/s/1WUqX4S73FOY8A_jEq1y-jQ?pwd=t326): Dataset with 700 training and 100 testing pairs
- [SPA-Data](https://www.kaggle.com/datasets/leftthomas/spadata): Real-world rain dataset

### Dataset Preparation

Download the datasets and organize them in the following structure:

```
./datasets/
├── Rain100L/
│   ├── train/
│   │   ├── input/    # Rainy images
│   │   └── target/   # Clean images
│   └── test/
│       ├── input/
│       └── target/
├── Rain200L/
│   ├── train/
│   └── test/
├── Rain200H/
│   ├── train/
│   └── test/
├── Rain800/
│   ├── train/
│   └── test/
└── SPA-Data/
    └── ...
```
## Usage

### Quick Start

#### Training
```bash
# Train on Rain100L with default settings (uses GPU with least memory)
make train_test

# Or train only
make train
```

#### Testing
```bash
# Test with trained model
make test
```
#### Training on Different Datasets
```bash
# Rain200H
make train_test DATASET=Rain200L
```
## Project Structure

```
DDCL/
├── data/                    # Data loading and preprocessing
│   ├── dataset.py          # Dataset classes
│   └── __init__.py
├── models/                  # Model architectures
│   ├── ddcl.py             # Main DDCL model
│   ├── archs/              # Network architectures
│   │   ├── generator.py    # Generator network
│   │   ├── discriminator.py # Discriminator network
│   │   ├── moco.py         # MoCo contrastive learning
│   │   ├── vgg16.py        # VGG16 for perceptual loss
│   │   └── utils.py        # Architecture utilities
│   └── losses/             # Loss functions
│       └── loss.py         # Custom loss implementations
├── metrics/                 # Evaluation metrics
│   └── metrics.py          # PSNR, SSIM implementations
├── utils/                   # Utility functions
│   ├── options.py          # Command line arguments
│   ├── saver.py            # Save/load checkpoints
│   └── utils.py            # Helper functions
├── statistic/              # MATLAB evaluation scripts
│   ├── compute_metrics.m
│   ├── psnr.m
│   └── ssim.m
├── train.py                # Training script
├── test.py                 # Testing script
├── Makefile                # Build automation
├── environment.yml         # Conda environment file
└── README.md              # This file
```

## Citation

If you find this work helpful, please consider citing:

```bibtex
@article{your_citation,
  title={Frequency-Aware Contrastive Learning and Spectral Disentanglement for Unsupervised Image Deraining},
  author={Your Name},
  journal={Your Venue},
  year={2026}
}
```

## Acknowledgements

This work builds upon several excellent projects:
- [DerainCycleGAN](https://github.com/OaDsis/DerainCycleGAN)
- [MoCo](https://github.com/facebookresearch/moco)
## License

This project is released under the MIT License.
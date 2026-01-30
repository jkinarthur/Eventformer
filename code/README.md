# Eventformer: Frame-Free Vision Transformer for Event Cameras

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the official implementation of **Eventformer**, a novel frame-free Vision Transformer for event camera data. Unlike existing approaches that convert events to frames or voxel grids, Eventformer processes raw events directly as 4D point clouds $(x, y, t, p)$, preserving their native sparse, asynchronous structure.

## Key Contributions

1. **Continuous-Time Positional Encoding (CTPE)**: Multi-scale Fourier features that encode timestamps at arbitrary temporal resolution, eliminating quantization artifacts.

2. **Polarity-Aware Asymmetric Attention (PAAA)**: Separate processing pathways for ON (+1) and OFF (-1) events with asymmetric cross-attention that captures motion direction.

3. **Adaptive Spatiotemporal Neighborhood Attention (ASNA)**: Local attention mechanism where each event attends to k-nearest neighbors with k adapting based on local event density.

## Results

### GEN1 Automotive Detection
| Method | mAP | Parameters | FLOPs |
|--------|-----|------------|-------|
| RED | 42.5 | 35M | 15.1G |
| MatrixLSTM | 38.3 | 18M | 8.2G |
| RVT-B | 44.1 | 28M | 12.3G |
| **Eventformer-B (Ours)** | **47.2** | 22M | 8.5G |

### DVS128 Gesture
| Method | Accuracy |
|--------|----------|
| PointNet++ | 88.8% |
| EST | 92.5% |
| **Eventformer-S (Ours)** | **96.3%** |

## 📁 Project Structure

```
code/
├── models/
│   ├── __init__.py
│   ├── eventformer.py          # Main model architecture
│   ├── ctpe.py                 # Continuous-Time Positional Encoding
│   ├── paaa.py                 # Polarity-Aware Asymmetric Attention
│   └── asna.py                 # Adaptive Spatiotemporal Neighborhood Attention
├── datasets/
│   ├── __init__.py
│   ├── gen1.py                 # GEN1 Automotive Detection
│   ├── ncaltech101.py          # N-Caltech101 Classification
│   ├── dvs_gesture.py          # DVS128 Gesture Recognition
│   └── event_utils.py          # Event processing utilities
├── configs/
│   ├── __init__.py
│   └── config.py               # Configuration classes
├── main.py                     # Entry point
├── train.py                    # Training script
├── evaluate.py                 # Evaluation script
├── ablation.py                 # Ablation studies
├── visualize.py                # Generate figures and graphs
└── requirements.txt
```

## 🚀 Installation

```bash
# Clone repository
git clone https://github.com/yourname/eventformer.git
cd eventformer

# Create environment
conda create -n eventformer python=3.9
conda activate eventformer

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Run Tests
```bash
python main.py --mode test
```

### Training
```bash
# Train on N-Caltech101
python main.py --mode train --dataset ncaltech101 --model tiny --epochs 100

# Train on DVS128 Gesture
python main.py --mode train --dataset dvs128_gesture --model small --epochs 100

# Train on GEN1 (detection)
python main.py --mode train --dataset gen1 --model base --epochs 50
```

### Evaluation
```bash
python main.py --mode eval --checkpoint outputs/best_model.pth --dataset ncaltech101
```

### Ablation Study
```bash
python main.py --mode ablation --dataset ncaltech101 --configs full no_ctpe no_paaa no_asna
```

### Generate Figures
```bash
python main.py --mode visualize --output figures/
```

## 📊 Datasets

Download datasets:
- **GEN1**: https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/
- **N-Caltech101**: https://www.garrickorchard.com/datasets/n-caltech101
- **DVS128 Gesture**: https://research.ibm.com/interactive/dvsgesture/

## 📈 Evaluation

```bash
python evaluate.py --config configs/gen1.yaml --checkpoint checkpoints/best_model.pth
```

## 🔬 Ablation Studies

```bash
python ablation.py --config configs/gen1.yaml --ablation all
```

## 📊 Generate Figures

```bash
python visualize.py --results_dir results/ --output_dir figures/
```

## 📝 Citation

```bibtex
@article{eventformer2026,
  title={Eventformer: Frame-Free Vision Transformers via Spatiotemporal Event Point Clouds},
  author={Authors},
  journal={Conference},
  year={2026}
}
```

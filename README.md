# Image Super-Resolution with RPQNet — ENNCA2026

A deep learning project implementing a full **4× Image Super-Resolution pipeline** using the **RPQNet_WSDR_EV2** architecture on the DIV2K dataset. Covers float training, model optimization, quantization, and pruning.

> Course: ENNCA2026 | Student ID: 342050

---

## Overview

This project builds an end-to-end super-resolution system that upscales low-resolution images by 4× while preserving fine detail. It explores the full model lifecycle from training to deployment-ready optimization.

### Pipeline

```
Float Training → Model Collapse → Post-Training Quantization (PTQ)
                                → Quantization-Aware Training (QAT)
                                → Unstructured Magnitude Pruning
                                → Layer-wise Sensitivity Pruning
```

---

## Results

| Stage | Strategy | PSNR (dB) |
|---|---|---|
| Float (RGB) | Baseline | **24.50** |
| Float (Y-channel) | YUV decomposition | 24.50 |
| PTQ | Per-channel | 24.33 |
| PTQ | Per-tensor | 24.14 |
| QAT | Per-channel | **24.48** |
| QAT | Per-tensor | 24.44 |
| Pruning 10% | Magnitude + fine-tune | 24.50 |
| Pruning 30% | Magnitude + fine-tune | 24.41 |
| Pruning 50% | Magnitude + fine-tune | 23.81 |
| Layer-wise pruning | 3 least-sensitive layers | **24.52** |

---

## Model Architecture

**RPQNet_WSDR_EV2** — a reparameterizable super-resolution network designed for efficient deployment.

| Property | Value |
|---|---|
| Parameters | ~1.67M |
| FLOPs (before collapse) | ~12.96B |
| FLOPs (after collapse) | ~960M |
| Scale factor | 4× |
| Loss function | L1 |
| Optimizer | Adam + Cosine Annealing LR |

Model collapse (reparameterization) reduces FLOPs by **~93%** with no accuracy loss.

---

## Dataset

**DIV2K** — a high-quality image benchmark for super-resolution tasks.

| Split | HR Images | LR Images (4× bicubic) |
|---|---|---|
| Training | 800 | 800 |
| Validation | 100 | 100 |

Download: [DIV2K on NTIRE](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

---

## Tech Stack

| Component | Technology |
|---|---|
| Framework | PyTorch |
| Quantization | `torch.ao.quantization` |
| Pruning | `torch.nn.utils.prune` |
| Training monitoring | TensorBoard |
| Image processing | PIL, torchvision |
| Visualization | Matplotlib |

---

## Getting Started

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended)

### Installation

```bash
git clone https://github.com/Mahmood-Alduhoun/ENNCA2026-Super-Resolution.git
cd ENNCA2026-Super-Resolution
pip install -r requirements.txt
```

### Run the Notebook

```bash
jupyter notebook ENNCA2026_project_342050.ipynb
```

> Make sure the DIV2K dataset is downloaded and placed in the expected directory structure before running training cells.

---

## Project Structure

```
ENNCA2026-Super-Resolution/
├── ENNCA2026_project_342050.ipynb   # Main notebook (full pipeline)
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## Key Findings

- **QAT outperforms PTQ**: Quantization-aware training recovers nearly all accuracy lost in post-training quantization.
- **Pruning is robust**: The model retains strong performance (23.81 dB) even at 50% sparsity.
- **Layer sensitivity varies**: Pruning the 3 least-sensitive layers actually improves PSNR slightly (24.52 dB), showing redundancy in certain layers.
- **YUV decomposition**: Training separately on the Y (luminance) channel and using bicubic for U/V reduces compute without sacrificing perceptual quality.

---

## Author

**Mahmood Alduhoun**  
Master's Student in Computer Science — Warsaw University of Technology  
[LinkedIn](https://www.linkedin.com/in/mahmood-alduhoun) | [GitHub](https://github.com/Mahmood-Alduhoun)

---

## License

MIT License

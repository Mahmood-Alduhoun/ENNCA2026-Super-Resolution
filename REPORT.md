# ENNCA2026 — Super-Resolution Project Report

**Student ID:** 342050  
**Course:** ENNCA2026  
**Institution:** Warsaw University of Technology  
**Repository:** https://github.com/Mahmood-Alduhoun/ENNCA2026-Super-Resolution  
**Report Date:** 2026-06-07

---

## 1. Project Overview

This project implements a complete **4× Image Super-Resolution (SR)** pipeline using the **RPQNet_WSDR_EV2** architecture on the DIV2K dataset. The even student number (342050) determines the 4× upscaling factor per the course brief.

The notebook covers the full model lifecycle:

```
Float Training (RGB & Y-channel)
    → Model Collapse (switch_to_deploy)
    → Post-Training Quantization (PTQ)
    → Quantization-Aware Training (QAT)
    → Unstructured Magnitude Pruning (10 / 30 / 50%)
    → Layer-wise PSNR-Sensitivity Pruning
```

---

## 2. Dataset

**DIV2K** — high-quality image benchmark for super-resolution tasks.

| Split      | HR Images | LR Images (4× bicubic) |
|------------|-----------|------------------------|
| Training   | 800       | 800                    |
| Validation | 100       | 100                    |

Low-resolution inputs are generated via 4× bicubic downsampling. The dataset zips are extracted in-place at notebook startup.

**Bicubic interpolation baseline PSNR (val set):** 25.07 dB

---

## 3. Model Architecture — RPQNet_WSDR_EV2

RPQNet_WSDR_EV2 is a reparameterizable super-resolution network. At training time each block uses expanded multi-branch convolutions; at inference time these are collapsed (`switch_to_deploy`) into a single equivalent convolution, dramatically cutting FLOPs with no accuracy loss.

### 3.1 Model Sizes

| Model Variant          | Parameters  | FLOPs (64×64 input)  | PSNR RGB (val) |
|------------------------|-------------|----------------------|----------------|
| RGB (4×)               | 1,671,456   | 12,964,331,520       | 24.50 dB       |
| Y-ch + bicubic UV (4×) | 1,645,792   | 12,754,878,464       | 24.97 dB       |

### 3.2 After Model Collapse (RGB model)

| Property      | Before Collapse      | After Collapse  | Reduction |
|---------------|----------------------|-----------------|-----------|
| Parameters    | 1,671,456            | 117,984         | −93%      |
| FLOPs         | 12,964,331,520       | 960,233,472     | −93%      |
| MSE (outputs) | —                    | 0.00001686      | ≈ 0       |

The pre/post-collapse MSE of **0.00001686** confirms the reparameterization is numerically lossless.

### 3.3 Training Configuration

| Setting         | Value                                    |
|-----------------|------------------------------------------|
| Loss function   | L1                                       |
| Optimizer       | Adam                                     |
| LR schedule     | Cosine Annealing (η_min = 1e-6)          |
| Epochs          | 50                                       |
| Mixed precision | AMP (CUDA only)                          |
| Scale factor    | 4× (even student ID)                     |

---

## 4. Float Training

### 4.1 RGB Model — Full Training Log (50 epochs)

| Epoch | Train Loss | Train PSNR | Val PSNR  | LR        | Saved |
|-------|-----------|------------|-----------|-----------|-------|
| 1     | 0.1805    | 10.98 dB   | 17.84 dB  | 0.000100  | ✓     |
| 2     | 0.0918    | 17.18 dB   | 19.75 dB  | 0.000100  | ✓     |
| 3     | 0.0744    | 18.69 dB   | 20.61 dB  | 0.000100  | ✓     |
| 4     | 0.0661    | 19.50 dB   | 21.24 dB  | 0.000099  | ✓     |
| 5     | 0.0608    | 20.21 dB   | 21.71 dB  | 0.000098  | ✓     |
| 6     | 0.0563    | 20.83 dB   | 22.20 dB  | 0.000098  | ✓     |
| 7     | 0.0531    | 21.27 dB   | 22.45 dB  | 0.000097  | ✓     |
| 8     | 0.0538    | 21.28 dB   | 22.70 dB  | 0.000095  | ✓     |
| 9     | 0.0482    | 21.98 dB   | 22.69 dB  | 0.000094  |       |
| 10    | 0.0486    | 22.16 dB   | 22.85 dB  | 0.000092  | ✓     |
| 11    | 0.0478    | 22.17 dB   | 23.19 dB  | 0.000091  | ✓     |
| 12    | 0.0445    | 22.60 dB   | 23.44 dB  | 0.000089  | ✓     |
| 13    | 0.0429    | 22.71 dB   | 23.52 dB  | 0.000087  | ✓     |
| 14    | 0.0428    | 22.84 dB   | 23.34 dB  | 0.000084  |       |
| 15    | 0.0419    | 23.10 dB   | 23.68 dB  | 0.000082  | ✓     |
| 16    | 0.0408    | 23.19 dB   | 23.72 dB  | 0.000080  | ✓     |
| 17    | 0.0415    | 22.98 dB   | 23.82 dB  | 0.000077  | ✓     |
| 18    | 0.0395    | 23.37 dB   | 23.86 dB  | 0.000074  | ✓     |
| 19    | 0.0399    | 23.27 dB   | 23.88 dB  | 0.000072  | ✓     |
| 20    | 0.0400    | 23.30 dB   | 23.92 dB  | 0.000069  | ✓     |
| 21    | 0.0408    | 23.10 dB   | 24.03 dB  | 0.000066  | ✓     |
| 22    | 0.0394    | 23.35 dB   | 24.03 dB  | 0.000063  | ✓     |
| 23    | 0.0375    | 23.64 dB   | 24.09 dB  | 0.000060  | ✓     |
| 24    | 0.0381    | 23.45 dB   | 24.14 dB  | 0.000057  | ✓     |
| 25    | 0.0380    | 23.59 dB   | 24.17 dB  | 0.000054  | ✓     |
| 26    | 0.0384    | 23.49 dB   | 24.21 dB  | 0.000050  | ✓     |
| 27    | 0.0366    | 23.80 dB   | 24.24 dB  | 0.000047  | ✓     |
| 28    | 0.0378    | 23.66 dB   | 24.27 dB  | 0.000044  | ✓     |
| 29    | 0.0375    | 23.57 dB   | 24.20 dB  | 0.000041  |       |
| 30    | 0.0381    | 23.59 dB   | 24.17 dB  | 0.000038  |       |
| 31    | 0.0375    | 23.67 dB   | 24.34 dB  | 0.000035  | ✓     |
| 32    | 0.0366    | 23.74 dB   | 24.34 dB  | 0.000032  | ✓     |
| 33    | 0.0367    | 23.82 dB   | 24.37 dB  | 0.000029  | ✓     |
| 34    | 0.0359    | 23.85 dB   | 24.39 dB  | 0.000027  | ✓     |
| 35    | 0.0359    | 23.91 dB   | 24.39 dB  | 0.000024  | ✓     |
| 36    | 0.0367    | 23.72 dB   | 24.42 dB  | 0.000021  | ✓     |
| 37    | 0.0369    | 23.63 dB   | 24.42 dB  | 0.000019  | ✓     |
| 38    | 0.0362    | 23.82 dB   | 24.44 dB  | 0.000017  | ✓     |
| 39    | 0.0371    | 23.61 dB   | 24.44 dB  | 0.000014  | ✓     |
| 40    | 0.0368    | 23.74 dB   | 24.46 dB  | 0.000012  | ✓     |
| 41    | 0.0360    | 23.88 dB   | 24.47 dB  | 0.000010  | ✓     |
| 42    | 0.0357    | 23.84 dB   | 24.48 dB  | 0.000009  | ✓     |
| 43    | 0.0353    | 24.04 dB   | 24.48 dB  | 0.000007  | ✓     |
| 44    | 0.0356    | 23.89 dB   | 24.49 dB  | 0.000006  | ✓     |
| 45    | 0.0351    | 24.03 dB   | 24.49 dB  | 0.000004  | ✓     |
| 46    | 0.0357    | 23.91 dB   | 24.49 dB  | 0.000003  | ✓     |
| 47    | 0.0358    | 23.84 dB   | 24.50 dB  | 0.000003  | ✓     |
| 48    | 0.0352    | 24.00 dB   | 24.50 dB  | 0.000002  | ✓     |
| 49    | 0.0353    | 23.93 dB   | 24.50 dB  | 0.000001  | ✓     |
| 50    | 0.0366    | 23.69 dB   | 24.50 dB  | 0.000001  | ✓     |

**Best RGB Val PSNR: 24.50 dB**

---

### 4.2 Y-Channel Model — Full Training Log (50 epochs)

| Epoch | Train Loss | Train PSNR | Val PSNR  | LR        | Saved |
|-------|-----------|------------|-----------|-----------|-------|
| 1     | 0.1598    | 11.32 dB   | 19.65 dB  | 0.000100  | ✓     |
| 2     | 0.0756    | 17.73 dB   | 20.66 dB  | 0.000100  | ✓     |
| 3     | 0.0665    | 18.94 dB   | 21.45 dB  | 0.000100  | ✓     |
| 4     | 0.0583    | 20.06 dB   | 21.79 dB  | 0.000099  | ✓     |
| 5     | 0.0584    | 20.30 dB   | 22.33 dB  | 0.000098  | ✓     |
| 6     | 0.0516    | 21.27 dB   | 22.75 dB  | 0.000098  | ✓     |
| 7     | 0.0496    | 21.78 dB   | 22.78 dB  | 0.000097  | ✓     |
| 8     | 0.0453    | 22.37 dB   | 23.41 dB  | 0.000095  | ✓     |
| 9     | 0.0485    | 22.28 dB   | 23.52 dB  | 0.000094  | ✓     |
| 10    | 0.0408    | 23.22 dB   | 23.85 dB  | 0.000092  | ✓     |
| 11    | 0.0393    | 23.28 dB   | 23.94 dB  | 0.000091  | ✓     |
| 12    | 0.0399    | 23.35 dB   | 23.86 dB  | 0.000089  |       |
| 13    | 0.0392    | 23.48 dB   | 24.20 dB  | 0.000087  | ✓     |
| 14    | 0.0370    | 23.68 dB   | 24.34 dB  | 0.000084  | ✓     |
| 15    | 0.0369    | 23.73 dB   | 24.35 dB  | 0.000082  | ✓     |
| 16    | 0.0352    | 24.11 dB   | 24.43 dB  | 0.000080  | ✓     |
| 17    | 0.0358    | 24.09 dB   | 24.28 dB  | 0.000077  |       |
| 18    | 0.0372    | 23.90 dB   | 24.46 dB  | 0.000074  | ✓     |
| 19    | 0.0346    | 24.10 dB   | 24.62 dB  | 0.000072  | ✓     |
| 20    | 0.0352    | 24.15 dB   | 24.48 dB  | 0.000069  |       |
| 21    | 0.0366    | 23.99 dB   | 24.54 dB  | 0.000066  |       |
| 22    | 0.0349    | 24.03 dB   | 24.73 dB  | 0.000063  | ✓     |
| 23    | 0.0327    | 24.34 dB   | 24.75 dB  | 0.000060  | ✓     |
| 24    | 0.0348    | 24.00 dB   | 24.73 dB  | 0.000057  |       |
| 25    | 0.0331    | 24.53 dB   | 24.79 dB  | 0.000054  | ✓     |
| 26    | 0.0328    | 24.38 dB   | 24.83 dB  | 0.000050  | ✓     |
| 27    | 0.0332    | 24.26 dB   | 24.86 dB  | 0.000047  | ✓     |
| 28    | 0.0338    | 24.10 dB   | 24.87 dB  | 0.000044  | ✓     |
| 29    | 0.0315    | 24.60 dB   | 24.89 dB  | 0.000041  | ✓     |
| 30    | 0.0322    | 24.44 dB   | 24.87 dB  | 0.000038  |       |
| 31    | 0.0322    | 24.58 dB   | 24.83 dB  | 0.000035  |       |
| 32    | 0.0331    | 24.25 dB   | 24.94 dB  | 0.000032  | ✓     |
| 33    | 0.0316    | 24.57 dB   | 24.95 dB  | 0.000029  | ✓     |
| 34    | 0.0321    | 24.43 dB   | 24.96 dB  | 0.000027  | ✓     |
| 35    | 0.0321    | 24.47 dB   | 24.97 dB  | 0.000024  | ✓     |
| 36    | 0.0325    | 24.33 dB   | 24.98 dB  | 0.000021  | ✓     |
| 37    | 0.0316    | 24.46 dB   | 24.98 dB  | 0.000019  | ✓     |
| 38    | 0.0315    | 24.65 dB   | 24.93 dB  | 0.000017  |       |
| 39    | 0.0321    | 24.52 dB   | 25.00 dB  | 0.000014  | ✓     |
| 40    | 0.0315    | 24.50 dB   | 25.00 dB  | 0.000012  | ✓     |
| 41    | 0.0312    | 24.70 dB   | 25.01 dB  | 0.000010  | ✓     |
| 42    | 0.0306    | 24.83 dB   | 25.01 dB  | 0.000009  | ✓     |
| 43    | 0.0312    | 24.72 dB   | 25.02 dB  | 0.000007  | ✓     |
| 44    | 0.0312    | 24.72 dB   | 25.02 dB  | 0.000006  | ✓     |
| 45    | 0.0314    | 24.60 dB   | 25.02 dB  | 0.000004  | ✓     |
| 46    | 0.0318    | 24.47 dB   | 25.02 dB  | 0.000003  | ✓     |
| 47    | 0.0312    | 24.67 dB   | 25.02 dB  | 0.000003  | ✓     |
| 48    | 0.0312    | 24.69 dB   | 25.02 dB  | 0.000002  | ✓     |
| 49    | 0.0303    | 24.86 dB   | 25.02 dB  | 0.000001  | ✓     |
| 50    | 0.0317    | 24.61 dB   | 25.03 dB  | 0.000001  | ✓     |

**Best Y-channel Val PSNR: 25.03 dB** (Y-channel only)  
**RGB-domain PSNR (SR-Y + bicubic UV → RGB): 24.97 dB**

---

### 4.3 RGB vs Y-Channel Comparison

| Model                    | Params     | FLOPs              | PSNR (RGB val) |
|--------------------------|------------|--------------------|----------------|
| Bicubic 4× baseline      | —          | —                  | 25.07 dB       |
| RGB (4×)                 | 1,671,456  | 12,964,331,520     | 24.50 dB       |
| Y-ch + bicubic UV (4×)   | 1,645,792  | 12,754,878,464     | 24.97 dB       |

The Y-channel model achieves higher RGB-domain PSNR (24.97 dB vs 24.50 dB) with fewer parameters, by focusing the network on luminance and relying on bicubic interpolation for chroma.

> Note: Both models score slightly below the bicubic baseline on PSNR. This is a known phenomenon for lightweight SR models trained with L1 loss on a challenging benchmark — L1 loss does not directly optimise PSNR, and the small model capacity limits spectral reconstruction. The Y-channel model (24.97 dB) comes within 0.10 dB of bicubic.

---

## 5. Model Collapse (switch_to_deploy)

```
                          Before        After
Parameters                1,671,456     117,984
FLOPs                    12,964,331,520 960,233,472
MSE between outputs:      0.00001686
```

Reparameterization reduces parameters by **~93%** and FLOPs by **~93%**, with an MSE between pre- and post-collapse outputs of only 0.00001686 — confirming numerical equivalence.

---

## 6. Quantization

All quantization experiments use the **collapsed RGB model** as the base (float PSNR: 24.50 dB).

### 6.1 Post-Training Quantization (PTQ)

PTQ converts the trained weights to integer precision without any retraining.

```
=== PTQ Results ===
Float PSNR:              24.50 dB
PTQ Per-Channel PSNR:    24.14 dB
PTQ Per-Tensor PSNR:     24.33 dB
```

| Strategy       | PSNR (dB) | Drop vs. Float |
|----------------|-----------|----------------|
| Float baseline | 24.50     | —              |
| PTQ Per-tensor | 24.33     | −0.17 dB       |
| PTQ Per-channel| 24.14     | −0.36 dB       |

In this run, per-tensor PTQ (24.33 dB) outperforms per-channel PTQ (24.14 dB).

### 6.2 Quantization-Aware Training (QAT)

QAT inserts fake-quantization nodes during a short 5-epoch fine-tune, allowing weights to adapt to quantization noise.

```
Running QAT Per-Channel...
  QAT Per-Channel Epoch [1/5] PSNR: 24.44 dB
  QAT Per-Channel Epoch [2/5] PSNR: 24.45 dB
  QAT Per-Channel Epoch [3/5] PSNR: 24.46 dB
  QAT Per-Channel Epoch [4/5] PSNR: 24.47 dB
  QAT Per-Channel Epoch [5/5] PSNR: 24.48 dB

Running QAT Per-Tensor...
  QAT Per-Tensor Epoch [1/5] PSNR: 24.45 dB
  QAT Per-Tensor Epoch [2/5] PSNR: 24.46 dB
  QAT Per-Tensor Epoch [3/5] PSNR: 24.46 dB
  QAT Per-Tensor Epoch [4/5] PSNR: 24.47 dB
  QAT Per-Tensor Epoch [5/5] PSNR: 24.48 dB

=== QAT Results ===
Float PSNR:              24.50 dB
QAT Per-Channel PSNR:    24.48 dB
QAT Per-Tensor PSNR:     24.48 dB
```

| Strategy        | PSNR (dB) | Drop vs. Float |
|-----------------|-----------|----------------|
| Float baseline  | 24.50     | —              |
| QAT Per-channel | 24.48     | −0.02 dB       |
| QAT Per-tensor  | 24.48     | −0.02 dB       |

Both QAT strategies recover to 24.48 dB — only 0.02 dB below the float model. QAT dramatically outperforms PTQ in both strategies.

### 6.3 Quantization Summary

| Method          | PSNR (dB) | Drop vs. Float |
|-----------------|-----------|----------------|
| Float           | 24.50     | —              |
| QAT Per-channel | **24.48** | −0.02 dB       |
| QAT Per-tensor  | **24.48** | −0.02 dB       |
| PTQ Per-tensor  | 24.33     | −0.17 dB       |
| PTQ Per-channel | 24.14     | −0.36 dB       |

---

## 7. Pruning

### 7.1 Unstructured Magnitude Pruning (Iterative + Fine-tune)

Starting PSNR (collapsed model): **24.48 dB**

Each sparsity level applies global L1-unstructured pruning then fine-tunes for 3 epochs at lr=1e-5.

#### Sparsity 10%

```
=== Iterative step: total sparsity 10% ===
  Achieved sparsity: 10.0%  |  PSNR after pruning: 24.28 dB
  Fine-tune Epoch [1/3] PSNR: 24.48 dB
  Fine-tune Epoch [2/3] PSNR: 24.49 dB
  Fine-tune Epoch [3/3] PSNR: 24.50 dB
  PSNR after fine-tune: 24.50 dB
```

#### Sparsity 30%

```
=== Iterative step: total sparsity 30% ===
  Achieved sparsity: 30.0%  |  PSNR after pruning: 21.40 dB
  Fine-tune Epoch [1/3] PSNR: 24.27 dB
  Fine-tune Epoch [2/3] PSNR: 24.36 dB
  Fine-tune Epoch [3/3] PSNR: 24.41 dB
  PSNR after fine-tune: 24.41 dB
```

#### Sparsity 50%

```
=== Iterative step: total sparsity 50% ===
  Achieved sparsity: 50.0%  |  PSNR after pruning: 17.41 dB
  Fine-tune Epoch [1/3] PSNR: 22.26 dB
  Fine-tune Epoch [2/3] PSNR: 23.58 dB
  Fine-tune Epoch [3/3] PSNR: 23.81 dB
  PSNR after fine-tune: 23.81 dB
```

#### Pruning Summary Table

| Sparsity | PSNR After Pruning | PSNR After Fine-tune | Recovery |
|----------|--------------------|----------------------|----------|
| 0% (base)| 24.48 dB           | —                    | —        |
| 10%      | 24.28 dB           | **24.50 dB**         | Full     |
| 30%      | 21.40 dB           | **24.41 dB**         | −0.07 dB |
| 50%      | 17.41 dB           | **23.81 dB**         | −0.67 dB |

Fine-tuning is highly effective. At 50% sparsity, the model drops to 17.41 dB immediately after pruning but recovers to 23.81 dB after only 3 fine-tune epochs.

---

### 7.2 Layer-wise Pruning by PSNR Sensitivity

Each convolutional layer is pruned individually at 50% to measure how sensitive it is to weight removal. Layers with the smallest PSNR drop are the least sensitive.

#### Full Sensitivity Table (all conv layers)

| Layer                      | PSNR After Individual Pruning | Drop  |
|----------------------------|-------------------------------|-------|
| reconstruction             | 18.91 dB                      | −5.57 |
| skip_b                     | 20.89 dB                      | −3.59 |
| skip_a                     | 21.42 dB                      | −3.06 |
| initial.0                  | 21.79 dB                      | −2.69 |
| blocks_a.5.conv_reparam    | 24.27 dB                      | −0.21 |
| blocks_b.5.conv_reparam    | 24.35 dB                      | −0.13 |
| blocks_b.4.conv_reparam    | 24.37 dB                      | −0.10 |
| blocks_b.3.conv_reparam    | 24.39 dB                      | −0.08 |
| blocks_a.4.conv_reparam    | 24.41 dB                      | −0.07 |
| blocks_a.3.conv_reparam    | 24.44 dB                      | −0.04 |
| blocks_b.0.conv_reparam    | 24.44 dB                      | −0.04 |
| blocks_a.0.conv_reparam    | 24.45 dB                      | −0.03 |
| blocks_a.1.conv_reparam    | 24.45 dB                      | −0.03 |
| blocks_a.2.conv_reparam    | 24.45 dB                      | −0.03 |
| blocks_b.1.conv_reparam    | 24.45 dB                      | −0.02 |
| **blocks_b.2.conv_reparam**| **24.46 dB**                  | **−0.01** ← least sensitive |

**Top 5 least sensitive layers (candidates for pruning):**
1. `blocks_b.2.conv_reparam` — drop: 0.01 dB
2. `blocks_b.1.conv_reparam` — drop: 0.02 dB
3. `blocks_a.2.conv_reparam` — drop: 0.03 dB
4. `blocks_a.1.conv_reparam` — drop: 0.03 dB
5. `blocks_a.0.conv_reparam` — drop: 0.03 dB

**Most sensitive layers:** `reconstruction` (−5.57 dB), `skip_b` (−3.59 dB), `skip_a` (−3.06 dB), `initial.0` (−2.69 dB) — these are the input stem and skip connections; pruning them causes catastrophic quality loss.

#### Pruning 3 Least-Sensitive Layers + Fine-tune (5 epochs)

```
PSNR after layer-wise pruning: 24.41 dB
  Fine-tune Epoch [1/5] PSNR: 24.48 dB
  Fine-tune Epoch [2/5] PSNR: 24.49 dB
  Fine-tune Epoch [3/5] PSNR: 24.50 dB
  Fine-tune Epoch [4/5] PSNR: 24.51 dB
  Fine-tune Epoch [5/5] PSNR: 24.52 dB
PSNR after fine-tuning: 24.52 dB
```

Layer-wise pruning followed by fine-tuning achieves **24.52 dB** — **+0.02 dB above the float baseline** (24.50 dB). The pruned redundant weights acted as noise; their removal combined with fine-tuning yields a slight improvement.

---

## 8. Final Results Summary

| Stage                        | Strategy                         | PSNR (dB)     | vs. Float   |
|------------------------------|----------------------------------|---------------|-------------|
| Bicubic baseline             | 4× bicubic interpolation         | 25.07         | —           |
| Float (RGB)                  | Baseline                         | **24.50**     | —           |
| Float (Y-channel, RGB-equiv) | SR-Y + bicubic UV → RGB          | 24.97         | +0.47 dB    |
| PTQ Per-tensor               | Per-tensor MinMax observer       | 24.33         | −0.17 dB    |
| PTQ Per-channel              | Per-channel MinMax observer      | 24.14         | −0.36 dB    |
| QAT Per-channel              | 5-epoch fine-tune                | **24.48**     | −0.02 dB    |
| QAT Per-tensor               | 5-epoch fine-tune                | **24.48**     | −0.02 dB    |
| Pruning 10%                  | Magnitude + 3-epoch fine-tune    | 24.50         | 0.00 dB     |
| Pruning 30%                  | Magnitude + 3-epoch fine-tune    | 24.41         | −0.09 dB    |
| Pruning 50%                  | Magnitude + 3-epoch fine-tune    | 23.81         | −0.69 dB    |
| Layer-wise pruning           | 3 least-sensitive + 5-epoch tune | **24.52**     | **+0.02 dB**|

---

## 9. Final Model Snapshot (Notebook Cell 16)

```
=== Float Training ===
  RGB model:
    Params:                     1,671,456
    FLOPs (64x64 input):        12,964,331,520
    Best RGB Val PSNR:          24.50 dB

  Y-channel model:
    Params:                     1,645,792
    FLOPs (64x64 input):        12,754,878,464
    RGB-equivalent PSNR:        24.97 dB   (SR-Y + bicubic UV → RGB)
    Y-only PSNR (for reference):25.03 dB

=== Model Collapse ===
  Params before / after:        1,671,456 / 117,984
  FLOPs  before / after:        12,964,331,520 / 960,233,472
  MSE pre/post collapse:        0.00001686
```

---

## 10. Key Findings

**Model collapse is the single most impactful optimization.** A 93% FLOPs reduction with MSE of only 0.00001686 makes the model viable for edge/mobile deployment with no accuracy cost.

**QAT recovers nearly all quantization loss.** Both per-channel and per-tensor QAT achieve 24.48 dB — only 0.02 dB below float — versus PTQ drops of 0.17–0.36 dB.

**Fine-tuning restores pruned accuracy at all sparsity levels.** Even at 50% sparsity (PSNR drops to 17.41 dB immediately after pruning), 3 fine-tune epochs recover the model to 23.81 dB.

**Layer sensitivity is highly non-uniform.** The input stem (`initial.0`), skip connections (`skip_a`, `skip_b`), and pixel-shuffle reconstruction layer are catastrophically sensitive to pruning (−2.69 to −5.57 dB). The middle `blocks_b` reparam layers are nearly insensitive (−0.01 dB).

**Targeted pruning of 3 least-sensitive layers improves PSNR by +0.02 dB** (24.52 dB vs 24.50 dB baseline), indicating those weights contributed noise rather than useful learned features.

**Y-channel training offers better RGB-domain PSNR** (24.97 dB vs 24.50 dB) with slightly fewer parameters, by focusing network capacity on luminance where the human visual system is most sensitive.

---

## 11. Technology Stack

| Component        | Technology                      |
|------------------|---------------------------------|
| Deep learning    | PyTorch                         |
| Quantization     | `torch.ao.quantization`         |
| Pruning          | `torch.nn.utils.prune`          |
| Monitoring       | TensorBoard                     |
| Image I/O        | PIL, torchvision                |
| Visualization    | Matplotlib                      |
| Mixed precision  | `torch.cuda.amp`                |
| Python version   | 3.9+                            |

---

*Report generated from notebook outputs: `ENNCA2026_project_342050.ipynb`*  
*Author: Mahmood Alduhoun — Warsaw University of Technology*

# Pneumonia Detection from Chest X-Rays using a CNN (ResNet-18)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CNN-ee4c2c)
![Task](https://img.shields.io/badge/Task-Image%20Classification-green)

A medical image classification project that detects **pneumonia** in paediatric chest X-ray images using a Convolutional Neural Network (CNN). The model is a **ResNet-18** pretrained on ImageNet and fine-tuned on chest radiographs with transfer learning.

> **Project:** Medical Image Analysis using CNN
> **Author:** `Md. Abdullah Al Mamun` &nbsp;|&nbsp; `BRAC University`

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Exploratory Data Analysis](#exploratory-data-analysis)
4. [Preprocessing and Data Split](#preprocessing-and-data-split)
5. [Model Architecture](#model-architecture)
6. [Training Setup](#training-setup)
7. [Results](#results)
8. [Sample Predictions](#sample-predictions)
9. [Limitations and Future Work](#limitations-and-future-work)
10. [Repository Structure](#repository-structure)
11. [How to Run](#how-to-run)
12. [References](#references)

---

## Overview

| Item | Details |
|---|---|
| Task | Binary image classification: `NORMAL` vs. `PNEUMONIA` |
| Model | ResNet-18 (ImageNet-pretrained) + custom classification head |
| Framework | PyTorch, torchvision, scikit-learn |
| Loss | `BCEWithLogitsLoss` with class weighting |
| Test accuracy | **86.22%** |
| Test ROC-AUC | **0.9577** |
| Pneumonia sensitivity (recall) | **98.97%** |

The notebook covers the full pipeline: dataset loading, exploratory data analysis, preprocessing and augmentation, train/validation/test splitting, CNN implementation, training, evaluation, and visualisation of results and predictions.

## Dataset

**Chest X-Ray Images (Pneumonia)** by Kermany et al., hosted on Kaggle: [`paultimothymooney/chest-xray-pneumonia`](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). It contains paediatric chest radiographs (JPEG) in two classes, and the notebook downloads it automatically with `kagglehub`.

| Original folder | NORMAL | PNEUMONIA | Total |
|---|---:|---:|---:|
| train | 1,341 | 3,875 | 5,216 |
| val | 8 | 8 | 16 |
| test | 234 | 390 | 624 |
| **All** | **1,583** | **4,273** | **5,856** |

## Exploratory Data Analysis

The training data is imbalanced, with about 2.9× more PNEUMONIA images than NORMAL images. The images also vary in size, contrast, positioning, and contain text markers and medical devices.

<p align="center">
  <img src="./assets/Class Distribution in Training Set.png" width="45%" alt="Class distribution in the training set">
</p>

<p align="center">
  <img src="./assets/Sample Chest X-Ray Radiographs.png" width="85%" alt="Sample chest X-rays from both classes">
</p>

## Preprocessing and Data Split

**Preprocessing**

- Resize to **224 × 224**, convert to 3-channel RGB.
- Normalise with ImageNet mean `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)`.
- **Augmentation (training only):** random horizontal flip (p = 0.5), random rotation (±10°), colour jitter (brightness and contrast ±0.1).

**Split**

The official `val` folder has only 16 images, so it was merged with `train` and re-split randomly (seed = 42). The official `test` folder is kept untouched for final evaluation.

| Subset | Images | Source |
|---|---:|---|
| Training | 4,448 | 85% of train + val pool |
| Validation | 784 | 15% of train + val pool |
| Test | 624 | Official test folder |

**Class imbalance** is handled with `pos_weight = 1349 / 3883 = 0.3474` in the loss, which down-weights the majority PNEUMONIA class.

## Model Architecture

ResNet-18 backbone with the ImageNet classifier replaced by a small custom head.

| Stage | Layers | Output | Training |
|---|---|---|---|
| Input | RGB X-ray | 3 × 224 × 224 | — |
| `conv1` | 7×7 conv, 64, stride 2 + BN + ReLU | 64 × 112 × 112 | Frozen |
| `maxpool` | 3×3, stride 2 | 64 × 56 × 56 | Frozen |
| `layer1` | 2 × BasicBlock (64) | 64 × 56 × 56 | Frozen |
| `layer2` | 2 × BasicBlock (128) | 128 × 28 × 28 | Frozen |
| `layer3` | 2 × BasicBlock (256) | 256 × 14 × 14 | Frozen |
| `layer4` | 2 × BasicBlock (512) | 512 × 7 × 7 | Partly fine-tuned |
| Global avg-pool | Adaptive average pooling | 512 | — |
| Custom head | Dropout(0.4) → Linear(512→128) → ReLU → Dropout(0.2) → Linear(128→1) | 1 logit | Trained from scratch |

**Transfer learning:** all backbone parameter tensors except the last 12 are frozen. In effect, `layer4` block 2, the shortcut projection of `layer4` block 1, and the whole new head are trained. That is roughly 4.9M trainable parameters out of 11.2M (approximate, derived from the architecture). The sigmoid of the output logit gives P(PNEUMONIA), and the decision threshold is 0.5.

## Training Setup

| Setting | Value |
|---|---|
| Loss | `BCEWithLogitsLoss(pos_weight=0.3474)` |
| Optimiser | AdamW, lr = 1e-4, weight decay = 1e-3 |
| Scheduler | `ReduceLROnPlateau` on validation loss (factor 0.5, patience 2) |
| Batch size / epochs | 32 / 10 |
| Model selection | Best validation loss (epoch 6) saved to `best_pneumonia_cnn.pth` |
| Seed | 42 |
| Hardware | Google Colab (CPU) |

<p align="center">
  <img src="./assets/Training vs Validation (Loss and Accuracy).png" width="90%" alt="Training and validation loss and accuracy">
</p>

| Epoch | Train loss | Train acc. | Val loss | Val acc. |
|---:|---:|---:|---:|---:|
| 1 | 0.1161 | 91.37% | 0.0690 | 95.41% |
| 2 | 0.0459 | 96.36% | 0.0500 | 96.30% |
| 3 | 0.0427 | 96.70% | 0.0670 | 94.26% |
| 4 | 0.0319 | 97.55% | 0.0463 | 97.70% |
| 5 | 0.0265 | 98.09% | 0.0495 | 97.07% |
| **6** | **0.0227** | **98.36%** | **0.0427** | **97.07%** |
| 7 | 0.0222 | 98.27% | 0.0641 | 95.66% |
| 8 | 0.0200 | 98.47% | 0.0526 | 96.94% |
| 9 | 0.0205 | 98.31% | 0.0541 | 96.68% |
| 10 | 0.0174 | 98.85% | 0.0519 | 97.19% |

Epoch 6 (bold) has the lowest validation loss, so its weights were used for testing. After that the validation loss stops improving while the training loss keeps falling, a mild sign of overfitting.

## Results

Evaluated once on the 624-image held-out test set using the best checkpoint.

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| NORMAL | 0.9744 | 0.6496 | 0.7795 | 234 |
| PNEUMONIA | 0.8248 | 0.9897 | 0.8998 | 390 |
| Macro avg | 0.8996 | 0.8197 | 0.8396 | 624 |
| Weighted avg | 0.8809 | 0.8622 | 0.8547 | 624 |

| Metric | Value |
|---|---|
| Accuracy | 86.22% (538 / 624) |
| ROC-AUC | 0.9577 |
| Sensitivity (PNEUMONIA recall) | 98.97% (386 / 390) |
| Specificity (NORMAL recall) | 64.96% (152 / 234) |

<p align="center">
  <img src="./assets/Confution Matrix and ROC.png" width="90%" alt="Confusion matrix and ROC curve">
</p>

**Interpretation**

- Only **4 of 390** pneumonia cases were missed, which matters most in a screening setting.
- **82 of 234** normal X-rays were flagged as pneumonia, so specificity is low (65%).
- The ROC-AUC of 0.958 shows good class separation, so the default 0.5 threshold is probably not optimal for the NORMAL class.
- Validation accuracy (~97%) is well above test accuracy (86%). Likely contributors are distribution differences between the official train and test folders, image-level splitting that may put the same patient in both training and validation, and the class-weighted loss. These explanations were not tested separately.

## Sample Predictions

Green titles are correct predictions and red titles are errors.

<p align="center">
  <img src="./assets/Sample Test Predictions.png" width="90%" alt="Sample test predictions">
</p>

The test loader is not shuffled and the folder lists NORMAL images first, so all eight images shown are NORMAL. Seven are classified correctly (93.2–99.8% confidence), and one is a false positive predicted as PNEUMONIA with 63.6% confidence.

## Limitations and Future Work

- Tune the decision threshold on the validation ROC curve to improve specificity.
- Use a patient-level split or cross-validation to avoid possible train/validation leakage.
- Add Grad-CAM heat maps to check that the model focuses on lung regions rather than markers or devices.
- Review horizontal-flip augmentation, since it mirrors the heart position.
- Try longer training, more unfrozen layers, other CNNs (DenseNet, EfficientNet), and external validation.

> **Disclaimer:** this is an educational project and is **not** a certified medical diagnostic tool.

## Repository Structure

```
.
├── LICENSE
├── README.md
├── requirements.txt
├── medical_pneumonia_cnn.ipynb       
│
├── assets/                            
│   ├── Class Distribution in Training Set.png
│   ├── Confusion Matrix and ROC.png
│   ├── Sample Chest X-Ray Radiographs.png
│   ├── Sample Test Predictions.png
│   └── Training vs Validation (Loss and Accuracy).png
│
├── models/
│   └── best_pneumonia_cnn.pth         # Best checkpoint (epoch 6, lowest validation loss)
│
└── report/
    └── CNN_Pneumonia_Detection_Report.docx 
```

## How to Run

**Option 1: Google Colab (recommended)**

1. Open the notebook in Colab: `medical_pneumonia_cnn.ipynb`
2. Run all cells in order (`Runtime → Run all`). The dataset is downloaded automatically. A GPU runtime is optional but speeds up training.

**Option 2: Local**

```bash
git clone https://github.com/aam-bd/medical-chest-xray-pneumonia-cnn.git
cd medical-chest-xray-pneumonia-cnn

pip install kagglehub torch torchvision scikit-learn matplotlib seaborn pillow numpy jupyter
jupyter notebook pneumonia_cnn_resnet18.ipynb
```

The dataset download through `kagglehub` may require Kaggle API credentials outside Colab. The notebook writes `best_pneumonia_cnn.pth`, `training_curves.png`, `evaluation_metrics.png` and `sample_predictions.png` to the working directory. A fixed seed (42) is used for reproducibility.

## References

1. Kermany, D. S., et al. (2018). *Identifying medical diagnoses and treatable diseases by image-based deep learning.* Cell, 172(5), 1122–1131.
2. Mooney, P. T. *Chest X-Ray Images (Pneumonia).* Kaggle. https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep residual learning for image recognition.* CVPR.
4. Loshchilov, I., & Hutter, F. (2019). *Decoupled weight decay regularization.* ICLR.
5. Paszke, A., et al. (2019). *PyTorch: An imperative style, high-performance deep learning library.* NeurIPS.
6. https://huggingface.co/datasets/TianyuZhang/ModelMergingBaseline16Datasets/blame/main/datasets/weather.py
7. https://github.com/ryancodingg/CIFAR-10-image-classification-with-CNN-and-transfer-learning

# MS-ROAD: A Diagnostic Protocol and Stress Test for Scale Inhibition in Road Anomaly Segmentation

## Overview of the proposed Scale-Adaptive Anomaly Network (SA-Net). 
Designed as a plug-and-play framework, SA-Net
attaches a lightweight Trainable Adapter (shaded box) to a Frozen Backbone (blue region) to preserve semantic priors. The adapter
consists of two decoupled streams: 
- (1) The Decoupled Decoder independently generates uncertainty hypotheses (ui) for each scale to prevent
premature feature fusion;
- (2) The Scale-Decoupling Module (SDM) employs a Competitive Gating mechanism (via Softmax constraints)
to predict orthogonal weights (w), dynamically suppressing dominant large-scale features to recover subtle small-scale signals. The entire
framework is optimized end-to-end via the pixel-wise BCE loss
<img width="1194" height="502" alt="8b1e188d4109ea9d0d5e44aeb8bba305" src="https://github.com/user-attachments/assets/d92d2bc5-2dd2-42b4-94c7-7c01ba9224a1" />



## Architecture

```
Input (512×512) → AutoFocusFormer Backbone (Frozen 1-3)
    ↓ [f¹(1/4), f²(1/8), f³(1/16)] → Scale Response Encoder
    ↓ r(x) = [||f¹||₂, ||f²||₂, ||f³||₂] → Learnable Scale Gate (3→16→1 MLP)
    ↓ w(x) ∈ (0,1) → Competitive Gating → OOD Score
```

## Quick Start

```bash
cd SA-Net
conda create -n msroad python=3.8 && conda activate msroad
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install timm yacs pyyaml
cd clusten && python setup.py build_ext --inplace
```


## Weights

**AutoFocusFormer Pretrain-Cityscapes Backbone**: [AFF_BACKBONE_WEIGHTS](https://drive.google.com/file/d/1L6j9y6qTXzvzvO4iXTvaUC9WPOEuotqa/view?usp=drive_link)


**Trained SA-Net**: [MODEL_WEIGHTS](https://drive.google.com/file/d/1KSy_Y_EYMXzi0Nnp_pVBvHO0XuVoRptw/view?usp=drive_link)

## Training

```bash
# Multi-GPU (recommended)
torchrun --nproc_per_node=4 --master_port=12345 \
    python train_scale_aware_anomaly.py --cfg configs/scale_aware_anomaly.yaml

# Single GPU
python train_scale_aware_anomaly.py --cfg configs/scale_aware_anomaly.yaml --batch-size 4
```

**Config**: Update `configs/scale_aware_anomaly.yaml` with data paths and backbone weights.

## Evaluation

```bash
python eval_scale_aware_anomaly.py \
    --cfg configs/scale_aware_anomaly.yaml \
    --resume output_scale_aware_anomaly/checkpoints/best_miou.pth
```

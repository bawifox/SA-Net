# MS-ROAD: Scale Inhibition Effect in Road Anomaly Segmentation

**SA-Net** addresses the **Scale Inhibition Effect** where large anomalies suppress small object detection signals through a plug-and-play Scale-Decoupling Module (SDM) with competitive gating.



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

# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Adapted for AutoFocusFormer by Ziwen 2023

from .aff_transformer import AutoFocusFormer
from .aff_seg import AFFSegModel
from .scale_aware_anomaly_detector import ScaleAwareAnomalyDetector


def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'aff':
        model = AutoFocusFormer(in_chans=config.DATA.IN_CHANS,
                                num_classes=config.MODEL.NUM_CLASSES,
                                embed_dim=config.MODEL.AFF.EMBED_DIM,
                                cluster_size=config.MODEL.AFF.CLUSTER_SIZE,
                                nbhd_size=config.MODEL.AFF.NBHD_SIZE,
                                alpha=config.MODEL.AFF.ALPHA,
                                ds_rate=config.MODEL.AFF.DS_RATE,
                                reserve_on=config.MODEL.AFF.RESERVE,
                                depths=config.MODEL.AFF.DEPTHS,
                                num_heads=config.MODEL.AFF.NUM_HEADS,
                                mlp_ratio=config.MODEL.AFF.MLP_RATIO,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                patch_norm=config.MODEL.AFF.PATCH_NORM,
                                layer_scale=config.MODEL.AFF.LAYER_SCALE,
                                img_size=config.DATA.IMG_SIZE)
    elif model_type == 'aff_seg':
        model = AFFSegModel(
            config,
            fusion_type=getattr(config.MODEL, "FUSION_TYPE", "adaptive"),
            decoder_type=getattr(config.MODEL, "DECODER_TYPE", "progressive"),
            use_multi_scale=getattr(config.MODEL, "USE_MULTI_SCALE", False),
        )
    elif model_type == 'scale_aware_anomaly':
        model = ScaleAwareAnomalyDetector(config)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model

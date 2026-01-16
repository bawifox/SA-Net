# --------------------------------------------------------
# Scale-Aware Anomaly Detection Network
# Based on AutoFocusFormer with Learnable Scale-Sensitive Gate
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from .aff_transformer import AutoFocusFormer


class ScaleResponseEncoder(nn.Module):
    """
    尺度响应建模：从三个尺度提取特征范数
    r(x) = [||f^(1)(x)||_2, ||f^(2)(x)||_2, ||f^(3)(x)||_2]
    """

    def __init__(self):
        super().__init__()

    def forward(self, features):
        """
        Args:
            features: List of features [f1, f2, f3] from three scales
                      f1: 1/4 resolution (most sensitive to small anomalies)
                      f2: 1/8 resolution
                      f3: 1/16 resolution (semantic stable, small anomalies may be smoothed)

        Returns:
            r_x: [B, H, W, 3] scale response vectors for each pixel
        """
        # features should be in order [f1, f2, f3] corresponding to scales 1/4, 1/8, 1/16
        assert len(features) == 3, "Expected 3 scale features"

        f1, f2, f3 = features

        # Compute L2 norms for each feature map
        norm1 = torch.norm(f1, p=2, dim=1, keepdim=True)  # [B, 1, H1, W1]
        norm2 = torch.norm(f2, p=2, dim=1, keepdim=True)  # [B, 1, H2, W2]
        norm3 = torch.norm(f3, p=2, dim=1, keepdim=True)  # [B, 1, H3, W3]

        # Interpolate to same resolution (use f1's resolution as target)
        target_h, target_w = f1.shape[2], f1.shape[3]

        norm2_interp = F.interpolate(norm2, size=(target_h, target_w), mode='bilinear', align_corners=False)
        norm3_interp = F.interpolate(norm3, size=(target_h, target_w), mode='bilinear', align_corners=False)

        # Concatenate scale responses: [B, 3, H, W]
        r_x = torch.cat([norm1, norm2_interp, norm3_interp], dim=1)

        # Permute to [B, H, W, 3] for pixel-wise processing
        r_x = r_x.permute(0, 2, 3, 1)

        return r_x


class LearnableScaleGate(nn.Module):
    """
    Learnable Scale Gate: 使用轻量MLP学习尺度权重
    w(x) = σ(MLP(r(x)))
    MLP结构: 3 → 16 → 1
    """

    def __init__(self):
        super().__init__()

        # Very lightweight MLP: 3 -> 16 -> 1
        self.mlp = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Output in (0, 1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize MLP weights"""
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, r_x):
        """
        Args:
            r_x: [B, H, W, 3] scale response vectors

        Returns:
            w_x: [B, H, W, 1] scale weights in (0, 1)
        """
        B, H, W, C = r_x.shape
        # Flatten spatial dimensions for MLP
        r_x_flat = r_x.reshape(-1, C)  # [B*H*W, 3]

        # Apply MLP
        w_x_flat = self.mlp(r_x_flat)  # [B*H*W, 1]

        # Reshape back to spatial dimensions
        w_x = w_x_flat.view(B, H, W, 1)  # [B, H, W, 1]

        return w_x


class LightWeightDecoder(nn.Module):
    """
    轻量Decoder：upsample + skip connections
    输出像素级logits p(x)
    """

    def __init__(self, in_channels, num_classes=2, hidden_dim=256):
        super().__init__()

        self.num_classes = num_classes

        # Simple upsampling decoder with skip-like connections
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim)

        # Final classification head
        self.conv_out = nn.Conv2d(hidden_dim, num_classes, 1)

        self._init_weights()

    def _init_weights(self):
        """Initialize decoder weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] input features

        Returns:
            logits: [B, num_classes, H, W] segmentation logits
        """
        # Simple feature processing
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # Output logits
        logits = self.conv_out(x)

        return logits


def entropy_from_logits(logits):
    """
    Compute entropy from logits: H(p(x)) = -∑p_i * log(p_i)
    Used as base uncertainty u(x) = H(p(x))

    Args:
        logits: [B, num_classes, H, W] segmentation logits

    Returns:
        entropy: [B, 1, H, W] entropy map
    """
    # Convert logits to probabilities
    probs = F.softmax(logits, dim=1)  # [B, num_classes, H, W]

    # Compute entropy: -∑p_i * log(p_i)
    log_probs = F.log_softmax(logits, dim=1)
    entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)  # [B, 1, H, W]

    return entropy


class ScaleAwareAnomalyDetector(nn.Module):
    """
    完整的尺度感知异常检测网络

    Architecture:
    1. Backbone: AutoFocusFormer (frozen, first 3 stages)
    2. Base Segmentation + Uncertainty: LightWeightDecoder + Entropy
    3. Scale-Sensitive Gate: Learnable MLP on scale responses
    4. Final OOD Score: u(x) * (1 + w(x))
    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        # Build AFF backbone (frozen)
        self.backbone = AutoFocusFormer(
            in_chans=config.DATA.IN_CHANS,
            num_classes=0,  # No classification head
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
            img_size=config.DATA.IMG_SIZE
        )

        # Freeze backbone stages 1-3 (we don't relearn semantics, just model scales)
        self._freeze_backbone_stages()

        # Get backbone dimensions for the first 3 stages
        self.stage_dims = config.MODEL.AFF.EMBED_DIM[:3]  # [128, 256, 512] typically
        # But the actual output might be EMBED_DIM (full dimensions): [128, 256, 512, 1024]

        # Scale Response Encoder
        self.scale_encoder = ScaleResponseEncoder()

        # Learnable Scale Gate (very lightweight MLP)
        self.scale_gate = LearnableScaleGate()

        # Base segmentation decoder
        # Use features from stage 3 (1/16 resolution, semantic stable)
        # AFF actually outputs [256, 512, 1024] for stages 1,2,3
        base_decoder_dim = 1024  # Hard-coded based on actual output
        self.base_decoder = LightWeightDecoder(
            in_channels=base_decoder_dim,
            num_classes=2,  # Binary: normal vs anomaly
            hidden_dim=256
        )

    def _freeze_backbone_stages(self):
        """Freeze the first 3 stages of backbone"""
        # Freeze all backbone parameters (we use pre-trained weights and don't update them)
        for param in self.backbone.parameters():
            param.requires_grad = False

    def extract_multi_scale_features(self, x):
        """
        Extract features from first 3 stages of backbone

        Args:
            x: [B, 3, H, W] input images

        Returns:
            features: List of [f1, f2, f3] features from stages 1,2,3
                     f1: 1/4 res (sensitive to small anomalies)
                     f2: 1/8 res
                     f3: 1/16 res (semantic stable)
        """
        # Get multi-scale features from backbone
        multi_scale_features = self.backbone.forward_backbone(x, return_multi_scale=True)

        # Extract first 3 stages
        # multi_scale_features is list of (feat, h, w) tuples
        features = [feat for feat, _, _ in multi_scale_features[:3]]

        return features

    def forward(self, x):
        """
        Forward pass

        Args:
            x: [B, 3, H, W] input images

        Returns:
            ood_scores: [B, 1, H, W] final anomaly scores OOD(x) = u(x) * (1 + w(x))
            base_logits: [B, 2, H, W] base segmentation logits (for analysis)
            uncertainty: [B, 1, H, W] base uncertainty u(x) = H(p(x))
            scale_weights: [B, 1, H, W] scale weights w(x)
        """
        B, C, H, W = x.shape

        # 1. Extract multi-scale features from backbone (first 3 stages)
        features = self.extract_multi_scale_features(x)  # [f1, f2, f3]

        # 2. Scale Response Encoding: r(x) = [||f1||_2, ||f2||_2, ||f3||_2]
        r_x = self.scale_encoder(features)  # [B, H, W, 3]

        # 3. Learnable Scale Gate: w(x) = σ(MLP(r(x)))
        scale_weights = self.scale_gate(r_x)  # [B, H, W, 1]
        # Convert to [B, 1, H, W] format
        scale_weights = scale_weights.permute(0, 3, 1, 2)  # [B, 1, H, W]

        # 4. Base segmentation using stage 3 features (semantic stable)
        f3 = features[-1]  # Stage 3 features (1/16 resolution)
        base_logits = self.base_decoder(f3)  # [B, 2, H', W']

        # Upsample base logits to original resolution if needed
        if base_logits.shape[2:] != (H, W):
            base_logits = F.interpolate(base_logits, size=(H, W), mode='bilinear', align_corners=False)

        # 5. Base uncertainty: u(x) = H(p(x))
        uncertainty = entropy_from_logits(base_logits)  # [B, 1, H, W] (upsampled to full resolution)

        # 6. Upsample scale weights to match uncertainty resolution if needed
        if scale_weights.shape[2:] != uncertainty.shape[2:]:
            scale_weights = F.interpolate(scale_weights, size=uncertainty.shape[2:], mode='bilinear', align_corners=False)

        # 7. Final OOD Score: OOD(x) = u(x) * (1 + w(x))
        # This amplifies uncertainty where scale weights suggest it should be uncertain
        ood_scores = uncertainty * (1 + scale_weights)  # [B, 1, H, W]

        # 7. Final OOD Score: OOD(x) = u(x) * (1 + w(x))
        # This amplifies uncertainty where scale weights suggest it should be uncertain
        ood_scores = uncertainty * (1 + scale_weights)  # [B, 1, H, W]

        return {
            'ood_scores': ood_scores,           # Final anomaly scores
            'base_logits': base_logits,         # Base segmentation logits
            'uncertainty': uncertainty,         # Base uncertainty H(p(x))
            'scale_weights': scale_weights,     # Scale weights w(x)
        }

    def load_backbone_weights(self, checkpoint_path):
        """
        Load pre-trained AFF backbone weights (Cityscapes pre-training)

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Handle different checkpoint formats
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Filter backbone weights
        backbone_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('backbone.') or (not k.startswith('head.') and not k.startswith('norm.')):
                # Remove 'backbone.' prefix if exists
                new_key = k.replace('backbone.', '')
                backbone_state_dict[new_key] = v

        # Load into backbone
        missing_keys, unexpected_keys = self.backbone.load_state_dict(backbone_state_dict, strict=False)

        print(f"Loaded backbone weights from {checkpoint_path}")
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")

        return missing_keys, unexpected_keys

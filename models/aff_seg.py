# --------------------------------------------------------
# AutoFocusFormer-based Semantic Segmentation Model
# with Text-Vision Feature Fusion
# --------------------------------------------------------

import torch
import torch.nn as nn
from .aff_transformer import AutoFocusFormer
from .text_encoder import DualTextEncoder, ClassTextEncoder
from .feature_fusion import CrossAttentionFusion, AdaptiveFusion, ConcatFusion
from .grounding_fusion import GroundingDINOFusion
from .decoder import SimpleDecoder, FPNDecoder, ProgressiveDecoder, Mask2FormerDecoder
from .mask2former_decoder import FullMask2FormerDecoder


class AFFSegModel(nn.Module):
    """
    Complete segmentation model:
    AFF Backbone -> Text-Vision Fusion -> Decoder -> Segmentation Logits
    """
    
    def __init__(self, config, fusion_type='adaptive', decoder_type='progressive', use_multi_scale=False):
        """
        Args:
            config: Configuration object
            fusion_type: 'cross_attention', 'adaptive', or 'concat'
            decoder_type: 'simple', 'fpn', or 'progressive'
            use_multi_scale: Whether to use multi-scale features from backbone
        """
        super().__init__()
        
        self.config = config
        self.use_multi_scale = use_multi_scale
        
        # Build AFF backbone
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
        
        # Get backbone output dimension
        backbone_dim = config.MODEL.AFF.EMBED_DIM[-1]
        
        self.text_enabled = fusion_type != 'none'
        fusion_dim = backbone_dim

        if self.text_enabled:
            # Build text encoder
            # Use dual text encoder: "foreground" and "unknown objects"
            text_dim = 256  # Text embedding dimension
            self.text_encoder = DualTextEncoder(
                embed_dim=text_dim,
                use_clip=False  # Use simple learnable embeddings
            )
            
            # Build feature fusion module
            if fusion_type == 'grounding_dino' or fusion_type == 'grounding':
                # GroundingDINO-style fusion
                num_enhancer_layers = getattr(config.MODEL, 'NUM_ENHANCER_LAYERS', 2)
                num_decoder_layers = getattr(config.MODEL, 'NUM_DECODER_LAYERS', 3)
                self.fusion = GroundingDINOFusion(
                    visual_dim=backbone_dim,
                    text_dim=text_dim,
                    num_enhancer_layers=num_enhancer_layers,
                    num_decoder_layers=num_decoder_layers,
                    num_heads=8,
                    ffn_dim=backbone_dim * 4,
                    dropout=0.1
                )
            elif fusion_type == 'cross_attention':
                self.fusion = CrossAttentionFusion(
                    visual_dim=backbone_dim,
                    text_dim=text_dim,
                    out_dim=fusion_dim
                )
            elif fusion_type == 'adaptive':
                self.fusion = AdaptiveFusion(
                    visual_dim=backbone_dim,
                    text_dim=text_dim,
                    out_dim=fusion_dim
                )
            elif fusion_type == 'concat':
                self.fusion = ConcatFusion(
                    visual_dim=backbone_dim,
                    text_dim=text_dim,
                    out_dim=fusion_dim
                )
            else:
                raise ValueError(f"Unknown fusion type: {fusion_type}")
        else:
            # Vision-only mode: no fusion module
            self.fusion = None
        
        # Build decoder
        num_classes = config.MODEL.NUM_CLASSES
        if decoder_type == 'simple':
            self.decoder = SimpleDecoder(
                in_channels=fusion_dim,
                num_classes=num_classes,
                hidden_dim=256
            )
        elif decoder_type == 'fpn':
            # For FPN, we need multi-scale features
            if not use_multi_scale:
                raise ValueError("FPN decoder requires use_multi_scale=True")
            # Use last few stages for FPN
            in_channels_list = config.MODEL.AFF.EMBED_DIM[-3:]  # Last 3 stages
            hidden_dim = getattr(config.MODEL, 'DECODER_HIDDEN_DIM', 256)
            # Use actual channel dims per stage; only highest may be fused to backbone_dim
            self.decoder = FPNDecoder(
                in_channels_list=in_channels_list,
                num_classes=num_classes,
                hidden_dim=hidden_dim
            )
        elif decoder_type == 'progressive':
            self.decoder = ProgressiveDecoder(
                in_channels=fusion_dim,
                num_classes=num_classes,
                hidden_dim=256,
                num_stages=3  # 3 upsampling stages (8x -> 4x -> 2x -> 1x)
            )
        elif decoder_type == 'mask2former':
            # Full Mask2Former decoder (requires multi-scale features)
            if not use_multi_scale:
                raise ValueError("Full Mask2Former decoder requires use_multi_scale=True")
            
            conv_dim = getattr(config.MODEL, 'DECODER_CONV_DIM', 256)
            mask_dim = getattr(config.MODEL, 'DECODER_MASK_DIM', 256)
            hidden_dim = getattr(config.MODEL, 'DECODER_HIDDEN_DIM', 256)
            num_queries = getattr(config.MODEL, 'DECODER_NUM_QUERIES', 100)
            nheads = getattr(config.MODEL, 'DECODER_NHEADS', 8)
            dim_feedforward = getattr(config.MODEL, 'DECODER_DIM_FFN', 2048)
            dec_layers = getattr(config.MODEL, 'DECODER_LAYERS', 9)
            pre_norm = getattr(config.MODEL, 'DECODER_PRE_NORM', False)
            norm = getattr(config.MODEL, 'DECODER_NORM', 'GN')
            num_groups = getattr(config.MODEL, 'DECODER_NUM_GROUPS', 32)
            
            # Use last 3 stages for multi-scale features
            in_channels_list = config.MODEL.AFF.EMBED_DIM[-3:]  # Last 3 stages
            
            self.decoder = FullMask2FormerDecoder(
                in_channels_list=in_channels_list,
                num_classes=num_classes,
                conv_dim=conv_dim,
                mask_dim=mask_dim,
                hidden_dim=hidden_dim,
                num_queries=num_queries,
                nheads=nheads,
                dim_feedforward=dim_feedforward,
                dec_layers=dec_layers,
                pre_norm=pre_norm,
                norm=norm,
                num_groups=num_groups,
            )
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")
        
        self.decoder_type = decoder_type
        self.fusion_type = fusion_type
    
    def forward(self, x, return_features=False, return_intermediate_features=False):
        """
        Args:
            x: [B, 3, H, W] input images
            return_features: Whether to return intermediate features (visual/text/fused)
            return_intermediate_features: Whether to return decoder intermediate features (for scale-weighted scoring)
        
        Returns:
            logits: [B, num_classes, H, W] segmentation logits
            (optional) features: Intermediate features if return_features=True
            (optional) intermediate_features: Decoder intermediate features if return_intermediate_features=True
        """
        B, C, H, W = x.shape
        target_size = (H, W)
        
        # Extract visual features from backbone
        if self.use_multi_scale:
            multi_scale_features = self.backbone.forward_backbone(x, return_multi_scale=True)
            # Use the last (finest) scale for fusion
            visual_feat = multi_scale_features[-1][0]  # [B, C, H', W']
        else:
            visual_feat, h_out, w_out = self.backbone.forward_backbone(x, return_multi_scale=False)
            # [B, C, H', W']
        
        if self.text_enabled:
            # Get text features: "foreground" and "unknown objects"
            text_feat_both = self.text_encoder(mode='both')  # [2, text_dim]
            # Expand to batch size
            text_feat = text_feat_both.unsqueeze(0).expand(B, -1, -1)  # [B, 2, text_dim]
            # For fusion, we can use both text features
            text_feat_combined = text_feat.mean(dim=1)  # [B, text_dim]
            
            # Fuse visual and text features
            if self.fusion_type == 'grounding_dino' or self.fusion_type == 'grounding':
                # GroundingDINO fusion expects [B, N, text_dim] format
                text_feat_for_fusion = text_feat_combined.unsqueeze(1)  # [B, 1, text_dim]
                fused_feat = self.fusion(visual_feat, text_feat_for_fusion)  # [B, fusion_dim, H', W']
            else:
                # Other fusion methods expect [B, text_dim]
                fused_feat = self.fusion(visual_feat, text_feat_combined)  # [B, fusion_dim, H', W']
        else:
            # 纯视觉分支：不做跨模态融合，直接用 backbone 特征
            fused_feat = visual_feat
            text_feat = None
        
        # Decode to segmentation logits
        if self.decoder_type == 'mask2former' and self.use_multi_scale:
            # Full Mask2Former: use multi-scale features directly
            # Get last 3 stages from backbone
            # multi_scale_features is a list of (feat, h, w) tuples from early to late layers
            # EMBED_DIM = [128, 256, 512, 1024] for 4 stages
            # We need stages 1, 2, 3 (channels: 256, 512, 1024)
            # in_channels_list = [256, 512, 1024] (from config)
            # multi_scale_features has 4 elements (one per stage), we need indices 1, 2, 3
            in_channels_list = self.config.MODEL.AFF.EMBED_DIM[-3:]  # [256, 512, 1024]
            
            # Match features by channel dimension
            multi_scale_feat_list = []
            for target_channels in in_channels_list:
                # Find feature with matching channel dimension
                found = False
                for feat, _, _ in multi_scale_features:
                    if feat.shape[1] == target_channels:
                        # Only fuse text on the highest-stage feature; lower stages stay visual-only to avoid dim mismatch
                        if self.text_enabled and target_channels == self.config.MODEL.AFF.EMBED_DIM[-1]:
                            if self.fusion_type == 'grounding_dino' or self.fusion_type == 'grounding':
                                text_feat_for_fusion = text_feat_combined.unsqueeze(1)  # [B, 1, text_dim]
                                fused = self.fusion(feat, text_feat_for_fusion)
                            else:
                                fused = self.fusion(feat, text_feat_combined)
                        else:
                            fused = feat
                        multi_scale_feat_list.append(fused)
                        found = True
                        break
                if not found:
                    raise ValueError(f"Could not find feature with {target_channels} channels")
            
            # Features should be in order [256, 512, 1024] (low->high res)
            # Decoder expects high->low res, so reverse
            multi_scale_feat_list = multi_scale_feat_list[::-1]  # Reverse to high->low resolution
            
            if return_intermediate_features:
                logits, intermediate_features = self.decoder(multi_scale_feat_list, target_size=target_size, return_intermediate=True)
            else:
                logits = self.decoder(multi_scale_feat_list, target_size=target_size)
        elif (self.decoder_type == 'fpn') and self.use_multi_scale:
            # FPN decoder with multi-scale
            fused_features_list = []
            in_channels_list = self.config.MODEL.AFF.EMBED_DIM[-3:]  # expected order low->high: [256, 512, 1024]
            backbone_dim = self.config.MODEL.AFF.EMBED_DIM[-1]

            # Build features in the expected channel order to avoid conv mismatch
            for target_channels in in_channels_list:
                matched = None
                for feat, _, _ in multi_scale_features:
                    if feat.shape[1] == target_channels:
                        if self.text_enabled and feat.shape[1] == backbone_dim:
                            if self.fusion_type == 'grounding_dino' or self.fusion_type == 'grounding':
                                text_feat_for_fusion = text_feat_combined.unsqueeze(1)
                                fused = self.fusion(feat, text_feat_for_fusion)
                            else:
                                fused = self.fusion(feat, text_feat_combined)
                        else:
                            fused = feat
                        matched = fused
                        break
                if matched is None:
                    raise ValueError(f"Could not find feature with {target_channels} channels for FPN")
                fused_features_list.append(matched)

            if return_intermediate_features:
                logits, intermediate_features = self.decoder(fused_features_list, target_size=target_size, return_intermediate=True)
            else:
                logits = self.decoder(fused_features_list, target_size=target_size)
        else:
            # For single-scale decoders (SimpleDecoder, ProgressiveDecoder)
            if return_intermediate_features:
                logits, intermediate_features = self.decoder(fused_feat, target_size=target_size, return_intermediate=True)
            else:
                logits = self.decoder(fused_feat, target_size=target_size)
        
        # Handle return values
        if return_features and return_intermediate_features:
            return logits, {
                'visual_feat': visual_feat,
                'text_feat': text_feat,
                'fused_feat': fused_feat,
            }, intermediate_features
        elif return_features:
            return logits, {
                'visual_feat': visual_feat,
                'text_feat': text_feat,
                'fused_feat': fused_feat,
            }
        elif return_intermediate_features:
            return logits, intermediate_features
        
        return logits
    
    def load_backbone_weights(self, checkpoint_path, strict=False):
        """
        Load pre-trained AFF backbone weights.
        
        Args:
            checkpoint_path: Path to checkpoint file
            strict: Whether to strictly match all keys
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Filter backbone weights (remove classification head)
        backbone_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('backbone.') or (not k.startswith('head.') and not k.startswith('norm.')):
                # Remove 'backbone.' prefix if exists
                new_key = k.replace('backbone.', '')
                backbone_state_dict[new_key] = v
        
        # Load into backbone
        missing_keys, unexpected_keys = self.backbone.load_state_dict(backbone_state_dict, strict=strict)
        
        if missing_keys:
            print(f"Missing keys when loading backbone: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys when loading backbone: {unexpected_keys}")
        
        return missing_keys, unexpected_keys








import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SimpleDecoder(nn.Module):
    """
    简单的上采样解码器：Conv -> Upsample 到输入分辨率。
    """

    def __init__(self, in_channels: int, num_classes: int, hidden_dim: int = 256):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, hidden_dim)
        self.conv2 = ConvBlock(hidden_dim, hidden_dim)
        self.head = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, x, target_size, return_intermediate=False):
        """
        Args:
            x: [B, C, H', W']
            target_size: (H, W)
            return_intermediate: If True, return intermediate features for scale-weighted scoring
        
        Returns:
            out: [B, num_classes, H, W] segmentation logits
            (optional) intermediate_features: dict with 'features_1_4' and 'features_1_16' if return_intermediate=True
        """
        B, C, H_in, W_in = x.shape
        
        # Store input as 1/16 scale feature (assuming input is at 1/16 scale)
        features_1_16 = x
        
        x = self.conv1(x)
        x = self.conv2(x)
        
        # After conv2, estimate 1/4 scale feature
        # If input was H/16 x W/16, after 2x upsampling we get H/8, need 4x total for H/4
        features_1_4 = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        x = self.head(x)
        if x.shape[-2:] != target_size:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        
        if return_intermediate:
            intermediate_features = {
                'features_1_4': features_1_4,
                'features_1_16': features_1_16,
            }
            return x, intermediate_features
        return x


class ProgressiveDecoder(nn.Module):
    """
    渐进式上采样解码器：多次 2x 上采样 + Conv。
    """

    def __init__(self, in_channels: int, num_classes: int, hidden_dim: int = 256, num_stages: int = 3):
        super().__init__()
        self.num_stages = num_stages
        stages = []
        in_ch = in_channels
        for _ in range(num_stages):
            stages.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    ConvBlock(in_ch, hidden_dim),
                )
            )
            in_ch = hidden_dim
        self.stages = nn.ModuleList(stages)
        self.head = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, x, target_size, return_intermediate=False):
        """
        Args:
            x: [B, C, H', W']
            target_size: (H, W)
            return_intermediate: If True, return intermediate features for scale-weighted scoring
        
        Returns:
            out: [B, num_classes, H, W] segmentation logits
            (optional) intermediate_features: dict with 'features_1_4' and 'features_1_16' if return_intermediate=True
        """
        # Store input as 1/16 scale feature (assuming input is at 1/16 scale)
        features_1_16 = x
        
        # Process through stages, capturing intermediate feature at ~1/4 scale
        features_1_4 = None
        for i, stage in enumerate(self.stages):
            x = stage(x)
            # After first stage, we're at ~1/8 scale, after second at ~1/4 scale
            if return_intermediate and i == 1:
                features_1_4 = x
        
        # If we didn't capture 1/4 feature (less than 2 stages), use current x
        if return_intermediate and features_1_4 is None:
            features_1_4 = x
        
        x = self.head(x)
        if x.shape[-2:] != target_size:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        
        if return_intermediate:
            intermediate_features = {
                'features_1_4': features_1_4,
                'features_1_16': features_1_16,
            }
            return x, intermediate_features
        return x


class FPNDecoder(nn.Module):
    """
    简化版 FPN 解码器，接受多尺度特征列表。
    """

    def __init__(self, in_channels_list, num_classes: int, hidden_dim: int = 256):
        """
        Args:
            in_channels_list: 多尺度特征通道数列表，例如 [C3, C4, C5]
            num_classes: 类别数
            hidden_dim: FPN 中间通道数
        """
        super().__init__()
        self.num_levels = len(in_channels_list)

        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(c, hidden_dim, kernel_size=1) for c in in_channels_list]
        )
        self.output_convs = nn.ModuleList(
            [ConvBlock(hidden_dim, hidden_dim) for _ in in_channels_list]
        )
        self.head = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, features, target_size, return_intermediate=False):
        """
        Args:
            features: List[Tensor], 每个元素形状 [B, C_i, H_i, W_i]，从高到低分辨率
            target_size: (H, W)
            return_intermediate: If True, return intermediate features for scale-weighted scoring
        
        Returns:
            out: [B, num_classes, H, W] segmentation logits
            (optional) intermediate_features: dict with 'features_1_4' and 'features_1_16' if return_intermediate=True
        """
        assert len(features) == self.num_levels, "FPNDecoder expects len(features) == len(in_channels_list)"

        laterals = [l_conv(f) for l_conv, f in zip(self.lateral_convs, features)]

        # 自顶向下路径融合
        for i in range(self.num_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[-2:]
            upsampled = F.interpolate(
                laterals[i],
                size=prev_shape,
                mode="bilinear",
                align_corners=False,
            )
            laterals[i - 1] = laterals[i - 1] + upsampled

        out = self.output_convs[0](laterals[0])
        
        # Extract intermediate features for scale-weighted scoring
        intermediate_features = None
        if return_intermediate:
            # laterals[0] is the highest resolution feature (typically 1/4 scale)
            # features[-1] is the lowest resolution feature (typically 1/16 scale)
            # We need to match the scales appropriately
            # Assuming features are ordered from high to low resolution:
            # - laterals[0] after fusion is at ~1/4 scale
            # - features[-1] (original input) is at ~1/16 scale
            features_1_4 = laterals[0]  # Highest resolution lateral (after fusion)
            features_1_16 = features[-1]  # Lowest resolution input feature
            intermediate_features = {
                'features_1_4': features_1_4,
                'features_1_16': features_1_16,
            }
        
        out = self.head(out)
        if out.shape[-2:] != target_size:
            out = F.interpolate(out, size=target_size, mode="bilinear", align_corners=False)
        
        if return_intermediate:
            return out, intermediate_features
        return out


class Mask2FormerPixelDecoder(nn.Module):
    """
    Mask2Former 风格的 Pixel Decoder (基于 FPN)。
    参考: mask2former/modeling/pixel_decoder/fpn.py
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        conv_dim: int = 256,
        mask_dim: int = 256,
        norm: str = "GN",  # GroupNorm
        num_groups: int = 32,
    ):
        """
        Args:
            in_channels: 输入特征通道数（backbone 输出）
            num_classes: 类别数
            conv_dim: FPN 中间通道数
            mask_dim: mask 特征维度
            norm: 归一化类型 ('GN' 或 'BN')
            num_groups: GroupNorm 的组数
        """
        super().__init__()
        
        # 使用单尺度输入（backbone 最后一层特征）
        # 如果需要多尺度，可以通过 use_multi_scale=True 传入多尺度特征
        
        if norm == "GN":
            self.norm = lambda dim: nn.GroupNorm(num_groups, dim)
        elif norm == "BN":
            self.norm = lambda dim: nn.BatchNorm2d(dim)
        else:
            raise ValueError(f"Unknown norm type: {norm}")
        
        use_bias = norm == ""
        
        # 输入投影层
        self.input_proj = nn.Conv2d(
            in_channels,
            conv_dim,
            kernel_size=1,
            bias=use_bias,
        )
        if norm != "":
            self.input_norm = self.norm(conv_dim)
        else:
            self.input_norm = nn.Identity()
        
        # FPN 风格的解码器（简化版，单尺度输入）
        # 使用多个上采样 + 卷积层来恢复分辨率
        self.decoder_layers = nn.ModuleList()
        in_dim = conv_dim
        for i in range(3):  # 3 个解码层
            self.decoder_layers.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(
                        in_dim,
                        conv_dim,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=use_bias,
                    ),
                    self.norm(conv_dim),
                    nn.ReLU(inplace=True),
                )
            )
            in_dim = conv_dim
        
        # Mask features (用于生成 mask)
        self.mask_features = nn.Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        
        # 分类头
        self.class_head = nn.Conv2d(mask_dim, num_classes, kernel_size=1)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, target_size, return_intermediate=False):
        """
        Args:
            x: [B, C, H', W'] 输入特征（backbone 输出）
            target_size: (H, W) 目标输出尺寸
            return_intermediate: If True, return intermediate features for scale-weighted scoring
        
        Returns:
            logits: [B, num_classes, H, W] 分割 logits
            (optional) intermediate_features: dict with 'features_1_4' and 'features_1_16' if return_intermediate=True
        """
        # Store input as 1/16 scale feature
        features_1_16 = x
        
        # 输入投影
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = F.relu(x)
        
        # FPN 风格解码，捕获中间特征
        features_1_4 = None
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x)
            # After first decoder layer, we're at ~1/8 scale, after second at ~1/4 scale
            if return_intermediate and i == 1:
                features_1_4 = x
        
        # If we didn't capture 1/4 feature (less than 2 layers), use current x
        if return_intermediate and features_1_4 is None:
            features_1_4 = x
        
        # 生成 mask features
        mask_feat = self.mask_features(x)
        
        # 分类头
        logits = self.class_head(mask_feat)
        
        # 插值到目标尺寸
        if logits.shape[-2:] != target_size:
            logits = F.interpolate(
                logits,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        
        if return_intermediate:
            intermediate_features = {
                'features_1_4': features_1_4,
                'features_1_16': features_1_16,
            }
            return logits, intermediate_features
        
        return logits


class Mask2FormerDecoder(nn.Module):
    """
    Mask2Former 风格的完整解码器（支持多尺度特征）。
    如果 use_multi_scale=True，使用多尺度 FPN；否则使用单尺度解码。
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        conv_dim: int = 256,
        mask_dim: int = 256,
        norm: str = "GN",
        num_groups: int = 32,
        use_multi_scale: bool = False,
        in_channels_list: list = None,  # 多尺度时的通道列表
    ):
        """
        Args:
            in_channels: 单尺度输入时的通道数
            num_classes: 类别数
            conv_dim: FPN 中间通道数
            mask_dim: mask 特征维度
            norm: 归一化类型
            num_groups: GroupNorm 的组数
            use_multi_scale: 是否使用多尺度特征
            in_channels_list: 多尺度时的通道列表，例如 [256, 512, 1024]
        """
        super().__init__()
        self.use_multi_scale = use_multi_scale
        
        if use_multi_scale:
            # 多尺度 FPN 解码器
            if in_channels_list is None:
                raise ValueError("in_channels_list must be provided when use_multi_scale=True")
            
            self.pixel_decoder = FPNDecoder(
                in_channels_list=in_channels_list,
                num_classes=num_classes,
                hidden_dim=conv_dim,
            )
        else:
            # 单尺度 Pixel Decoder
            self.pixel_decoder = Mask2FormerPixelDecoder(
                in_channels=in_channels,
                num_classes=num_classes,
                conv_dim=conv_dim,
                mask_dim=mask_dim,
                norm=norm,
                num_groups=num_groups,
            )
    
    def forward(self, x, target_size):
        """
        Args:
            x: 单尺度时为 [B, C, H', W']，多尺度时为 List[Tensor]
            target_size: (H, W)
        """
        return self.pixel_decoder(x, target_size)






























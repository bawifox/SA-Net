import torch
import torch.nn as nn


class FullMask2FormerDecoder(nn.Module):
    """
    占位实现：仅为避免导入失败。
    如需使用 Mask2Former 全量解码器，请恢复完整实现并设置 DECODER_TYPE='mask2former'。
    当前配置使用 FPN，不会实例化本类。
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError(
            "FullMask2FormerDecoder 占位实现，当前配置请使用 DECODER_TYPE='fpn'"
        )

    def forward(self, *args, **kwargs):  # pragma: no cover - 占位
        raise NotImplementedError(
            "FullMask2FormerDecoder 占位实现，当前配置请使用 DECODER_TYPE='fpn'"
        )



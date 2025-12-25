import torch
import torch.nn as nn
from utils.mask_input import mask_input

class PositionEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_pos_feats = config.dim_feature[-1]
        self.temperature = config.max_length

        # 在初始化时预计算位置编码的除数，避免在每次前向传播时重新计算
        dim_t = torch.arange(self.num_pos_feats,dtype=torch.float)
        dim_t = self.temperature ** (2 * dim_t / self.num_pos_feats)
        dim_t = dim_t / (2 * torch.pi)  # 使用torch.pi代替硬编码的值
        self.register_buffer('dim_t', dim_t.view(1, 1, -1), persistent=False)

        # 初始化alpha参数
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x, mask=None):
        b, l, d = x.shape

        # 更高效地生成位置索引
        pos_indices = torch.arange(1, l + 1, device=x.device, dtype=x.dtype).view(1, l, 1)

        # 除以预计算的dim_t（自动进行广播）
        pos_enc = pos_indices / self.dim_t

        # 对偶数索引应用sin，对奇数索引应用cos，提高效率
        pos_enc_sin = torch.sin(pos_enc[:, :, 0::2])
        pos_enc_cos = torch.cos(pos_enc[:, :, 1::2])

        # 交错排列正弦和余弦值
        pos_enc = torch.zeros_like(x)
        pos_enc[:, :, 0::2] = pos_enc_sin
        pos_enc[:, :, 1::2] = pos_enc_cos

        # 将位置编码添加到输入并应用mask
        x = x + pos_enc * self.alpha
        if mask is not None:
            x = mask_input(x, mask)

        return x


import torch.nn as nn
from utils.mask_input import mask_input


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, act_layer=nn.LeakyReLU):
        super().__init__()
        # 残差连接
        self.res_linear = nn.Linear(input_dim, output_dim)

        # 主要网络结构，使用Sequential整合
        self.main_path = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act_layer(),
            nn.Linear(hidden_dim, output_dim),
            act_layer()
        )

    def forward(self, x, mask=None):
        # 残差连接
        residual = self.res_linear(x)

        # 主路径
        x = self.main_path(x)

        # 残差相加
        x = x + residual

        # 应用掩码（如果提供）
        if mask is not None:
            x = mask_input(x, mask)

        return x



import torch.nn as nn
from models.DWC import DWC
from models.MoveNorm import MoveNorm
from models.FocusedLinearAttention import FocusedLinearAttention
from models.MLP import MLP
from models.PositionEmbedding import PositionEmbedding
from utils.mask_input import mask_input


class CrossAttention(nn.Module):
    """
    CrossAttention模块实现注意力机制，用于在两组输入之间进行交互
    """

    def __init__(self, dim, config):
        super().__init__()
        self.dwc = DWC(dim, dim, config)
        # 位置编码
        self.pos_q = PositionEmbedding(config)
        self.pos_k = PositionEmbedding(config)

        # 标准化层
        self.norm1 = MoveNorm(config)
        self.norm2 = MoveNorm(config)

        # 注意力层
        self.attn = FocusedLinearAttention(dim, config)

        # MLP层
        self.mlp = MLP(
            input_dim=dim,
            hidden_dim=(dim * config.mlp_ratio),
            output_dim=dim,
            act_layer=nn.GELU
        )

        # 最终投影层
        self.fc = nn.Linear(dim, dim, bias=True)

    def forward(self, query, input_x, mask_q=None, mask_kv=None):
        """
        前向传播函数

        参数:
            query: 查询张量
            input_x: 输入张量(key-value)
            mask_q: 查询的掩码
            mask_kv: key-value的掩码
        """
        # 应用位置编码和掩码
        query_pos = self.pos_q(query)
        q = mask_input(query_pos, mask_q) if mask_q is not None else query_pos

        k_pos = self.pos_k(input_x)
        k = mask_input(k_pos, mask_kv) if mask_kv is not None else k_pos

        # 应用注意力机制，实现残差连接
        v = self.norm1(k)

        attn_output = self.attn(q, k, v, mask_q, mask_kv)
        q = q + attn_output

        # 应用MLP，实现残差连接
        q_norm = self.norm2(q)
        mlp_output = self.mlp(q_norm, mask_q)
        output = q + mlp_output

        # 最终投影

        if mask_q is not None:
            output = mask_input(self.fc(output), mask_q)

        return output

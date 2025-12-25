import torch.nn as nn
from models.DWC import DWC
from models.MoveNorm import MoveNorm
from models.FocusedLinearAttention import FocusedLinearAttention
from models.MLP import MLP
from models.PositionEmbedding import PositionEmbedding
from utils.mask_input import mask_input

class SelfAttention(nn.Module):
    def __init__(self, dim, config):
        super().__init__()
        self.dwc = DWC(dim, dim, config)
        self.pos_q = PositionEmbedding(config)
        self.pos_k = PositionEmbedding(config)
        self.norm1 = MoveNorm(config)
        self.attn = FocusedLinearAttention(dim, config)
        self.norm2 = MoveNorm(config)
        self.mlp = MLP(input_dim=dim,hidden_dim=(dim * config.mlp_ratio),output_dim=dim,act_layer=nn.GELU)
        self.fc = nn.Linear(dim, dim)

    def forward(self, input_x, mask=None):
        # 保存原始输入用于残差连接
        dwc_term = self.dwc(input_x, mask)

        # 应用位置编码到查询和键值
        q = self.pos_q(input_x)
        k = self.pos_k(input_x)

        # 如果有掩码，立即应用
        if mask is not None:
            q = mask_input(q, mask)
            k = mask_input(k, mask)

        # 归一化和自注意力
        v = self.norm1(k)
        attn_output = self.attn(q, k, v, mask, mask)

        # 第一个残差连接
        q = q + attn_output

        # 归一化和MLP
        q_norm = self.norm2(q)
        mlp_output = self.mlp(q_norm, mask)

        # 第二个残差连接
        q = q + mlp_output

        # 最终的线性变换和残差连接
        output = self.fc(q) + dwc_term

        # 最后应用掩码（如果有）
        if mask is not None:
            output = mask_input(output, mask)

        return output

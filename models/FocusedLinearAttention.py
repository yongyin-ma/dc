import torch
import torch.nn as nn
import math
from einops import rearrange


class FocusedLinearAttention(nn.Module):
    def __init__(self, dim, config):
        super().__init__()
        num_heads = config.num_heads
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        # 投影矩阵
        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

        # 聚焦因子
        self.focusing_factor = nn.Parameter(torch.tensor(3.0))
        self.kernel_function = nn.GELU()
        self.eps = 1e-6

    def forward(self, q, k, v, mask_q=None, mask_kv=None):
        batch_size, seq_len, dim_size = q.shape

        # 线性投影
        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)

        # 应用核函数并确保数值稳定性
        q = torch.abs(self.kernel_function(q)) + self.eps
        k = torch.abs(self.kernel_function(k)) + self.eps

        # 计算归一化因子
        q_norm = q.mean(dim=-1, keepdim=True)
        k_norm = k.mean(dim=-1, keepdim=True)

        # 应用聚焦变换
        q = torch.log(q) * self.focusing_factor
        k = torch.log(k) * self.focusing_factor

        # Softmax归一化并缩放
        q = torch.nn.functional.softmax(q, dim=-1) * q_norm
        k = torch.nn.functional.softmax(k, dim=-1) * k_norm

        # 应用掩码
        if mask_q is not None:
            q = q * mask_q

        if mask_kv is not None:
            k = k * mask_kv
            v = v * mask_kv

        # 重排张量维度为多头格式
        if mask_kv is not None:
            # 处理有掩码情况
            mask_expanded_kv = torch.repeat_interleave(mask_kv, repeats=self.num_heads, dim=0)
            q_heads = rearrange(q, "b n (h c) -> (b h) n c", h=self.num_heads)
            k_heads = rearrange(k, "b n (h c) -> (b h) n c", h=self.num_heads)
            v_heads = rearrange(v, "b n (h c) -> (b h) n c", h=self.num_heads)

            # 计算有效长度及均值
            length_k = k_heads.shape[-2]
            sqrt_length_k = math.sqrt(length_k)

            # 计算掩码下的平均值
            mask_sum = torch.sum(mask_expanded_kv, dim=-2).clamp(min=self.eps)
            k_mean = torch.sum(k_heads, dim=-2) / mask_sum

        else:
            # 处理无掩码情况
            q_heads = rearrange(q, "b n (h c) -> (b h) n c", h=self.num_heads)
            k_heads = rearrange(k, "b n (h c) -> (b h) n c", h=self.num_heads)
            v_heads = rearrange(v, "b n (h c) -> (b h) n c", h=self.num_heads)

            length_k = k_heads.shape[-2]
            sqrt_length_k = math.sqrt(length_k)
            k_mean = torch.mean(k_heads, dim=-2)

        # 计算归一化因子的倒数
        z_inv = torch.einsum("b n d, b d -> b n", q_heads, k_mean) + self.eps
        z_inv = z_inv.unsqueeze(-1)

        # 计算键值对乘积
        kv = torch.einsum("b n c, b n d -> b c d", k_heads / sqrt_length_k, v_heads / sqrt_length_k)

        # 计算注意力输出
        y_hat = torch.einsum("b i c, b c d -> b i d", q_heads, kv)
        y_hat = y_hat / z_inv

        # 重排回原始维度
        y_hat = rearrange(y_hat, "(b h) n c -> b n (h c)", h=self.num_heads)

        # 最终投影
        y_hat = self.proj(y_hat)
        
        if not mask_q == None:
            y_hat = y_hat.masked_fill(mask_q < 0.5, torch.tensor(0.0, device=q.device))
        else:
            y_hat = y_hat
        
        
        
        return y_hat

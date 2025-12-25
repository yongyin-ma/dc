import torch
import torch.nn as nn
import math
from einops import rearrange, repeat

class AdvancedSparseFocusedAttention(nn.Module):
    """
    高级稀疏化的FocusedLinearAttention实现，整合多种优化技术：
    - 块稀疏化 (Block Sparsity)
    - 结构化稀疏 (Structured Sparsity)
    - 动态稀疏度 (Dynamic Sparsity)
    - 量化感知 (Quantization-aware)
    - 可学习稀疏度 (Learnable Sparsity)
    """

    def __init__(self, dim, config):
        super().__init__()
        num_heads = config.num_heads
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # 投影矩阵
        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

        # 聚焦因子
        self.focusing_factor = nn.Parameter(torch.tensor(3.0))
        self.kernel_function = nn.LeakyReLU()
        self.eps = 1e-6

        # 基本稀疏化参数
        self.sparsity_mode = getattr(config, 'sparsity_mode', 'topk')  # 'topk', 'block', 'butterfly', 'lowrank'
        self.base_sparsity = getattr(config, 'base_sparsity', 0.3)  # 基础稀疏度
        self.sparse_q = getattr(config, 'sparse_q', True)  # 是否对查询向量稀疏化
        self.sparse_k = getattr(config, 'sparse_k', True)  # 是否对键向量稀疏化

        # 块稀疏化参数
        self.block_size = getattr(config, 'block_size', 16)  # 稀疏块大小

        # 动态稀疏度参数
        self.use_dynamic_sparsity = getattr(config, 'use_dynamic_sparsity', False)

        # 可学习稀疏度参数
        self.use_learnable_sparsity = getattr(config, 'use_learnable_sparsity', False)
        if self.use_learnable_sparsity:
            self.q_sparsity = nn.Parameter(torch.tensor(self.base_sparsity))
            self.k_sparsity = nn.Parameter(torch.tensor(self.base_sparsity))

        # 低秩分解参数
        self.lowrank_ratio = getattr(config, 'lowrank_ratio', 0.25)
        self.lowrank_dim = max(int(self.head_dim * self.lowrank_ratio), 1)

        if self.sparsity_mode == 'lowrank':
            # 低秩分解投影
            self.q_lowrank = nn.Linear(self.head_dim, self.lowrank_dim, bias=False)
            self.k_lowrank = nn.Linear(self.head_dim, self.lowrank_dim, bias=False)

        # 量化参数
        self.use_quantization = getattr(config, 'use_quantization', False)
        self.quant_bits = getattr(config, 'quant_bits', 8)

        # 初始化蝶形稀疏模式
        if self.sparsity_mode == 'butterfly':
            self.butterfly_pattern = self._generate_butterfly_pattern()

    def _generate_butterfly_pattern(self):
        """生成蝶形稀疏模式"""
        dim = self.head_dim
        log_n = int(math.log2(dim))
        if 2 ** log_n != dim:
            # 如果dim不是2的幂，使用最接近的较小2的幂
            dim = 2 ** log_n

        pattern = torch.zeros(dim, dim, dtype=torch.bool)
        for i in range(log_n):
            block_size = 2 ** (i + 1)
            half_block = block_size // 2
            for j in range(0, dim, block_size):
                for k in range(half_block):
                    # 连接每个块的上半部分到两个区域
                    pattern[j + k, j:j + block_size] = True
                    # 连接每个块的下半部分到两个区域
                    pattern[j + half_block + k, j:j + block_size] = True

        # 如果原始维度不是2的幂，调整模式大小
        if self.head_dim != dim:
            full_pattern = torch.zeros(self.head_dim, self.head_dim, dtype=torch.bool)
            full_pattern[:dim, :dim] = pattern
            return full_pattern

        return pattern

    def _apply_topk_sparsity(self, x, sparsity, dim=-1):
        """应用TopK稀疏化"""
        k = max(int((1 - sparsity) * x.size(dim)), 1)  # 至少保留一个元素
        _, indices = torch.topk(x.abs(), k, dim=dim)
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask.scatter_(dim, indices, 1)
        return x * mask

    def _apply_block_sparsity(self, x, sparsity, block_size=None):
        """应用块稀疏化"""
        if block_size is None:
            block_size = self.block_size

        batch_size, seq_len, dim = x.shape

        # 确保block_size可以被维度整除
        if dim % block_size != 0:
            pad_size = block_size - (dim % block_size)
            x = torch.nn.functional.pad(x, (0, pad_size, 0, 0, 0, 0))
            dim = x.shape[-1]

        # 重组为块
        num_blocks = dim // block_size
        x_blocked = x.view(batch_size, seq_len, num_blocks, block_size)

        # 计算每个块的重要性（使用绝对值总和）
        block_importance = x_blocked.abs().sum(dim=-1)

        # 选择重要块
        k = max(int((1 - sparsity) * num_blocks), 1)  # 至少保留一个块
        _, selected_blocks = torch.topk(block_importance, k, dim=-1)

        # 创建块掩码
        block_mask = torch.zeros_like(block_importance, dtype=torch.bool)
        batch_indices = torch.arange(batch_size).view(-1, 1, 1).expand(-1, seq_len, k)
        seq_indices = torch.arange(seq_len).view(1, -1, 1).expand(batch_size, -1, k)
        block_mask[batch_indices, seq_indices, selected_blocks] = True

        # 扩展块掩码到元素级
        mask = block_mask.unsqueeze(-1).expand(-1, -1, -1, block_size)
        mask = mask.reshape(batch_size, seq_len, dim)

        return x * mask

    def _apply_butterfly_sparsity(self, x):
        """应用蝶形稀疏模式"""
        batch_size, seq_len, dim = x.shape
        pattern = self.butterfly_pattern
        if pattern.shape[0] != dim:
            # 如果维度不匹配，重新生成模式
            self.butterfly_pattern = self._generate_butterfly_pattern()
            pattern = self.butterfly_pattern

        # 将蝶形模式扩展到批次和序列维度
        expanded_pattern = pattern.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1)

        # 应用稀疏模式
        result = torch.zeros_like(x)
        for i in range(batch_size):
            for j in range(seq_len):
                # 对每个样本的每个位置使用蝶形模式进行矩阵乘法
                result[i, j] = torch.matmul(expanded_pattern[i, j], x[i, j])

        return result

    def _apply_lowrank(self, q, k):
        """应用低秩分解近似"""
        # 将查询和键投影到低维空间
        q_low = self.q_lowrank(q)
        k_low = self.k_lowrank(k)
        return q_low, k_low

    def _apply_quantization(self, x):
        """应用量化"""
        if not self.training and self.use_quantization:
            # 确定量化范围
            x_min = x.min()
            x_max = x.max()

            # 计算量化比例和零点
            scale = (x_max - x_min) / (2 ** self.quant_bits - 1)
            zero_point = (0 - x_min / scale).round().clamp(0, 2 ** self.quant_bits - 1)

            # 执行量化和反量化操作
            x_quant = ((x / scale) + zero_point).round().clamp(0, 2 ** self.quant_bits - 1)
            x_dequant = (x_quant - zero_point) * scale

            return x_dequant
        return x

    def _get_dynamic_sparsity(self, x, base_sparsity):
        """根据输入计算动态稀疏度"""
        if not self.use_dynamic_sparsity:
            return base_sparsity

        # 基于输入的方差确定稀疏度
        # 方差越大，稀疏度越低（保留更多信息）
        var = x.var(dim=-1).mean()

        # 映射方差到合理的稀疏度范围
        # 使用sigmoid确保稀疏度在[0.1, 0.9]范围内
        dynamic_sparsity = base_sparsity * (1.0 - torch.sigmoid(var * 10 - 5) * 0.5)
        return dynamic_sparsity.clamp(0.1, 0.9)

    def _get_sparsity(self, x, is_query=True):
        """获取实际使用的稀疏度"""
        if self.use_learnable_sparsity:
            # 使用可学习的稀疏度参数
            base_s = self.q_sparsity if is_query else self.k_sparsity
            base_s = torch.sigmoid(base_s).clamp(0.1, 0.9)  # 限制在合理范围内
        else:
            base_s = self.base_sparsity

        if self.use_dynamic_sparsity:
            # 根据输入动态调整稀疏度
            return self._get_dynamic_sparsity(x, base_s)

        return base_s

    def apply_sparsity(self, x, is_query=True):
        """根据选择的模式应用稀疏化"""
        sparsity = self._get_sparsity(x, is_query)

        if self.sparsity_mode == 'topk':
            return self._apply_topk_sparsity(x, sparsity)
        elif self.sparsity_mode == 'block':
            return self._apply_block_sparsity(x, sparsity)
        elif self.sparsity_mode == 'butterfly':
            return self._apply_butterfly_sparsity(x)
        elif self.sparsity_mode == 'lowrank':
            # 对于低秩模式，在forward中另外处理
            return x
        else:
            return x  # 默认不应用稀疏化

    def forward(self, q, k, v, mask_q=None, mask_kv=None):
        batch_size, seq_len, _ = q.shape

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

        # 根据设置进行量化
        if self.use_quantization:
            q = self._apply_quantization(q)
            k = self._apply_quantization(k)
            v = self._apply_quantization(v)

        # 重排张量维度为多头格式
        q_heads = rearrange(q, "b n (h c) -> (b h) n c", h=self.num_heads)
        k_heads = rearrange(k, "b n (h c) -> (b h) n c", h=self.num_heads)
        v_heads = rearrange(v, "b n (h c) -> (b h) n c", h=self.num_heads)

        # 应用稀疏化 - 根据选择的模式
        if self.sparse_q:
            if self.sparsity_mode == 'lowrank':
                q_heads_original = q_heads
                q_heads, k_heads_low = self._apply_lowrank(q_heads, k_heads)
                # 在低秩空间中计算注意力
                attention_weights = torch.einsum("b n c, b m c -> b n m", q_heads, k_heads_low)
                # 使用注意力权重将原始v投影
                y_hat = torch.einsum("b n m, b m d -> b n d", attention_weights, v_heads)
                # 跳过后续的标准注意力计算
                use_standard_attn = False
            else:
                q_heads = self.apply_sparsity(q_heads, is_query=True)
                use_standard_attn = True
        else:
            use_standard_attn = True

        if self.sparse_k and use_standard_attn:
            k_heads = self.apply_sparsity(k_heads, is_query=False)

        # 应用掩码
        if mask_q is not None:
            q_heads = q_heads * rearrange(mask_q, "b n -> (b h) n 1", h=self.num_heads)

        if mask_kv is not None:
            k_heads = k_heads * rearrange(mask_kv, "b n -> (b h) n 1", h=self.num_heads)
            v_heads = v_heads * rearrange(mask_kv, "b n -> (b h) n 1", h=self.num_heads)

        if use_standard_attn:
            # 标准注意力计算流程
            if mask_kv is not None:
                # 计算有效长度及均值
                mask_kv_heads = rearrange(mask_kv, "b n -> (b h) n 1", h=self.num_heads)
                length_k = torch.sum(mask_kv_heads, dim=1, keepdim=True)
                sqrt_length_k = torch.sqrt(length_k.clamp(min=self.eps))

                # 计算掩码下的加权平均
                k_sum = torch.sum(k_heads * mask_kv_heads, dim=1)
                mask_sum = torch.sum(mask_kv_heads, dim=1).clamp(min=self.eps)
                k_mean = k_sum / mask_sum
            else:
                # 无掩码情况
                length_k = k_heads.shape[1]
                sqrt_length_k = math.sqrt(length_k)
                k_mean = torch.mean(k_heads, dim=1)

            # 计算归一化因子的倒数
            z_inv = torch.einsum("b n d, b d -> b n", q_heads, k_mean) + self.eps
            z_inv = z_inv.unsqueeze(-1)

            # 计算键值对乘积
            kv = torch.einsum("b n c, b n d -> b c d", k_heads / sqrt_length_k, v_heads / sqrt_length_k)

            # 计算注意力输出
            y_hat = torch.einsum("b i c, b c d -> b i d", q_heads, kv)
            y_hat = y_hat / z_inv

        # 重排回原始维度
        y_hat = rearrange(y_hat, "(b h) n c -> b n (h c)", h=self.num_heads, b=batch_size)

        # 最终投影
        output = self.proj(y_hat)

        # 应用输出掩码
        if mask_q is not None:
            output = output.masked_fill(mask_q < 0.5, 0.0)

        return output

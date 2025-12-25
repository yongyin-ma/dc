import torch
import torch.nn as nn

class ScaleAdapt(nn.Module):
    def __init__(self, dim_in, config):
        """
        初始化 ScaleAdapt 模块。

        参数:
        - dim_in: 输入特征的维度大小。
         """
        super().__init__()
        # 初始化缩放参数 scale，不参与梯度更新
        self.register_buffer('scale', torch.ones(dim_in, dtype=torch.float))
        # self.scale = nn.Parameter(torch.ones(size=(dim_in,)), requires_grad=False)
        self.eps = config.eps  # 防止除零的小值
        self.move_rate = config.move_rate  # 滑动平均更新率
        self.affine = config.affine

        if self.affine:
            self.affine_layer = nn.Parameter(torch.ones(dim_in, dtype=torch.float) * self.affine)

    def forward(self, input_x, mask=None):
        """
        前向传播函数。

        参数:
        - input_x: 输入张量，形状为 (batch_size, length, dim)。
        - mask: 可选的掩码张量，形状与 input_x 相同。

        返回:
        - 输出张量，经过缩放和仿射变换处理。
        """
        if self.training:  # 训练模式
            if mask is None:  # 如果没有掩码
                # 计算批次和长度维度上的平均绝对值
                abs_x = torch.mean(torch.abs(input_x), dim=[0, 1]) + self.eps  # 按长度维度求均值
            else:  # 如果有掩码
                # 应用掩码并计算掩码下的均值
                abs_sum = torch.sum(torch.abs(input_x) * mask, dim=[0, 1])
                # 避免除零，使用 clamp 设置最小值
                mask_sum = torch.sum(mask, dim=[0, 1]).clamp(min=self.eps)
                abs_x = abs_sum / mask_sum

            # 使用滑动平均更新 scale 参数
                # 使用滑动平均更新 scale 参数
            with torch.no_grad():  # 确保不计算梯度
                self.scale.copy_(self.move_rate * abs_x + (1 - self.move_rate) * self.scale)

        # 对输入进行归一化，使用 view 确保广播维度正确
        normalized_x = input_x / (self.scale.view(1, 1, -1) + self.eps)

        # 如果启用了仿射变换，应用仿射层
        if self.affine:
            normalized_x = normalized_x * self.affine_layer.view(1, 1, -1)

        return normalized_x

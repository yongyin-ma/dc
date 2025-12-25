import torch.nn as nn
import torch
from utils.mask_input import mask_input

class DWC(nn.Module):
    """
    深度可分离卷积模块， 带有残差连接
    用于处理时序数据，支持掩码处理变长序列
    """
    def __init__(self, dim_in, dim_out,config):
        """
        初始化 DWC 模块。

        参数:
            dim_in: 输入特征维度
            dim_out: 输出特征维度
            config: 包含 DWC 参数的配置对象
        """
        super().__init__()
        ks = config.kernel_size # 卷积核大小
        padding = (ks - 1) // 2 # 计算填充大小，保持序列长度不变
        # 第一个卷积层
        self.dwc1 = nn.Conv1d(in_channels=dim_in, out_channels=dim_out, kernel_size=ks, padding=padding)
        # 激活函数
        self.relu = nn.LeakyReLU(inplace=True)
        # 第二个卷积层
        self.dwc2 = nn.Conv1d(in_channels=dim_out, out_channels=dim_out, kernel_size=ks, padding=padding)
        # 残差连接的卷积层，用于调整维度
        self.res = nn.Conv1d(in_channels=dim_in, out_channels=dim_out, kernel_size=ks, padding=padding)

    def forward(self, input_x: torch.Tensor, mask=None):
        """
        前向传播函数。

        参数:
            input_x: 输入张量，形状为 [batch_size, seq_len, dim_in]
            mask: 可选的掩码张量，形状与 input_x 相同

        返回:
            处理后的张量，形状为 [batch_size, seq_len, dim_out]
        """
        # 转置输入使通道维度在中间，以适应 Conv1d 的输入格式 [batch_size, channels, seq_len]
        output_x = input_x.transpose(-2, -1)

        # 处理掩码，如果提供了掩码
        mask_conv = None
        if mask is not None:
            # 转置掩码以匹配卷积输入的维度格式
            mask_conv = mask.transpose(-2, -1)

        # 计算残差项
        res_term = mask_input(self.res(output_x), mask_conv)

        # 主路径：第一个卷积 + 激活函数
        output_x = self.dwc1(output_x)
        output_x = self.relu(output_x)
        output_x = mask_input(output_x, mask_conv)

        # 主路径：第二个卷积
        output_x = self.dwc2(output_x)
        output_x = mask_input(output_x, mask_conv)

        # 添加残差连接
        output_x = output_x + res_term          # 残差连接

        # 转置输出以适应输出格式 [batch_size, seq_len, dim_out]
        output_x = output_x.transpose(-2, -1)

        return output_x

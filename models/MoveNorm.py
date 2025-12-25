import torch
import torch.nn as nn


class MoveNorm(nn.Module):
    def __init__(self, config):

        super().__init__()
        # 初始化统计参数和仿射参数
        self.register_buffer('stat_ave', torch.ones(1, dtype=torch.float))  # 不参与梯度更新的缓冲区参数
        self.affine_ave = nn.Parameter(torch.ones(1, dtype=torch.float))  # 可学习的仿射参数
        self.eps = config.eps
        self.move_rate = config.move_rate

    def forward(self, input_x, mask=None):
        # 训练模式下更新滑动平均
        if self.training:
            # 根据是否有掩码计算当前批次的平均值
            if mask is not None:
                # 计算掩码下的平均绝对值
                abs_sum = torch.sum(torch.abs(input_x) * mask)
                mask_sum = torch.sum(mask).clamp(min=self.eps)
                one_ave = abs_sum / mask_sum
            else:
                one_ave = torch.mean(torch.abs(input_x))

            # 使用原地操作高效更新统计平均值
            with torch.no_grad():
                self.stat_ave.mul_(1 - self.move_rate).add_(one_ave * self.move_rate)

        # 对输入进行归一化
        normalized_x = input_x / (torch.abs(self.stat_ave) + self.eps)

        # 应用仿射变换
        return normalized_x * self.affine_ave

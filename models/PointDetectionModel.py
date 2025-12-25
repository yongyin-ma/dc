import torch.nn as nn
import torch
from models.ScaleAdapt import ScaleAdapt
from models.DWC import DWC
from models.CrossAttention import CrossAttention
from models.SelfAttention import SelfAttention

class PointDetectionModel(nn.Module):
    def __init__(self, dim_in, config) -> None:
        super().__init__()
        # 归一化模块
        self.scale_adapt = ScaleAdapt(dim_in, config)

        # 深度可分离卷积层 (DWC)，用于特征提取
        self.dwc_l = nn.ModuleList([DWC(dim_in if i == 0 else config.dim_feature[i - 1],
                                    config.dim_feature[i],
                                    config)
                                for i in range(len(config.dim_feature))])

        # 最终特征维度
        dim_feature = config.dim_feature[-1]

        # 查询向量，作为解码器的初始输入
        self.query = nn.Parameter(torch.rand(size=(1, config.out_num, dim_feature)), requires_grad=True)

        # 编码器的自注意力层（用于输入特征）
        self.encoder_self = nn.ModuleList([
            SelfAttention(dim_feature, config)
            for _ in range(config.encoder_depth)
        ])

        # 解码器的自注意力层（查询自身）
        self.decoder_self = nn.ModuleList([
            SelfAttention(dim_feature, config)
            for _ in range(config.decoder_depth)
        ])

        # 解码器的交叉注意力层（查询 - 输入特征）
        self.decoder_cross = nn.ModuleList([
            CrossAttention(dim_feature, config)
            for _ in range(config.decoder_depth)
        ])

        # 最终输出层
        self.last_linear_class = nn.Linear(dim_feature, config.class_num)  # 分类预测
        self.last_linear_loc = nn.Linear(dim_feature, 1)  # 位置预测
        self.last_linear_vis = nn.Linear(dim_feature, 1)  # 可见性预测

        # 位置输出的限制范围
        self.loc_lim = config.loc_lim

    def forward(self, input_x, mask=None):
        """
        前向传播：
        1. 归一化输入
        2. 深度可分离卷积提取特征
        3. 编码器（输入特征自注意力）
        4. 解码器（查询的自注意力 + 交叉注意力）
        5. 位置、可见性、分类预测
        """
        batch, length, dim = input_x.shape[-3:]

        # 归一化输入
        input_x = self.scale_adapt(input_x, mask)

        # 复制查询向量，避免共享同一数据导致梯度更新错误
        batch_query = self.query.repeat(batch, 1, 1).clone()


        # 通过深度可分离卷积层提取特征
        for layer in self.dwc_l:
            input_x = layer(input_x, mask)

        # 编码器 - 输入特征的自注意力
        for i in range(len(self.encoder_self)):
            input_x = self.encoder_self[i](input_x, mask)  # Encoder (DWC SelfAttention)

        # 解码器阶段 - 查询自注意力 + 交叉注意力
        for i in range(len(self.decoder_self)):
            batch_query = self.decoder_self[i](batch_query)  # 查询的自注意力
            batch_query = self.decoder_cross[i](batch_query, input_x, None, mask) # 交叉注意力


        # 模型输出处理
        loc = self.loc_lim * torch.sigmoid(self.last_linear_loc(batch_query)) # 位置预测
        vis = torch.sigmoid(self.last_linear_vis(batch_query)) # 可见性预测
        classify = torch.softmax(self.last_linear_class(batch_query), dim=-1) # 分类预测
        return {"loc": loc, "vis": vis, "class": classify}

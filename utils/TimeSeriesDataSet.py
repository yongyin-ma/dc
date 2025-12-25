'''
@File    :   TimeSeriesDataSet.py
@Time    :   2024/6/9
@Author  :   shu.shiwei
@Version :   1.0
@Contact :   shu.shiwei@fdbatt.com
@License :   (C)Copyright 2024, FDB Corp.
@Desc    :   Brief description of the script or file purpose.
'''
import os
import random
import numpy as np
import pandas as pd
import torch
from sqlalchemy import create_engine, text
from torch.utils.data import Dataset, DataLoader


class TimeSeriesDataset(Dataset):
    def __init__(self, data_source_dir, end_str, aug=True):
        file_l = os.listdir(data_source_dir)
        self.file_l = []
        for index in range(len(file_l)):
            if file_l[index].endswith(end_str):
                self.file_l.append(os.path.join(data_source_dir,file_l[index]))
        self.COL_DIFFTIME = 0
        self.COL_CURRENT = 1
        self.COL_SOC = 2
        self.COL_VMIN = 3
        self.COL_VCELL = 4
        self.COL_TMIN = 5
        self.CAP_NOMINAL = 6
        self.DIFF_V = 7
        self.ACCUMULATED_ABS_AH = 8
        self.DELTA_AH = 9
        self.HTP_CELL = 10
        self.HTP_VMIN = 11
        self.AUG = 0
        self.dataframe = pd.DataFrame()


    def __getitem__(self, index):
        self.dataframe = pd.read_csv(self.file_l[index], header=None, skiprows=1)
        # 按照 \\ 分隔字符串
        parts = self.file_l[index].split("\\")  # 注意：Python 中需要用双反斜杠表示单个反斜杠
        # 获取分割后的最后一个部分
        last_part = parts[-1]
        # 取出最后一个部分的前17个字符
        vin = last_part[:25]
        self.dataframe[self.COL_DIFFTIME] = (-self.dataframe[self.COL_DIFFTIME] / 100).apply(np.exp)
        df_picked = pd.concat(
            [self.dataframe[[self.ACCUMULATED_ABS_AH]],self.dataframe[[self.COL_DIFFTIME]], self.dataframe[[self.COL_CURRENT]], self.dataframe[[self.COL_SOC]],
             self.dataframe[[self.COL_VMIN]], self.dataframe[[self.COL_VCELL]], self.dataframe[[self.COL_TMIN]], self.dataframe[[self.CAP_NOMINAL]],
             self.dataframe[[self.DIFF_V]]], axis=1)
        numpy_array = df_picked.to_numpy()
        tensor_input = torch.tensor(numpy_array)
        df_label_info = pd.concat([self.dataframe[[self.ACCUMULATED_ABS_AH]], self.dataframe[[self.HTP_CELL]], self.dataframe[[self.HTP_VMIN]]], axis=1)
        loc_label, class_label, visible_label = self.get_label_info(df_label_info)
        return tensor_input.float(), loc_label.float(), class_label.float(), visible_label.float(), vin
    def __len__(self):
        return len(self.file_l)

    def get_label_info(self,df_label):
        # 提供3个信息：拐点置/拐点类型/拐点是否可见
        vcell_htp_df = df_label[df_label[self.HTP_CELL] == 1]
        if vcell_htp_df.shape[0] > 0:
            # vcell高拐点可见
            vcell_abs_ah = [vcell_htp_df[self.ACCUMULATED_ABS_AH].iloc[0]] # vmin高拐点对应的绝对安时量累积
            class_vcell = [0, 1]
            vis_vcell = [1]
        else:
            # vmin高拐点不可见
            vcell_abs_ah = [0]
            class_vcell = [0, 1]
            vis_vcell = [0]

        # vmin高拐点部分
        vmin_htp_df = df_label[df_label[self.HTP_VMIN] == 1]
        if vmin_htp_df.shape[0] > 0:
            vmin_abs_ah = [vmin_htp_df[self.ACCUMULATED_ABS_AH].iloc[0]] # vmax高拐点对应的绝对安时量累积
            class_vmin = [1, 0]
            vis_vmin = [1]
        else:
            # vmax高拐点不可见
            vmin_abs_ah = [0]
            class_vmin = [1, 0]
            vis_vmin = [0]
        loc_label = torch.tensor([vmin_abs_ah, vcell_abs_ah])
        class_label = torch.tensor([class_vmin, class_vcell])
        visibel_label = torch.tensor([vis_vmin, vis_vcell])
        return loc_label, class_label, visibel_label


    
    @staticmethod
    def collate_fn(batch):
        # 找到batch中最长的序列
        max_length = max([x.shape[0] for x,loc_label,class_label,vis_label, vin in batch])
        features_dim = max([x.shape[1] for x,loc_label,class_label,vis_label, vin in batch])
        # batch_size是这个batch中样本的数量，max_length是所有样本中最大的长度，features_dim是所有样本中最大的特征数量
        batch_size = len(batch)
        padded_batch = torch.zeros(batch_size, max_length, features_dim)
        masks = torch.zeros(batch_size, max_length, 1)
        loc_label = torch.zeros(batch_size, 2, 1)
        class_label = torch.zeros(batch_size, 2, 2)
        vis_label = torch.zeros(batch_size, 2, 1)
        vin = []
        # 对每个序列进行padding
        for i, (input_, loc_label_, class_label_, vis_label_, vin_) in enumerate(batch):
            # 获取当前序列的长度
            seq_length = input_.shape[0]
            # 将当前序列填充到padded_batch中
            padded_batch[i, :seq_length, :] = input_
            masks[i,:seq_length,:] = 1
            loc_label[i,:] = loc_label_
            class_label[i,:] = class_label_
            vis_label[i,:] = vis_label_
            vin.append(vin_)
        return padded_batch, masks, loc_label, class_label, vis_label, vin


# if __name__ == '__main__':
#     # 从数据集中处理一些示例
#     for i in range(min(5, len(dataset))):
#         data,label = dataset[i]
#
#     # 调试collate_fn
#     batch = [dataset[i] for i in range(min(len(dataset),8))]
#     data, masks, label = TimeSeriesDataset.collate_fn(batch)








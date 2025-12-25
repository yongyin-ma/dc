import torch
from scipy.optimize import linear_sum_assignment
from torchvision.utils import save_image
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch.nn as nn


def loss_class_funct(class_pre, class_label):
    loss = nn.BCELoss(reduce="mean")(class_pre, class_label)
    return loss


def loss_visibel_funct(vis_pre, vis_label):
    loss = nn.MSELoss(reduce="mean")(vis_pre, vis_label)
    return loss


def loss_loc_funct(loc_pre, loc_label):
    loss = torch.pow((loc_label - loc_pre) / 300, 2)
    return loss


def safe_diou(interval_pre, interval_label, w_iou, w_dist):
    if (len(interval_pre) != 2) or (len(interval_label) != 2):
        return 0
    # 计算交集
    intersect_left = torch.max(interval_pre[0], interval_label[0])
    intersect_right = torch.min(interval_pre[1], interval_label[1])
    intersection = torch.clamp(intersect_right - intersect_left, min=0.0)

    # 计算并集
    union = (interval_pre[1] - interval_pre[0]) + (interval_label[1] - interval_label[0]) - intersection
    iou = intersection / (union + 1e-5) if union > 0 else 0.0
    iou_res = 1 - torch.pow(iou, 2)

    # 计算中心距离
    center_pre = (interval_pre[0] + interval_pre[1]) / 2.0
    center_label = (interval_label[0] + interval_label[1]) / 2.0
    center_dist = torch.abs(center_pre - center_label)
    center_cover_dist = torch.max(interval_pre[1], interval_label[1]) - torch.min(interval_pre[0], interval_label[0])
    dist_res = torch.pow(center_dist / (center_cover_dist + 1e-5), 2)

    # 计算diou最后结果
    diou_loss_res = w_iou * iou_res + w_dist * dist_res
    return diou_loss_res


def loss_all(loc_pre, class_pre, vis_pre, loc_label, class_label, vis_label):
    w_iou = 0.5
    w_dist = 0.5
    k_dtp = 1
    coeff_diou = 0.02

    device = loc_pre.device
    dtype = loc_pre.dtype
    batch_size = loc_label.shape[0]
    loss_loc = 0
    loss_class = 0
    loss_vis = 0
    loss_diou = 0
    # 每个数据做匹配算法
    for ii in range(batch_size):
        loc_pre_ii = loc_pre[ii].detach().cpu().numpy()  # arr: (2,1)
        loc_label_ii = loc_label[ii].detach().cpu().numpy()  # arr: (2,1)
        loc_label_ii = np.expand_dims(loc_label_ii, axis=-2)  # arr: (1,2,1) 维度扩展
        loc_pre_ii = np.expand_dims(loc_pre_ii, axis=-3)  # arr: (1,2,1) 维度扩展
        diff_mat = np.power(loc_pre_ii - loc_label_ii, 2)  # arr: (2,2,1) 广播计算实现2*2所有差值平方，加速计算避免for循环
        diff_mat = np.mean(diff_mat, axis=-1, keepdims=False)  # arr: (2,2) 去除最后一个维度

        # loss计算参数
        dtp = np.maximum(np.abs(loc_label_ii.flatten()[0] - loc_label_ii.flatten()[1]), 1)
        # k阶dtp归一化
        if k_dtp == 2:
            coeff_loc = 100 / np.power(dtp, 2)  # dtp归一化系数(power2)
        elif k_dtp == 1:
            coeff_loc = 20 / (dtp + 1e-5)  # dtp归一化系数(abs)
        elif k_dtp == 0:
            coeff_loc = 1

        row_ind, col_ind = linear_sum_assignment(diff_mat)  # 二分图匹配 (2*2是找最小对角线)
        # row_index 代表loc_label_ii的index, col_ind代表loc_pre_ii的index
        # 这里匹配了
        all_col_list = [ii for ii in range(loc_pre.shape[-2])]
        col_list = list(col_ind)
        rest_col_list = set(all_col_list).difference((set(col_list)))
        rest_col_list = list(rest_col_list)

        for jj in range(row_ind.shape[0]):

            row = row_ind[jj]
            col = col_ind[jj]
            # 如果不可见，不计算位置和分类的损失函数
            # print(loc_pre[ii][col].shape)
            # print(class_pre[ii][col].shape)
            # print(vis_pre[ii][col].shape)
            # exit(0)

            # print(vis_pre[ii][col],vis_label[ii][row])
            # print(vis_pre[ii][col],vis_label[ii][row])
            if loc_label[ii][row] != 0:
                loss_loc_ii = loss_loc_funct(loc_pre[ii][col], loc_label[ii][row])
                loss_class_ii = loss_class_funct(class_pre[ii][col], class_label[ii][row])
                loss_vis_ii = loss_visibel_funct(vis_pre[ii][col], vis_label[ii][row])
            else:
                loss_loc_ii = torch.zeros(size=[1]).to(device)
                loss_class_ii = torch.zeros(size=[1]).to(device)
                loss_vis_ii = loss_visibel_funct(vis_pre[ii][col], vis_label[ii][row])

            loss_loc = loss_loc + loss_loc_ii * coeff_loc  # loss设计(dtp归一化): 每个点计算都除以dtp进行归一化
            loss_class = loss_class + loss_class_ii
            loss_vis = loss_vis + loss_vis_ii

        interval_pre = sorted([loc_pre[ii][col] for col in col_ind])
        interval_label = sorted([loc_label[ii][row] for row in row_ind])  # loss设计(区间IOU): 每个点计算都除以dtp进行归一化

        # loss_iou = safe_iou(interval_pre, interval_label)
        # loss_loc = loss_loc + loss_iou * coeff_iou

        # loss_diou = loss_diou + safe_diou(interval_pre, interval_label)
        # loss_loc = loss_loc + loss_diou * coeff_diou
        loss_diou = loss_diou + safe_diou(interval_pre, interval_label, w_iou, w_dist) * coeff_diou

        for col in rest_col_list:
            loss_vis_ii = loss_visibel_funct(vis_pre[ii][col],
                                             torch.zeros_like(vis_pre[ii][col], device=device, dtype=dtype))
            loss_vis = loss_vis + loss_vis_ii
    return loss_loc / batch_size, loss_class / batch_size, loss_vis / batch_size, loss_diou / batch_size

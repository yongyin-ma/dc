import os
import sys
import warnings
import datetime
from datetime import timedelta
import random
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import numpy as np
from models.PointDetectionModel import PointDetectionModel
from models.Configs import Configs
from utils.safe_load import safe_load
from utils.get_loader import get_loader
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.loss_function import loss_all
from utils.plot_files import plot_tensor_data
from torch.optim.lr_scheduler import StepLR

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
warnings.filterwarnings("ignore")


# 设置随机种子
def set_seed(seed,n_gpu):
    # 设置Python自带的random库的种子
    random.seed(seed)
    # 设置Numpy的随机数生成器的种子
    np.random.seed(seed)
    # 设置PyTorch CPU 随机数生成的种子
    torch.manual_seed(seed)
    # 如果有可用的GPU，设置PyTorch的CUDA随机数生成器的种子
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

def valid(model, test_loader, epoch_count, num_steps, local_rank):
    """
    验证模型性能的函数。

    参数：
        model: 模型实例。
        test_loader: 测试数据加载器。
        epoch_count: 当前训练轮数。
        num_steps: 总训练步数。
        local_rank: 分布式训练中的本地排名（用于控制输出进度条）。

    返回：
        accuracy: 模型在验证集上的准确率。
    """
    # 将模型切换到评估模式，关闭 Dropout 和 BatchNorm 的训练行为

    model.eval()
    # 初始化变量以存储验证损失和预测结果
    eval_losses = 0
    eval_step = 0
    all_logits, all_labels = [], []

    # 创建一个 tqdm 进度条，用于显示验证过程的进度
    epoch_iterator = tqdm(
        test_loader,
        desc="Validating... (loss=X.X)",  # 初始描述信息
        bar_format="{l_bar}{r_bar}",  # 进度条格式
        dynamic_ncols=True,  # 动态调整列宽
        disable=local_rank not in [-1, 0]  # 控制是否禁用进度条（分布式训练时）
    )


    # 使用 torch.no_grad() 禁用梯度计算，减少内存占用并加速推理
    for step, batch in enumerate(epoch_iterator):
        # Validation!
        time_series_data, mask, loc_label, class_label, vis_label, vin = batch
        eval_step += 1
        x = time_series_data.to(device)
        mask = mask.to(device)
        loc_label = loc_label.to(device)
        class_label = class_label.to(device)
        vis_label = vis_label.to(device)
        with torch.no_grad():
        # 前向传播：通过模型获取预测值 logits 和解码值 decoded
            logits = model(x, mask)
            loc_pre = logits['loc']
            vis_pre = logits['vis']
            class_pre = logits['class']
            # 计算多种损失项
            loss_loc, loss_class, loss_vis, loss_diou = loss_all(loc_pre, class_pre, vis_pre, loc_label, class_label,
                                                                 vis_label)
            eval_loss = loss_loc + loss_class + loss_vis + loss_diou
            eval_losses += eval_loss  # 累加当前批次的损失值
            # 更新进度条描述信息，显示当前轮次、总轮次以及当前批次的损失值
            epoch_iterator.set_description(
                "Validating... (%d / %d Epoch)(loss=%2.5f)" % (epoch_count, num_steps, eval_loss.item())
            )

            # 每 10 步绘制一次张量数据（可视化）
            # if (step + 1) % 10 == 1:
            #     plot_tensor_data(x, loc_label, class_label, vis_label, loc_pre, vis_pre, class_pre, mask, vin, batch_index=0)

        # 将 logits 和 label 移动到 CPU 并转换为 numpy 数组
        # x = x.cpu.numpy()
        # loc_label = loc_label.cpu().numpy()  # 同样将label转换为numpy数组并移到CPU上
        # class_label = class_label.cpu().numpy()
        # vis_label = vis_label.cpu().numpy()
        #
        # # 移除 logits 和 label 最后一个维度（大小为 1）
        # logits_squeezed = logits.squeeze(axis=2)
        # loc_label_squeezed = loc_label.squeeze(axis=2)
        # class_label_squeezed = class_label.squeeze(axis=2)
        # vis_label_squeezed = vis_label.squeeze(axis=2)
        #
        # # 将每个样本的 logits 和 label 转换为列表并追加到全局存储中
        # all_logits.extend(logits_np.tolist())
        # all_labels.extend(label_np.tolist())

        # 计算平均验证损失
    avg_eval_loss = eval_losses / len(test_loader)

    # 将平均损失记录到 TensorBoard 中
    writer.add_scalar('Loss/validating', avg_eval_loss, epoch_count)

    # 计算模型在验证集上的准确率
    # accuracy = calculate_accuracy(all_logits, all_labels)

    # return accuracy  # 返回验证集上的准确率


if __name__ == '__main__':
    # 设定模型名称
    model_name = 'discharge_turning_point_detection'
    # dataset 存放地址
    csv_dir = os.path.join(BASE_DIR, "dataset")
    # 设置学习率
    lr = 3e-5
    # 设置中间显示步数
    show_step = 5
    # 设置训练代数
    all_epoch = 20000
    # 模型保存步数
    save_step = 100
    # 设定local_rank通常用来标识当前进程在当前节点（可以是一台物理机或虚拟机）内的编号，默认-1代表用来表示程序不是以分布式训练模式运行
    local_rank = -1
    # 模型保存路径
    model_path = os.path.join(BASE_DIR, "checkpoint/discharge_turning_point_detection.pth")
    # 设置随机种子的个数
    seed = 42
    # 设置模型类型
    model_type = "key_point_detr"
    # 设置模型输入的参数个数
    model_dim_input = 9
    # 设置训练和验证的batch size
    train_batch_size = 32
    eval_batch_size = 32
    # 设置weight decay
    weight_decay = 0
    # 设置 decay type
    decay_type = "cosine"
    # 设置 warmup steps
    warmup_steps = 500
    # 设置 Total number of training epochs to perform
    num_steps = 20000
    # 设置 gradient_accumulation_steps
    gradient_accumulation_steps = 1
    # 设置 Run prediction on validation set every so many steps
    eval_every = 100
    # 初始化TensorBoard writer
    tensorboard_dir = os.path.join(BASE_DIR, "tensorboard")
    writer = SummaryWriter(tensorboard_dir)

    # end para
    # ----------------
    # 配置显卡
    # Setup CUDA, GPU & distributed training
    if local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        torch.distributed.init_process_group(backend='nccl', timeout=timedelta(minutes=60))
        n_gpu = 1
    # 设置随机种子
    set_seed(seed,n_gpu)
    # Prepare model
    config = Configs()
    # 初始化一个FlattenTransformer模型实例
    model = PointDetectionModel(model_dim_input, config)
    # model = safe_load(model, model_path)
    model.to(device)
    # train model
    data_source_dir = r'D:\development_projects\deep_learning\04_detect_tp\dch_aipd_003\dataset\dataset_source'
    train_loader, test_loader = get_loader(local_rank, train_batch_size, eval_batch_size,data_source_dir)
    # prepare optimizer and scheduler
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)  # L2的系数
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)  # L2
    # 优化器设置
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=150, gamma=0.75)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    t_total = num_steps
    # if decay_type == "cosine":
    #     scheduler = WarmupCosineSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)
    # else:
    #     scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)

    # 将模型的所有梯度设置为零
    model.zero_grad()
    set_seed(seed, n_gpu)
    global_step, best_acc = 0, 0
    epoch_count = 0
    batch_count = 0
    while True:
        model.train()
        epoch_count += 1
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X epoch) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=local_rank not in [-1, 0])
        losses = 0

        for step, batch in enumerate(epoch_iterator):
            batch_count = batch_count + 1

            input_x, mask, loc_label, class_label, vis_label, vin = batch

            x = input_x.to(device)
            mask = mask.to(device)
            loc_label = loc_label.to(device)
            class_label = class_label.to(device)
            vis_label = vis_label.to(device)

            pred = model(x, mask)
            loc_pre = pred['loc']
            vis_pre = pred['vis']
            class_pre = pred['class']

            # plot
            # if (step + 1) % 10 == 1:
            #     plot_tensor_data(x, label, pred, mask, packid,batch_index=0)

            # 计算损失
            loss_loc, loss_class, loss_vis, loss_diou = loss_all(loc_pre, class_pre, vis_pre, loc_label, class_label,
                                                                 vis_label)
            loss = loss_loc + loss_class + loss_vis + loss_diou

            # 求出梯度并且放入优化器
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scheduler.step()
            optimizer.step()
            optimizer.zero_grad()
            print("epoch",epoch_count, "batch_index:", batch_count, f"loss_loc: {loss_loc.item(): .4f}", f"loss_diou: {loss_diou.item(): .4f}",
                "loss_class", loss_class.item(), "loss_vis", loss_vis.item())
            losses += loss
            global_step += 1
            epoch_iterator.set_description(
                "Training (%d / %d Epoch)(losses=%2.5f)" % (epoch_count, num_steps, loss.item())
            )
            writer.add_scalar('Loss/train', loss.item(), batch_count)
            writer.add_scalar("location loss", loss_loc.item(), batch_count)
            writer.add_scalar("DIoU loss", loss_diou.item(), batch_count)
            writer.add_scalar("class loss", loss_class.item(), batch_count)
            writer.add_scalar("visibility loss", loss_vis.item(), batch_count)
        losses = losses / len(epoch_iterator)

        # 记录每个step的loss到TensorBoard
        valid(model, test_loader, epoch_count, num_steps, local_rank)
        # print(f'accuracy is {accuracy}%')
        # if best_acc < accuracy:
        #     torch.save(model, f'./checkpoint/{model_name}.pth')
        #     best_acc = accuracy

        if global_step % t_total == 0:
            break
    # 关闭writer
    writer.close()

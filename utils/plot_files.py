import datetime
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from datetime import datetime

# 定义一个函数来找到符合条件的索引
def find_indices(values, target_array):
    indices = []
    for value in values:
        # 找到第一个大于等于 value 的索引
        idx = np.where(target_array >= value)[0]
        if len(idx) > 0:  # 确保找到了符合条件的索引
            indices.append(idx[0])
        else:
            indices.append(-1)  # 如果没有找到，填充 -1
    return indices

def plot_tensor_data(x, loc_label, class_label, vis_label, loc_pre, vis_pre, class_pre, mask, vin, file_dir, batch_index=0):
    """
    绘制tensor数据。
    参数：
    x: 三维tensor，包含输入数据，形状为(batchsize, time_length, features)
    label: 三维tensor，包含标签数据，形状为(batchsize, time_length, 1)
    predict: 三维tensor，包含预测数据，形状为(batchsize, time_length, 1)
    batch_index: int，指定要绘制的batch的索引
    """
    try:
    # 提取指定batch的数据
        mask = mask[batch_index].detach().to('cpu').numpy()
        x_batch = x[batch_index].detach().to('cpu').numpy()
        loc_label_batch = loc_label[batch_index].detach().to('cpu').squeeze().numpy()
        class_label_batch = class_label[batch_index].detach().to('cpu').squeeze().numpy()
        vis_label_batch = vis_label[batch_index].detach().to('cpu').squeeze().numpy()
        loc_pre_batch = loc_pre[batch_index].detach().to('cpu').squeeze().numpy()
        class_pre_batch = class_pre[batch_index].detach().to('cpu').squeeze().numpy()
        vis_pre_batch = vis_pre[batch_index].detach().to('cpu').squeeze().numpy()
        vin = vin[batch_index]
        vin = vin.split('/')[-1]

        # 使用mask筛选有效数据
        flag = mask == 1
        masked_x_batch = x_batch[flag.flatten() == 1]

        time_length = masked_x_batch.shape[0]
        x_time = range(time_length)

        # 提取x的第二列和第三列
        current = masked_x_batch[:, 2]
        soc = masked_x_batch[:, 3]
        vcell = masked_x_batch[:, 5]
        vmin = masked_x_batch[:, 4]
        t_min = masked_x_batch[:, 6]
        diff_v = masked_x_batch[:, 8] * 1000
        accumulated_abs_ah = masked_x_batch[:, 0]

        # 找到 loc_label_batch 和 loc_pre_batch 对应的索引
        label_indices = find_indices(loc_label_batch, accumulated_abs_ah)
        pre_indices = find_indices(loc_pre_batch, accumulated_abs_ah)

        # 提取对应的 soc 和 accumulated_soc_ah
        label_soc = [soc[i] if i != -1 else np.nan for i in label_indices]
        label_accumulated = [accumulated_abs_ah[i] if i != -1 else np.nan for i in label_indices]
        pre_soc = [soc[i] if i != -1 else np.nan for i in pre_indices]
        pre_accumulated = [accumulated_abs_ah[i] if i != -1 else np.nan for i in pre_indices]

        # 创建图像
        fig, axes = plt.subplots(3, 1, figsize=(20, 12), sharex=True)

        # 第一个子图：soc vs vcell 和 vmin
        axes[0].plot(soc, vcell, label='vcell', color='blue')
        axes[0].plot(soc, vmin, label='vmin', color='green')
        axes[0].set_title('SOC vs Vcell and Vmin')  # 英文标题
        axes[0].set_ylabel('Voltage (V)')
        axes[0].set_ylim(3.1, 3.5)  # 固定纵坐标范围
        axes[0].legend()
        axes[0].grid(True, which='major', linestyle='-', linewidth=0.8)  # 主网格
        axes[0].grid(True, which='minor', linestyle='--', linewidth=0.5)  # 次网格
        axes[0].xaxis.set_minor_locator(MultipleLocator(1))  # 每个 SOC 一个子刻度

        # 第二个子图：soc vs diff_v，并添加点
        y_level = 0  # 设置水平线的纵坐标
        axes[1].plot(soc, diff_v, label='diff_v', color='orange')
        axes[1].axhline(y=y_level, color='gray', linestyle='--', linewidth=1)  # 绘制水平线
        axes[1].scatter(label_soc, [y_level] * len(label_soc), facecolors='none', edgecolors='blue', s=100,
                        label='Label Points (Blue)')
        axes[1].scatter(pre_soc, [y_level] * len(pre_soc), color='red', s=100, label='Predicted Points (Red)')
        axes[1].set_title('SOC vs Diff_V and Key Points')
        axes[1].set_ylabel('Diff_V (mV) / Fixed Level')
        axes[1].set_ylim(-5, 60)  # 固定纵坐标范围
        axes[1].legend()
        axes[1].grid(True, which='major', linestyle='-', linewidth=0.8)  # 主网格
        axes[1].grid(True, which='minor', linestyle='--', linewidth=0.5)  # 次网格
        axes[1].xaxis.set_minor_locator(MultipleLocator(1))  # 每个 SOC 一个子刻度

        # 第三个子图：soc vs current 和 t_min（双坐标轴）
        ax_twin = axes[2].twinx()  # 创建双坐标轴
        axes[2].plot(soc, current, label='Current', color='purple')
        ax_twin.plot(soc, t_min, label='Min Temperature', color='darkorange')  # 更改颜色为高饱和度
        axes[2].set_title('SOC vs Current and Min Temperature')
        axes[2].set_xlabel('SOC')
        axes[2].set_ylabel('Current (A)', color='purple')
        ax_twin.set_ylabel('Min Temperature (°C)', color='darkorange')
        axes[2].set_ylim(-100, 100)  # 固定电流范围
        ax_twin.set_ylim(-10, 40)  # 固定温度范围
        axes[2].tick_params(axis='y', colors='purple')  # 设置左侧 y 轴颜色
        ax_twin.tick_params(axis='y', colors='darkorange')  # 设置右侧 y 轴颜色
        axes[2].legend(loc='upper left')
        ax_twin.legend(loc='upper right')
        axes[2].grid(True, which='major', linestyle='-', linewidth=0.8)  # 主网格
        axes[2].grid(True, which='minor', linestyle='--', linewidth=0.5)  # 次网格
        axes[2].xaxis.set_minor_locator(MultipleLocator(1))  # 每个 SOC 一个子刻度


        # 公共设置：反转 x 轴、固定范围
        for ax in axes:
            ax.invert_xaxis()  # 反转 x 轴
            ax.set_xlim(100, 40)  # 固定 x 轴范围

        # 调整布局
        plt.tight_layout()

         # 获取当前时间戳并格式化
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 构造文件名和文件路径
        file_name = f"{vin}_{timestamp}.png"
        file_path = os.path.join(file_dir, file_name)

        # 保存图像
        plt.savefig(file_path, dpi=300, bbox_inches='tight')  # 保存为高分辨率 PNG 文件

        # 显示图像
        # plt.show()
    except Exception as e:
        print(f"Error processing batch_index {batch_index}: {e}")

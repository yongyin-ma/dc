import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # 进程融合避免画图报错
import numpy as np
from itertools import chain
import os
import pandas as pd
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

def plot_dch_env(tensor, vin_cell_dch, save_path, title_description =''):
    tensor = tensor.cpu().detach().numpy()
    fig = plt.figure(figsize=(15, 10))
    # model in n out
    grid = plt.GridSpec(21, 10, wspace=0.15, hspace=0.15)

    is_not_pad = tensor[:, 0]!=0
    tensor = tensor[is_not_pad,:]
    # I
    ax1 = plt.subplot(grid[0:5, 0:10])
    ax1.plot(tensor[:, 0], tensor[:, 1] , color='mediumorchid', label='I')
    mean_i = pd.Series(tensor[:, 1]).rolling(window=30, min_periods=1, center=True).mean()
    ax1.plot(tensor[:, 0], mean_i, color='indigo', label='ave_I')
    ax1.set_ylabel(ylabel='Current')
    ax1.set_ylim([-20, 80])
    #ax1.set_xlim([100, 50])
    ax1.legend(loc ='upper right')
    ax1.set_title(
        f"{vin_cell_dch[0]} #{(vin_cell_dch[1])} \n {title_description}")
    # Temp
    ax1_right = ax1.twinx()
    ax1_right.plot(tensor[:, 0], tensor[:, 4], color='y', label='Temperature')
    ax1_right.legend(loc='upper left')
    ax1_right.set_ylim([0, 50])
    ax1_right.set_ylabel(ylabel='min T')
    # vdiff
    ax2 =plt.subplot(grid[5:10, 0:10], sharex=ax1)
    ax2.plot(tensor[:, 0], tensor[:, 2], color='black', label="Envelop (V)")
    ax2.set_ylabel(ylabel='$V_{cell}$' + f'{vin_cell_dch[1]} - ' + '$V_{min}$ ')
    ax2.set_ylim([0, 0.04])
    ax2.legend(loc ='upper right')
    # label ypred
    ax3 = plt.subplot(grid[10:15, 0:10], sharex=ax1)
    ax3.plot(tensor[:, 0], tensor[:, -2], label='M01_label')
    ax3.plot(tensor[:, 0], tensor[:, -1], label='y_pred')
    ax3.set_ylabel(ylabel='Label')
    ax3.set_xlabel(xlabel='SOC')
    ax3.legend(loc ='upper right')
    ax3.set_ylim([0, 1])
    #vmax vmin
    ax7 = plt.subplot(grid[15:20, 0:10], sharex=ax1)
    ax7.plot(tensor[:, 0], tensor[:, -3], color='red', label='$V_{cell}$' + f'-{vin_cell_dch[1]} ')
    ax7.plot(tensor[:, 0], tensor[:, -4], color='blue', label='$V_{min}$ ')
    ax7.set_ylabel(ylabel='V')
    ax7.set_xlabel(xlabel='SOC')
    ax7.legend(loc='upper right')
    ax7.set_ylim([3.2, 3.4])
    fig.savefig(save_path)
    plt.clf()
    plt.close()

def get_soc_loc(df, loc_list):
    pred, label = loc_list

    pred0 = df.loc[(df['abs_ah'] > float(pred[0])).idxmax(), 'soc'] # 这里的意义其实是abs_ah
    pred1 = df.loc[(df['abs_ah'] > float(pred[1])).idxmax(), 'soc'] # 这里的意义其实是abs_ah
    label0 = df.loc[(df['abs_ah'] > float(label[0])).idxmax(), 'soc'] # 这里的意义其实是abs_ah
    label1 = df.loc[(df['abs_ah'] > float(label[1])).idxmax(), 'soc'] # 这里的意义其实是abs_ah

    return np.array([pred0, pred1, label0, label1])

def plot_dch_only_mp(df, res_list, vin_cell_dch, save_path ='' ,have_pad_frame = True, title_description =''  ):
    plt.rcParams['font.sans-serif'] = ['AR PL UKai CN']  # 使用 Noto Sans CJK SC 字体（简体中文）
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题
    pred, visible, cls, label = res_list


    pred0 = df.loc[(df['abs_ah'] > float(pred[0])).idxmax(), 'soc'] # 这里的意义其实是abs_ah
    pred1 = df.loc[(df['abs_ah'] > float(pred[1])).idxmax(), 'soc'] # 这里的意义其实是abs_ah
    cls0 = 'Vmin拐点' if float(cls[0, 0]) > float(cls[0, 1]) else 'Vmax拐点'
    cls1 = 'Vmin拐点' if float(cls[1, 0]) > float(cls[1, 1]) else 'Vmax拐点'

    color_map = {'Vmax拐点': 'red', 'Vmin拐点': 'blue'}


    label0 = df.loc[(df['abs_ah'] > float(label[0])).idxmax(), 'soc']
    label1 = df.loc[(df['abs_ah'] > float(label[1])).idxmax(), 'soc']


    fig = plt.figure(figsize=(15, 10))
    grid = plt.GridSpec(21, 10, wspace=0.15, hspace=0.15)
    # I
    ax1 = plt.subplot(grid[0:5, 0:10])
    ax1.plot(df['soc'], df['i'] , color='mediumorchid', label='I')
    mean_i = pd.Series(df['i']).rolling(window=30, min_periods=1, center=True).mean()
    ax1.plot(df['soc'], mean_i, color='indigo', label='ave_I')
    if have_pad_frame:
        ax1.set_ylabel(ylabel='Current')
    else:
        ax1.tick_params(axis='y', direction='in')

    ax1.set_ylim([-20, 80])
    ax1.set_xlim([40 , 100])
    ax1.legend(loc ='upper right')
    title_list = title_description.split('_')
    ax1.set_title(
        f"{title_list}")
    # ax1.set_title(
    #     f"{vin_cell_dch[0]} #{(vin_cell_dch[1])} \n {title_description}")

    # Temp
    ax1_right = ax1.twinx()
    ax1_right.plot(df['soc'], df['temperature'], color='y', label='Temperature')
    ax1_right.legend(loc='upper left')
    ax1_right.set_ylim([0, 50])
    ax1_right.set_ylabel(ylabel='min T')
    # vdiff
    ax2 = plt.subplot(grid[5:10, 0:10], sharex=ax1)
    ax2.plot(df['soc'], df['vdiff'], color='black', label="Envelop (V)")
    # ax2.plot(df['soc'], df['label'], color='blue', label="AIDE label")
    ax2.set_ylabel(ylabel='$V_{cell}$' + f'{vin_cell_dch[1]} - ' + '$V_{min}$ ')
    ax2.set_ylim([0, 0.04])
    ax2.legend(loc ='upper right')
    # label ypred
    ax3 = plt.subplot(grid[10:15, 0:10], sharex=ax1)
    ax3.plot([min(df['soc']),max(df['soc'])], [0.5, 0.5], '-.r', label='0.5 threshold')
    ax3.plot(pred0, 0.5, marker='x', ms=15, color=color_map[cls0], label=f'{cls0} Visibility: {float(visible[0]): .2f} Class Probability: {float(cls[0, 0]): .2f}, {float(cls[0, 1]): .2f}')
    ax3.plot(pred1, 0.5, marker='x', ms=15, color=color_map[cls1], label=f'{cls1} Visibility: {float(visible[1]): .2f} Class Probability: {float(cls[1, 0]): .2f}, {float(cls[1, 1]): .2f}')
    ax3.plot(label0, 0.5, marker='o', ms=5, color='blue', label=f'Vmin拐点标签')
    ax3.plot(label1, 0.5, marker='o', ms=5, color='red', label=f'Vmax拐点标签')


    ax3.set_ylabel(ylabel='Label')
    ax3.set_xlabel(xlabel='SOC')
    ax3.legend(loc ='upper left')
    ax3.set_ylim([-0.05, 1.05])
    #vmax vmin
    ax7 = plt.subplot(grid[15:20, 0:10], sharex=ax1)
    ax7.plot(df['soc'], df['vcell'], color='red', label='$V_{cell}$' + f'-{vin_cell_dch[1]} ')
    ax7.plot(df['soc'], df['vmin'], color='blue', label='$V_{min}$ ')
    ax7.set_ylabel(ylabel='Volatge')
    ax7.set_xlabel(xlabel='SOC')
    ax7.legend(loc='upper right')
    ax7.set_ylim([3.2, 3.4])
    if save_path != '':
        if have_pad_frame:
            fig.savefig(save_path + '.jpg', bbox_inches='tight', pad_inches=0.05, dpi=300)
        else:
            plt.subplots_adjust(right=0.999, left=0.001 )
            fig.savefig(save_path + '.jpg')
        # print(save_path)
    else:
        plt.show()
    plt.clf()
    plt.close()
    return [pred0, pred1, label0, label1]

def plot_c_dc_and_save(tensor, vin_cell_dch, save_path ='',is_tensor = True,have_pad_frame = True, title_description ='',save_tensor = True,  dtp_m01=[], charge_df=pd.DataFrame,capnominal = 0):
    '''

    :param tensor:
    :param vin_cell_dch: batch info
    :param save_path: 保存路径
    :param is_tensor: tensor传参是否是Torch.tensor 还是 numpy。array
    :param have_pad_frame: 是否有边框
    :param title_description: 加在标题后的数字
    :param save_tensor: 是否保存画图的tensor
    :param dtp_m01: 当前电芯拐点SOC与最低电芯拐点SOC
    :param charge_df: 该电芯充电的电压与最低电压
    :return:
    '''
    if is_tensor:
        tensor = tensor.cpu().detach().numpy()
    else:
        tensor = np.array(tensor)
        tensor = tensor[:,list(range(1,11))]#去掉index列
    fig = plt.figure(figsize=(15, 10))
    # model in n out
    grid = plt.GridSpec(26, 10, wspace=0.15, hspace=0.15)
    is_not_pad = tensor[:, 0]!=0
    tensor = tensor[is_not_pad,:]
    # I
    ax1 = plt.subplot(grid[0:5, 0:10])
    ax1.plot(tensor[:, 0], tensor[:, 1] , color='mediumorchid', label='I')
    mean_i = pd.Series(tensor[:, 1]).rolling(window=30, min_periods=1, center=True).mean()
    ax1.plot(tensor[:, 0], mean_i, color='indigo', label='ave_I')
    if have_pad_frame:
        ax1.set_ylabel(ylabel='Current')
    else:
        ax1.yaxis.set_visible(False)
    ax1.set_ylim([-20, 80])
    ax1.set_xlim([40 , 100])
    ax1.legend(loc ='upper right')
    ax1.set_title(
        f"{vin_cell_dch[0]} #{(vin_cell_dch[1])} \n {title_description}")
    # Temp
    ax1_right = ax1.twinx()
    ax1_right.plot(tensor[:, 0],tensor[:, 4], color='y', label='Temperature')
    ax1_right.legend(loc='upper left')
    ax1_right.set_ylim([0, 50])
    ax1_right.set_ylabel(ylabel='min T')
    # vdiff
    ax2 =plt.subplot(grid[5:10, 0:10], sharex=ax1)
    ax2.plot(tensor[:, 0], tensor[:, 2], color='black', label="Envelop (V)")
    ax2.set_ylabel(ylabel='$V_{cell}$' + f'{vin_cell_dch[1]} - ' + '$V_{min}$ ')
    ax2.set_ylim([0, 0.04])
    ax2.legend(loc ='upper right')
    # label ypred
    ax3 = plt.subplot(grid[10:15, 0:10], sharex=ax1)
    ax3.plot(tensor[:, 0], tensor[:, -2], label='Label')
    ax3.plot(tensor[:, 0], tensor[:, -1], label='y_pred')
    ax3.plot([dtp_m01[0], dtp_m01[0]], [-0.05, 1.05], '.-b', label='M01 Result')
    ax3.plot([dtp_m01[1], dtp_m01[1]], [-0.05, 1.05], '.-b')
    ax3.plot([min(tensor[:, 0]),max(tensor[:, 0])],[0.5,0.5],'-.r',label='0.5 threshold')
    ax3.set_ylabel(ylabel='Label')
    ax3.set_xlabel(xlabel='SOC')
    ax3.legend(loc ='upper right')
    ax3.set_ylim([-0.05, 1.05])
    #vmax vmin
    ax7 = plt.subplot(grid[15:20, 0:10], sharex=ax1)
    ax7.plot(tensor[:, 0], tensor[:, -3], color='red', label='$V_{cell}$' + f'-{vin_cell_dch[1]} ')
    ax7.plot(tensor[:, 0], tensor[:, -4], color='blue', label='$V_{min}$ ')
    ax7.set_ylabel(ylabel='Volatge')
    ax7.set_xlabel(xlabel='SOC')
    ax7.legend(loc='upper right')
    # charge vmin vmax
    #new_soc = charge_df['SOC'].values


    ax8 = plt.subplot(grid[20:25, 0:10], sharex=ax1)
    ax8.plot(charge_df['SOC'],charge_df.iloc[:,-1], label='$charge V_{min}$' )
    ax8.plot(charge_df['SOC'], charge_df.iloc[:,-2], label='$charge V_{min}$'+ f'-{vin_cell_dch[1]} ')
    ax8.plot([dtp_m01[0], dtp_m01[0]], [3.3, 3.45],'.-b', label='M01 Result')
    ax8.plot([dtp_m01[1], dtp_m01[1]], [3.3, 3.45],'.-b')
    ax8.legend(loc='upper right')
    ax8.set_ylim([3.3,3.45])

    if save_path != '':
        save_path = save_path + f'_-_{capnominal}_-_-_-_-_-'
        if have_pad_frame:
            fig.savefig(save_path + '.jpg', bbox_inches='tight', pad_inches=0.05, dpi=300)
        else:
            plt.subplots_adjust(right=0.999, left=0.001 )
            fig.savefig(save_path + '.jpg')
    if save_tensor:
        df = pd.DataFrame(tensor)
        df = df.rename(columns={0: 'soc', 1: 'i', 2: 'vdiff', 3: 'time', 4: 'temperature', 5: 'ah', 6: 'vmin', 7: 'vcell', 8: 'label', 9:'model_prediction'})
        df['cap_nominal'] = capnominal
        new_df = df[['time',  'temperature',  'soc',  'i',  'vdiff',  'ah',  'cap_nominal',  'label',  'vmin',  'vcell', 'model_prediction']]
        new_df['ah'] = new_df['ah'] * capnominal
        new_df.to_csv(save_path + '.csv')
    else:
        plt.show()
    plt.clf()
    plt.close()

def plot_joint_tensor(tensor, vin_cell_dch, df, save_path,var_monte = None,show=False):
    #  ["soc", "i", "vdiff", "time", "temperature", "ah", 'vmin', 'vcell']
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
    plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题
    fig = plt.figure(figsize=(15, 10))
    # model in n out
    grid = plt.GridSpec(21, 22, wspace=0.15, hspace=0.15)
    tensor = tensor.cpu().detach().numpy()
    is_not_pad = tensor[:, 0]!=0
    tensor = tensor[is_not_pad,:]
    if var_monte is not None:
        var_monte = var_monte.cpu().detach().numpy()
        var_monte = var_monte[is_not_pad ]
    # I
    ax1 = plt.subplot(grid[0:5, 0:10])
    ax1.plot(tensor[:, 0], tensor[:, 1] , color='mediumorchid', label='I')
    mean_i = pd.Series(tensor[:, 1]).rolling(window=30, min_periods=1, center=True).mean()
    ax1.plot(tensor[:, 0], mean_i, color='indigo', label='均值I')
    ax1.set_ylabel(ylabel='Current')
    ax1.set_ylim([-20, 80])
    #ax1.set_xlim([100, 50])
    ax1.legend(loc ='upper right')
    ax1.set_title(
        f"{vin_cell_dch[0]} #{(vin_cell_dch[1])}节 \n MODEL IN AND OUT")
    # Temp
    ax1_right = ax1.twinx()
    ax1_right.plot(tensor[:, 0], tensor[:, 4], color='y', label='温度')
    ax1_right.legend(loc='upper left')
    ax1_right.set_ylim([0, 50])
    ax1_right.set_ylabel(ylabel='最低温度')
    # vdiff
    ax2 =plt.subplot(grid[5:10, 0:10], sharex=ax1)
    ax2.plot(tensor[:, 0], tensor[:, 2], color='black', label="Envelop (V)")
    ax2.set_ylabel(ylabel='$V_{cell}$' + f'{vin_cell_dch[1]} - ' + '$V_{min}$ ')
    ax2.set_ylim([0, 0.04])
    ax2.legend(loc ='upper right')
    # label ypred
    ax3 = plt.subplot(grid[10:15, 0:10], sharex=ax1)
    ax3.plot(tensor[:, 0], tensor[:, -2], label='M01标签')
    ax3.plot(tensor[:, 0], tensor[:, -1], label='模型输出')
    ax3.set_ylabel(ylabel='Label')
    ax3.set_xlabel(xlabel='SOC')
    ax3.legend(loc ='upper right')
    ax3.set_ylim([0, 1])
    if var_monte is not None:
        ax3_right = ax3.twinx()
        ax3_right.plot(tensor[:, 0], var_monte,color='green', label='var Monte Carlo')
        ax3_right.legend(loc='upper left')
        ax1_right.set_ylabel(ylabel='确信度')
    #vmax vmin
    ax7 = plt.subplot(grid[15:20, 0:10], sharex=ax1)
    ax7.plot(tensor[:, 0], tensor[:, -3], color='red', label='$V_{cell}$' + f'-{vin_cell_dch[1]} ')
    ax7.plot(tensor[:, 0], tensor[:, -4], color='blue', label='$V_{min}$ ')
    ax7.set_ylabel(ylabel='V')
    ax7.set_xlabel(xlabel='SOC')
    ax7.legend(loc='upper right')
    ### 对照组
    # I
    ax4 = plt.subplot(grid[0:5, 12:22])
    ax4.plot(df['soc'].values, df['i'].values, color='indigo', label='I')
    # mean_i = df['i'].rolling(window=30, min_periods=1, center=True).mean()
    # ax4.plot(df['soc'].values, mean_i, color='darkgreen', label='均值I')
    ax4.set_ylabel(ylabel='Current')
    ax4.set_ylim([-80, 20])
    #ax4.set_xlim([50, 100])
    ax4.legend(loc ='upper right')
    ax4.set_title(
        f"{vin_cell_dch[0]} #{(vin_cell_dch[1])}节  ")
    #  Temp
    ax4_right = ax4.twinx()
    ax4_right.plot(df['soc'].values, df['temperature'].values, color='y', label='温度')
    ax4_right.legend(loc='upper left')
    ax4_right.set_ylim([0, 50])
    ax4_right.set_ylabel(ylabel='最低温度')
    # vdiff
    ax5 = plt.subplot(grid[5:10, 12:22], sharex=ax4)
    ax5.plot(df['soc'].values, df['vdiff'].values, color='black', label="Envelop (V)")
    ax5.set_ylabel(ylabel='$V_{cell}$' + f'{vin_cell_dch[1]} - ' + '$V_{min}$ ')
    ax5.set_ylim([0, 0.04])
    ax5.legend(loc ='upper right')
    # label
    ax6 =plt.subplot(grid[10:15, 12:22], sharex=ax4)
    ax6.plot(df['soc'].values, df['label'].values, label='M01 label')
    ax6.set_ylabel(ylabel='Label')
    ax6.set_xlabel(xlabel='SOC')
    ax6.set_ylim([0, 1])
    ax6.legend(loc ='upper right')
    # vmax vmin
    ax8 = plt.subplot(grid[15:20,12:22], sharex=ax4)
    ax8.plot(df['soc'].values, df['vcell'].values, color='red', label='$V_{cell}$' + f'-{vin_cell_dch[1]} ')
    ax8.plot(df['soc'].values, df['vmin'].values, color='blue', label='$V_{min}$ ')
    ax8.set_ylabel(ylabel='Label')
    ax8.set_xlabel(xlabel='SOC')
    ax8.legend(loc ='upper right')
    if show:
        plt.show()
    else:
        fig.savefig(save_path)
    plt.clf()
    plt.close()

def plot_joint_m03_tensor(tensor, vin_cell_dch, df, save_path):
    #  ["soc", "i", "vdiff", "time", "temperature", "ah", 'vmin', 'vcell']
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
    plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题
    fig = plt.figure(figsize=(15, 10))
    # model in n out
    grid = plt.GridSpec(21, 22, wspace=0.15, hspace=0.15)
    tensor = tensor.cpu().detach().numpy()
    is_not_pad = tensor[:, 0]!=0
    tensor = tensor[is_not_pad,:]
    # I
    ax1 = plt.subplot(grid[0:5, 0:10])
    ax1.plot(tensor[:, 0], tensor[:, 1] , color='mediumorchid', label='I')
    mean_i = pd.Series(tensor[:, 1]).rolling(window=30, min_periods=1, center=True).mean()
    ax1.plot(tensor[:, 0], mean_i, color='indigo', label='均值I')
    ax1.set_ylabel(ylabel='Current')
    ax1.set_ylim([-20, 80])
    #ax1.set_xlim([100, 50])
    ax1.legend(loc ='upper right')
    ax1.set_title(
        f"{vin_cell_dch[0]} #{(vin_cell_dch[1])}节 \n MODEL IN AND OUT")
    # Temp
    ax1_right = ax1.twinx()
    ax1_right.plot(tensor[:, 0], tensor[:, 4], color='y', label='温度')
    ax1_right.legend(loc='upper left')
    ax1_right.set_ylim([0, 50])
    ax1_right.set_ylabel(ylabel='最低温度')
    # vdiff
    ax2 =plt.subplot(grid[5:10, 0:10], sharex=ax1)
    ax2.plot(tensor[:, 0], tensor[:, 2], color='black', label="Envelop (V)")
    ax2.set_ylabel(ylabel='$V_{cell}$' + f'{vin_cell_dch[1]} - ' + '$V_{min}$ ')
    ax2.set_ylim([0, 0.04])
    ax2.legend(loc ='upper right')
    # label ypred
    ax3 = plt.subplot(grid[10:15, 0:10], sharex=ax1)
    ax3.plot(tensor[:, 0], tensor[:, -2], label='M01标签')
    ax3.plot(tensor[:, 0], tensor[:, -1], label='模型输出')
    ax3.plot(tensor[:, 0], tensor[:, -5], label='M03')
    ax3.set_ylabel(ylabel='Label')
    ax3.set_xlabel(xlabel='SOC')
    ax3.legend(loc ='upper right')
    ax3.set_ylim([0, 1])
    #vmax vmin
    ax7 = plt.subplot(grid[15:20, 0:10], sharex=ax1)
    ax7.plot(tensor[:, 0], tensor[:, -3], color='red', label='$V_{cell}$' + f'-{vin_cell_dch[1]} ')
    ax7.plot(tensor[:, 0], tensor[:, -4], color='blue', label='$V_{min}$ ')
    ax7.set_ylabel(ylabel='V')
    ax7.set_xlabel(xlabel='SOC')
    ax7.legend(loc='upper right')
    ### 对照组
    # I
    ax4 = plt.subplot(grid[0:5, 12:22])
    ax4.plot(df['soc'].values, df['i'].values, color='indigo', label='I')
    # mean_i = df['i'].rolling(window=30, min_periods=1, center=True).mean()
    # ax4.plot(df['soc'].values, mean_i, color='darkgreen', label='均值I')
    ax4.set_ylabel(ylabel='Current')
    ax4.set_ylim([-80, 20])
    #ax4.set_xlim([50, 100])
    ax4.legend(loc ='upper right')
    ax4.set_title(
        f"{vin_cell_dch[0]} #{(vin_cell_dch[1])}节  ")
    #  Temp
    ax4_right = ax4.twinx()
    ax4_right.plot(df['soc'].values, df['temperature'].values, color='y', label='温度')
    ax4_right.legend(loc='upper left')
    ax4_right.set_ylim([0, 50])
    ax4_right.set_ylabel(ylabel='最低温度')
    # vdiff
    ax5 = plt.subplot(grid[5:10, 12:22], sharex=ax4)
    ax5.plot(df['soc'].values, df['vdiff'].values, color='black', label="Envelop (V)")
    ax5.set_ylabel(ylabel='$V_{cell}$' + f'{vin_cell_dch[1]} - ' + '$V_{min}$ ')
    ax5.set_ylim([0, 0.04])
    ax5.legend(loc ='upper right')
    # label
    ax6 =plt.subplot(grid[10:15, 12:22], sharex=ax4)
    ax6.plot(df['soc'].values, df['label'].values, label='M01 label')
    ax6.set_ylabel(ylabel='Label')
    ax6.set_xlabel(xlabel='SOC')
    ax6.set_ylim([0, 1])
    ax6.legend(loc ='upper right')
    # vmax vmin
    ax8 = plt.subplot(grid[15:20,12:22], sharex=ax4)
    ax8.plot(df['soc'].values, df['vcell'].values, color='red', label='$V_{cell}$' + f'-{vin_cell_dch[1]} ')
    ax8.plot(df['soc'].values, df['vmin'].values, color='blue', label='$V_{min}$ ')
    ax8.set_ylabel(ylabel='Label')
    ax8.set_xlabel(xlabel='SOC')
    ax8.legend(loc ='upper right')
    fig.savefig(save_path)
    plt.clf()
    plt.close()
def plot_dch_env_csv(csv_root_path, csv_file_name, fig_save_path = None):
    # ['time', 'temperature', 'soc', 'i', 'vdiff','ah', 'cap_nominal','soc_x_axis','label' ]
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
    plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题
    fig = plt.figure(figsize=(15, 10))
    ax1 = plt.subplot(311)
    df = pd.read_csv(os.path.join(csv_root_path, csv_file_name))
    ax1.plot(df['soc'].values, df['i'].values, label='I')
    ax1.set_ylabel(ylabel='Current')
    ax1.set_ylim([-10, 50])
    ax1.set_xlim([100, 50])
    ax1.legend(loc ='upper right')
    vin_cell_dch = csv_file_name.split('_')
    ax1.set_title(
        f"{vin_cell_dch[0]} #{(vin_cell_dch[1])}节 {(vin_cell_dch[2])}dch")
    ax2 = plt.subplot(312, sharex=ax1)
    ax2.plot(df['soc'].values, df['vdiff'].values, label="V diff filtered")
    ax2.set_ylabel(ylabel='Envelop (V)')
    ax2.set_ylim([0, 0.04])
    ax2.legend(loc ='upper right')
    ax3 = plt.subplot(313, sharex=ax1)
    ax3.plot(df['soc'].values, df['label'].values, label='label')
    ax3.set_ylabel(ylabel='标签')
    ax3.set_xlabel(xlabel='SOC')
    ax3.set_ylim([0, 1])
    if fig_save_path:
        fig.savefig(os.path.join(fig_save_path, csv_file_name.split('csv')[0] + 'png'))
    else:
        plt.show()
    plt.clf()
    plt.close()

#
# def plot_dch_env_csv(csv_root_path, csv_file_name, fig_save_path):
#     # ['time', 'temperature', 'soc', 'i', 'vdiff','ah', 'cap_nominal','soc_x_axis','label' ]
#     plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
#     plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题
#     fig = plt.figure(figsize=(15, 10))
#     ax1 = plt.subplot(311)
#     df = pd.read_csv(os.path.join(csv_root_path, csv_file_name))
#     ax1.plot(df['soc'].values, df['i'].values, label='I')
#     ax1.set_ylabel(ylabel='Current')
#     ax1.set_ylim([-10, 50])
#     ax1.set_xlim([100, 50])
#     ax1.legend(loc ='upper right')
#     vin_cell_dch = csv_file_name.split('_')
#     ax1.set_title(
#         f"{vin_cell_dch[0]} #{(vin_cell_dch[1])}节 {(vin_cell_dch[2])}dch")
#     ax2 = plt.subplot(312, sharex=ax1)
#     ax2.plot(df['soc'].values, df['vdiff'].values, label="V diff filtered")
#     ax2.set_ylabel(ylabel='Envelop (V)')
#     ax2.set_ylim([0, 0.04])
#     ax2.legend(loc ='upper right')
#     ax3 = plt.subplot(313, sharex=ax1)
#     ax3.plot(df['soc'].values, df['label'].values, label='label')
#     ax3.set_ylabel(ylabel='标签')
#     ax3.set_xlabel(xlabel='SOC')
#     ax3.set_ylim([0, 1])
#     fig.savefig(os.path.join(fig_save_path, csv_file_name.split('csv')[0] + 'png'))
#     plt.clf()
#     plt.close()


def plot_joint_csv(csv_root_path, csv_file_name, fig_save_path):
    # ['time', 'temperature', 'soc', 'i', 'vdiff','ah', 'cap_nominal', 'label' ,'cdc']
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
    plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题
    fig = plt.figure(figsize=(15, 10))
    df_cdc = pd.read_csv(os.path.join(csv_root_path, csv_file_name))
    ### 放电段
    df = df_cdc[df_cdc['cstate'] == 0]
    ax1 = plt.subplot(321)
    ax1.plot(df['soc'].values, df['i'].values, label='I')
    ax1.set_ylabel(ylabel='Current')
    ax1.set_ylim([-10, 50])
    ax1.set_xlim([100, 50])
    ax1.legend(loc ='upper right')
    vin_cell_dch = csv_file_name.split('_')
    ax1.set_title(
        f"{vin_cell_dch[0]} #{(vin_cell_dch[1])}节 {(vin_cell_dch[2])}dch")
    ax2 = plt.subplot(323, sharex=ax1)
    ax2.plot(df['soc'].values, df['vdiff'].values, label="Envelop (V)")
    ax2.set_ylabel(ylabel='$V_{cell}$' + f'{vin_cell_dch[1]} - ' + '$V_{min}$ ')
    ax2.set_ylim([0, 0.04])
    ax2.legend(loc ='upper right')
    ax3 = plt.subplot(325, sharex=ax1)
    ax3.plot(df['soc'].values, df['label'].values, label='M01 label')
    ax3.set_ylabel(ylabel='Label')
    ax3.set_xlabel(xlabel='SOC')
    ax3.set_ylim([0, 1])
    ### 充电段
    df = df_cdc[df_cdc['cstate'] == 1]
    ax4 = plt.subplot(322)
    ax4.plot(df['soc'].values, df['i'].values, label='I')
    ax4.set_ylabel(ylabel='Current')
    ax4.set_ylim([-50, 10])
    ax4.set_xlim([50, 100])
    ax4.legend(loc ='upper right')
    vin_cell_dch = csv_file_name.split('_')
    ax4.set_title(
        f"{vin_cell_dch[0]} #{(vin_cell_dch[1])}节 M01充电段")
    ax5 = plt.subplot(324, sharex=ax4)
    ax5.plot(df['soc'].values, df['vdiff'].values, label="Envelop (V)")
    ax5.set_ylabel(ylabel='$V_{cell}$' + f'{vin_cell_dch[1]} - ' + '$V_{min}$ ')
    ax5.set_ylim([0, 0.04])
    ax5.legend(loc ='upper right')
    ax6 = plt.subplot(326, sharex=ax4)
    ax6.plot(df['soc'].values, df['label'].values, label='M01 label')
    ax6.set_ylabel(ylabel='Label')
    ax6.set_xlabel(xlabel='SOC')
    ax6.set_ylim([0, 1])

    # plt.show()
    #
    if True:
        fig.savefig(os.path.join(fig_save_path, csv_file_name.split('csv')[0] + 'png'))
        plt.clf()
        plt.close()
    else:
        plt.show()





class DistPlot():
    @staticmethod
    def plot_dist(df, dist_name_a='m03_dtp', dist_name_b='m01_dtp',title_description = ''):
        plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
        plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题
        grid = plt.GridSpec(7, 12, wspace=0.15, hspace=0.15)
        ax0 = plt.subplot(grid[0:3, 0:12])
        ax0.set_title(f"{dist_name_a}/{dist_name_b} distribution "+f'\n{title_description}')
        sns.histplot(df[dist_name_a], kde=True, binwidth=0.5, ax=ax0, color='C01', label=dist_name_a )
        sns.histplot(df[dist_name_b], kde=True, binwidth=0.5, ax=ax0, color='C02', label=dist_name_b )
        ax1 = plt.subplot(grid[4:7, 0:12])
        df['dtp_diff'] = df[dist_name_a] - df[dist_name_b]
        ax1.set_title(f"{dist_name_a} - {dist_name_b} μ: {df['dtp_diff'].mean():.1f} σ: {df['dtp_diff'].var():.1f}")
        sns.histplot(df['dtp_diff'], kde=True, binwidth=0.5, ax=ax1, label=f'{dist_name_a} - {dist_name_b} dtp'+ f"[{df['dtp_diff'].min():.2f}~{df['dtp_diff'].max():.2f}]")
        ax0.legend(loc ='upper right')
        #ax1.plot([df['dtp_diff'].mean(), df['dtp_diff'].mean()], [0, 3000], '-.r', )
        ax1.legend(loc ='upper right')
        plt.show()
    @staticmethod
    def plot_error_current(df ):
        error = (df['模型输出'] - df['M01']).abs().values
        current = df[7]
        plt.scatter(error,current)
        plt.show()

    @staticmethod
    def plot_m01_m03_model_dist(df_model,df_m03 , title_description=''):
        plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
        plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题
        df = pd.merge(df_model, df_m03, left_on=[0, 1, 2], right_on=['vin', 'cell', 'dch'], how='outer')

        dist_name_a = 'M01'
        dist_name_b = '模型输出'
        dist_name_c = 'M03'
        grid = plt.GridSpec(7, 12, wspace=0.15, hspace=0.15)
        ax0 = plt.subplot(grid[0:3, 0:12])
        sns.histplot(df[dist_name_a], kde=True, binwidth=0.5, ax=ax0, color='C01', label=dist_name_a)
        sns.histplot(df[dist_name_b], kde=True, binwidth=0.5, ax=ax0, color='C02', label=dist_name_b)
        sns.histplot(df[dist_name_c], kde=True, binwidth=0.5, ax=ax0, color='C03', label=dist_name_c)
        ax1 = plt.subplot(grid[4:7, 0:12])
        df['dtp_diff'] = df[dist_name_b] - df[dist_name_a]
        df['dtp_diff_m03_m01'] =(df[dist_name_c] - df[dist_name_a]).dropna()
        df['model_m03_diff'] = abs(df[dist_name_b] - df[dist_name_c]) < 3
        df['dtp_diff_model_m03'] = df[df['model_m03_diff']==True]['dtp_diff']
        ax1.set_title(f"{dist_name_b} - {dist_name_a} μ: {df['dtp_diff'].mean():.1f} σ: {df['dtp_diff'].var():.1f}\n"
                      f"{dist_name_c} - {dist_name_a} μ: {df['dtp_diff_m03_m01'].mean():.1f} σ: {df['dtp_diff_m03_m01'].var():.1f} \n"
                      f"经过M03验证的{dist_name_b} - {dist_name_a} μ: {df['dtp_diff_model_m03'].mean():.1f} σ: {df['dtp_diff_model_m03'].var():.1f} ")
        # sns.histplot(df['dtp_diff'], kde=True, binwidth=0.5, ax=ax1, label=f'{dist_name_b} - {dist_name_a} dtp')
        # sns.histplot(df['dtp_diff_m03_m01'], kde=True, binwidth=0.5, ax=ax1, label=f'{dist_name_c} - {dist_name_a} dtp')
        sns.histplot(df['dtp_diff_model_m03'], kde=True, binwidth=0.5, ax=ax1, label=f'经过M03验证的{dist_name_b} - {dist_name_a} dtp')
        ax0.legend(loc='upper right')
        # ax1.plot([df['dtp_diff'].mean(), df['dtp_diff'].mean()], [0, 3000], '-.r', )
        ax1.legend(loc='upper right')
        plt.show()

def multi_res_hist_plotter_AIPD(result_path):
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
    plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题
    df = pd.read_csv(result_path)
    df['dtp'] = abs(df['最高节电芯拐点-SOC'] - df['最低节电芯拐点-SOC'])
    df['模型输出最低节电芯拐点-SOC'] = df.apply(lambda row: row['模型输出拐点1-SOC'] if row['拐点1类别'] == 1 else row['模型输出拐点2-SOC'], axis=1)
    df['模型输出最高节电芯拐点-SOC'] = df.apply(lambda row: row['模型输出拐点2-SOC'] if row['拐点1类别'] == 1 else row['模型输出拐点1-SOC'], axis=1)

    df['低拐点偏差'] = abs(df['模型输出最低节电芯拐点-SOC'] - df['最低节电芯拐点-SOC'])
    df['高拐点偏差'] = abs(df['模型输出最高节电芯拐点-SOC'] - df['最高节电芯拐点-SOC'])

    print(df['高拐点偏差'].sort_values(ascending=False).head(10))
    print(df['低拐点偏差'].sort_values(ascending=False).head(10))
    print(df['高拐点偏差'].sort_values(ascending=False).tail(10))
    print(df['低拐点偏差'].sort_values(ascending=False).tail(10))
    # 尾部区域均值
    print(df[df['高拐点偏差'] >= df['高拐点偏差'].quantile(0.95)]['高拐点偏差'].mean())
    print(df[df['低拐点偏差'] >= df['低拐点偏差'].quantile(0.95)]['低拐点偏差'].mean())
    # 尾部区域比例
    tail_vcell_num = len(df[(df['高拐点偏差'] >= 10)])
    tail_vmin_num = len(df[(df['低拐点偏差'] >= 10)])
    print((tail_vcell_num + tail_vmin_num)/(2*len(df)))
    # df['低拐点偏差'] = df['低拐点偏差'] / df['标称容量'] * 100
    # df['高拐点偏差'] = df['高拐点偏差'] / df['标称容量'] * 100

    # 绘制直方图
    plt.figure(figsize=(10, 6))
    bins = np.linspace(0, 20, 20)

    sns.histplot(data=df, x='低拐点偏差', bins=bins, alpha=0.4, label=f"最低节电芯拐点预测 - 最低节电芯拐点标签 σ: {df['低拐点偏差'].std(): .2f} μ: {df['低拐点偏差'].mean(): .2f}", kde=False)
    sns.histplot(data=df, x='高拐点偏差', bins=bins, alpha=0.4, label=f"最高节电芯拐点预测 - 最高节电芯拐点标签 σ: {df['高拐点偏差'].std(): .2f} μ: {df['高拐点偏差'].mean(): .2f}", kde=False)
    # sns.histplot(data=df, x='低拐点偏差', bins=20, alpha=0.4, label=f"最低节电芯拐点预测 - 最低节电芯拐点标签 σ: {df['低拐点偏差'].std(): .2f} μ: {df['低拐点偏差'].mean(): .2f}", kde=False)
    # sns.histplot(data=df, x='高拐点偏差', bins=20, alpha=0.4, label=f"最高节电芯拐点预测 - 最高节电芯拐点标签 σ: {df['高拐点偏差'].std(): .2f} μ: {df['高拐点偏差'].mean(): .2f}", kde=False)

    # 添加图例和轴标签
    plt.tick_params(labelsize=16)
    plt.legend()
    plt.xlabel('拐点位置偏差 / SOC(%)', fontsize=16)
    plt.ylabel('Count', fontsize=16)
    plt.title('模型预测拐点位置与真实拐点位置偏差')

    # 显示图形
    plt.show()


# if __name__ == '__main__':
#
#     matplotlib.use('tkagg')
#     '''
#     画csv
#     '''
#     result_path = r'D:\AnYuting\deep_learning_project\04_detect_tp\point_detect\big_files\evaluation\20250327\train_res_20250327_2025-03-27_08-41_980.csv'
#     multi_res_hist_plotter_AIPD(result_path)

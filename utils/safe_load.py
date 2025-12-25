import os
import torch

def safe_load(net, pt_path):
    if os.path.exists(pt_path):
        # 询问用户是否要加载模型
        user_input = input("找到模型文件，是否要加载并继续训练？(y/n): ")
        if user_input.lower() == 'y':
            print("要加载模型")
            pt_file = torch.load(pt_path, map_location=torch.device("cpu")).state_dict()
            net_dict = net.state_dict()
            pretrained_dict = {k: v for k, v in pt_file.items() if k in net_dict}
            for key in pretrained_dict.keys():
                net_tensor = net_dict[key]
                pretrain_tensor = pretrained_dict[key]
                if net_tensor.shape == pretrain_tensor.shape:
                    net_dict.update({key: pretrained_dict[key]})
            net.load_state_dict(net_dict, strict=False)
        else:
            print("不加载模型")
    else:
        print(f"不存在模型{pt_path}")
    return net

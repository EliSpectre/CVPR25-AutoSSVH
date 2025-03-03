import torch


def AutoSSVH_inference(cfg, data, model,device):
    data = {key: value.to(device) for key, value in data.items()}

    my_H = model.inference(data["visual_word"].to(device))
    my_H = torch.mean(my_H, 1)
    
    BinaryCode = torch.sign(my_H)
    return BinaryCode
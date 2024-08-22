import torch

def noise_estimation_loss(model,
                          target: torch.Tensor,
                          source: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False):
    
    
    
    # b是beta序列，1-b也就是\alpha_{t},根据随机生成的时刻t，来选择对应的\bar{\alpha}_{t}
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)

    # 传入的x0应该是[B,2,H,W]，其中在维度1上，dim=0表示low resolution， dim=1 为high resolution
    # 对x0进行切分

    # 扰动过程是施加在high resolution部分
    x_t = target * a.sqrt() + e * (1.0 - a).sqrt()
    # 将x_lr和x_t拼接
    x = torch.cat([x_t, source], dim=1)
    # 模型根据输入的图片x_{t}和时刻t来估计噪声
    output = model(x, t.float())
    # print(f'the shape of e :{e.shape} and output: {output.shape}' )
    
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3, 4))
    else:
        return (e - output).square().sum(dim=(1, 2, 3, 4)).mean(dim=0)

loss_registry = {
    'simple': noise_estimation_loss,
}

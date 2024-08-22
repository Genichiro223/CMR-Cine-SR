import torch

def noise_estimation_loss(model,
                          target: torch.Tensor,
                          source: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False):
    
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)

    # The noise should add on the target part and the 
    # source part is set as the condition.
    x_t = target * a.sqrt() + e * (1.0 - a).sqrt()
    # Concatenate the data in the channel dimension.
    x = torch.cat([x_t, source], dim=1)
    output = model(x, t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)

loss_registry = {
    'simple': noise_estimation_loss,
}
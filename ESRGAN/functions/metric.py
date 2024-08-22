import torch
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from typing import Tuple

def metric(pred:torch.tensor, target:torch.tensor) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    r"""
        input:
            pred or imag: B C H W  real number
        return:
            psnr, ssim, nmse
     """


    pred = pred.unsqueeze(0)  # b C H W
    target = target.unsqueeze(0)  # b C H w

    return peak_signal_noise_ratio(pred, target), structural_similarity_index_measure(pred, target), torch.norm(pred - target, 2) / torch.norm(target, 2)


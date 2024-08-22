import torch

def anneal_dsm_score_estimation(scorenet, hr, lr, sigmas, labels=None, anneal_power=2., hook=None):
    if labels is None:
        labels = torch.randint(0, len(sigmas), (hr.shape[0],), device=hr.device)
    used_sigmas = sigmas[labels].view(hr.shape[0], *([1] * len(hr.shape[1:])))
    noise = torch.randn_like(hr) * used_sigmas
    perturbed_hr = hr+ noise
    target = - 1 / (used_sigmas ** 2) * noise
    concat_input = torch.cat([perturbed_hr, lr], dim=1)
    scores = scorenet(concat_input, labels)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power

    if hook is not None:
        hook(loss, labels)

    return loss.mean(dim=0)

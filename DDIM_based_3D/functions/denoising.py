import torch

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1, 1)
    return a

def generalized_steps(initial_noise, seq, model, source_image, betas, **kwargs):

    with torch.no_grad():
        n = source_image.size(0)
        seq_next = [-1] + list(seq[:-1])

        device = initial_noise.device
        
        x0_preds = []
        xs = [initial_noise]  # xs is a list to store the intermediate sampled result

        for i, j in zip(reversed(seq), reversed(seq_next)):

            t = (torch.ones(n) * i).to(device)
            next_t = (torch.ones(n) * j).to(device)

            at = compute_alpha(betas, t.long())
            at_next = compute_alpha(betas, next_t.long())

            xt = xs[-1].to(device)

            y = torch.cat([xt, source_image], dim=1).to(device)
            print('y',y.shape, y.device)
            et = model(y, t)

            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            # x0_t = torch.clamp(x0_t, -1, 1)
            x0_preds.append(x0_t.to('cpu'))
            
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(initial_noise) + c2 * et
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds


# def ddpm_steps(initial_noise, seq, model, source_image, b, **kwargs):
#     with torch.no_grad():
#         n = source_image.size(0)
#         seq_next = [-1] + list(seq[:-1])
#         xs = [initial_noise]
#         x0_preds = []
#         betas = b
#         device = initial_noise.device
#         for i, j in zip(reversed(seq), reversed(seq_next)):
            
#             t = (torch.ones(n) * i).to(source_image.device)
#             next_t = (torch.ones(n) * j).to(device)
            
#             at = compute_alpha(betas, t.long())
#             atm1 = compute_alpha(betas, next_t.long())
            
#             beta_t = 1 - at / atm1
#             xt = xs[-1].to('cuda')
#             y = torch.concatenate([source_image, xt], dim=1).to(initial_noise.device)
#             output = model(y, t.float())
#             e = output

#             x0_from_e = (1.0 / at).sqrt() * xt - (1.0 / at - 1).sqrt() * e
#             x0_from_e = torch.clamp(x0_from_e, -1, 1)
#             x0_preds.append(x0_from_e.to('cpu'))
#             mean_eps = (
#                 (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * xt
#             ) / (1.0 - at)

#             mean = mean_eps
#             noise = torch.randn_like(x)
#             mask = 1 - (t == 0).float()
#             mask = mask.view(-1, 1, 1, 1)
#             logvar = beta_t.log()
#             sample = mean + mask * torch.exp(0.5 * logvar) * noise
#             xs.append(sample.to('cpu'))
#     return xs, x0_preds

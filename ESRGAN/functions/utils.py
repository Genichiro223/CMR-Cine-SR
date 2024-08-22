import torch
import math
import matplotlib.pyplot as plt
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt
import math
import numpy as np
import torch
import torchvision.utils
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity

def ShowPARS_total(imgs, ncols=4, dpi=100, cmap='jet', is_colorbar=True, is_ticks=False,
            col_width=4, row_width=3, margin_ratio=0.1, n_images_max=50, filename2save=None,  **imshow_kwargs):
    '''
    list for imgs
    assume imgs is Sequence[ndarray[Nx, Ny]]
    '''
    num_imgs = len(imgs)
    titles = ['A_real map','B_real map','T1*_real map','T1_real map',\
              'A_gen map', 'B_gen map', 'T1*_gen map', 'T1_gen map',\
              'A_diff map', 'B_diff map', 'T1*_diff map', 'T1_diff map' ]
    vmin = [0,0,0,0,0,0,0,0,0,0,0,0]
    vmax = [1,1,2000,2000,1,1,2000,2000,1,1,2000,2000]
    if num_imgs > n_images_max:
        print(
            f"[WARNING] Too many images ({num_imgs}), clip to argument n_images_max({n_images_max}) for performance reasons.")
        imgs = imgs[:n_images_max]
        num_imgs = n_images_max

    if isinstance(cmap, list):
        assert len(cmap) == len(imgs)
    else:
        cmap = [cmap, ] * num_imgs

    nrows = math.ceil(num_imgs / ncols)

    # compute the figure size, compute necessary size first, then add margin
    figsize = (ncols * col_width, nrows * row_width)
    figsize = (figsize[0] * (1 + margin_ratio), figsize[1] * (1 + margin_ratio))
    fig = plt.figure(dpi=dpi, figsize=figsize)
    for i in range(num_imgs):
        ax = plt.subplot(nrows, ncols, i + 1)
        im = ax.imshow(imgs[i], cmap=cmap[i],vmin=vmin[i],vmax=vmax[i], **imshow_kwargs)
        if titles:
            plt.title(titles[i])
        if is_colorbar:  # 生成colorbar的方式
            cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.01, ax.get_position().height])
            plt.colorbar(im, cax=cax)
        if not is_ticks:
            ax.set_xticks([])
            ax.set_yticks([])
    if filename2save is not None:
        fig.savefig(filename2save)
    else:
        plt.show()
    plt.close('all')
    
def gradient_penalty(critic, real, fake, device): # 会产生一个梯度的梯度
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device) # batch之间的随机数不同，但是每张图之间的随机数相同
    interpolated_images = real * alpha + fake * (1 - alpha) # 两张数据进行合并
    interpolated_images.requires_grad_(True)

    # calcelate critic scores
    mixed_scores = critic(interpolated_images)
    # take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(   # 用于计算反向传播中的梯度 对真假图像合并的结果计算梯度 用于梯度惩罚
        inputs = interpolated_images, # 相对于这些张量计算梯度
        outputs=mixed_scores,         # 表示要计算梯度的目标张量
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,  # 是否在计算图中创建梯度计算图
        retain_graph=True,  # 是否保留计算图用于多次计算
    )[0]  # 实现自动更新模型参数
    gradient = gradient.view(gradient.shape[0], -1) # 将数据变换为二维向量 B,C,H,W->B,C*H*W
    gradient_norm = gradient.norm(2, dim=1) # 计算梯度2范数
    return  torch.mean((gradient_norm - 1) ** 2)

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"): # .pth.tar相比.pth文件可能会包含更多的信息
    print("=> Saving checkpoint")
    if isinstance(model, list):
        checkpoint={
        "flstm_state_dict" : model[0].state_dict(),
        "blstm_state_dict" : model[1].state_dict(),
        "gen_state_dict" : model[2].state_dict(),
        "optimizer": optimizer.state_dict(),
        }
    else:
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location='cuda')
    # 为了保证可以读取到存储的参数
    if isinstance(model,list):
        model[0].load_state_dict(checkpoint["flstm_state_dict"]) 
        model[1].load_state_dict(checkpoint["blstm_state_dict"]) 
        model[2].load_state_dict(checkpoint["gen_state_dict"]) 
    else:
        model.load_state_dict(checkpoint["state_dict"]) # load 指定部分的参数
    optimizer.load_state_dict(checkpoint["optimizer"])
    # if dont do this then it will just have learning rate of old checkpoint and it will lead to many hours of debugging
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def image2kspace(x):
    if isinstance(x,np.ndarray):
        x=np.fft.ifftshift(x,axes=(-2,-1))
        x=np.fft.fft2(x)
        x=np.fft.fftshift(x,axes=(-2,-1))
        return x
    elif isinstance(x,torch.Tensor):
        x=torch.fft.ifftshift(x,dim=(-2,-1))
        x=torch.fft.fft2(x)
        x=torch.fft.fftshift(x,dim=(-2,-1))
        return x
    else:
        raise RuntimeError("Unsupported type!")

def kspace2image(x):
    if isinstance(x,np.ndarray):
        x=np.fft.ifftshift(x,axes=(-2,-1))
        x=np.fft.ifft2(x)
        x=np.fft.fftshift(x,axes=(-2,-1))
        return x

    elif isinstance(x,torch.Tensor):
        x=torch.fft.ifftshift(x,dim=(-2,-1))
        x=torch.fft.ifft2(x)
        x=torch.fft.fftshift(x,dim=(-2,-1))
        return x
    
    else:
        raise RuntimeError('Unsupported type!')
# for vgg want to get 3 channels
def img_net_normalize(x):
    return torch.cat(((x-0.485)/0.229, (x-0.456)/0.224, (x-0.406)/0.225), 1)

class AverageMeter(object):
    """Computes and stores the average and current value"""  # 计算和存储平均值和当前值
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def norm_fix(data, min_target, max_target):
    new_min = min_target
    new_max = max_target

    # 计算矩阵的最小值和最大值
    min_value = data.min()
    max_value = data.max()

    # 缩放矩阵的数值
    return  (data - min_value) * (new_max - new_min) / (max_value - min_value) + new_min

def norm_normalize(data,coef = 1.):
    norm_ = torch.norm(data, p = 1)/torch.numel(data)
    norm_data = data/norm_
    norm_data = norm_data * coef

    return norm_data



def psnr_slice(gt, pred, maxval=None):
    assert type(gt) == type(pred)
    if type(pred) is torch.Tensor:
        gt, pred = gt.detach().cpu().numpy(), pred.detach().cpu().numpy()
    batch_size = gt.shape[0]
    T = gt.shape[1]

    PSNR = 0.0
    # PSNR1=0.0
    
    for i in range(batch_size):
        max_val = gt[i].max() if maxval is None else maxval  
        # print('psnr-gt-shape:',gt.shape)
        # print('psnr-pred-shape:',pred.shape)
        # print('psnr-T:',T)
        # print('psnr-batch_size:',batch_size)
        # for j in range(T):
        PSNR += peak_signal_noise_ratio(gt[i].squeeze(), pred[i].squeeze(), data_range=max_val)
        # PSNR += PSNR1/T 
        # PSNR1=0.0

    return PSNR / batch_size


def ssim_slice(gt, pred, maxval=None):
    assert type(gt) == type(pred)
    if type(pred) is torch.Tensor:
        gt, pred = gt.detach().cpu().numpy(), pred.detach().cpu().numpy()
    batch_size = gt.shape[0]
    T = gt.shape[1]

    SSIM = 0.0
    # SSIM1 = 0.0
    for i in range(batch_size):
        # print('ssim-gt-shape:',gt.shape)
        # print('ssim-pred-shape:',pred.shape)
        # print('ssim-T:',T)
        # print('ssim-batch_size:',batch_size)
        max_val = gt[i].max() if maxval is None else maxval
        # for j in range(T):
        SSIM += structural_similarity(gt[i].squeeze(), pred[i].squeeze(), data_range=max_val,multichannel=True)
        # SSIM += SSIM1/T
        # SSIM1=0.0

    return SSIM / batch_size


def center_crop(data, shape):
    assert 0 < shape[0] <= data.shape[-2]
    assert 0 < shape[1] <= data.shape[-1]
    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to]


def normalize_zero_to_one(data, eps=0.):
    data_min = float(data.min())
    data_max = float(data.max())
    return (data - data_min) / (data_max - data_min + eps)


#####################################################################

def plot_loss(loss):
    plt.figure()
    plt.plot(loss)
    plt.show()
    plt.close('all')


def imgshow(im, cmap=None, rgb_axis=None, dpi=100, figsize=(6.4, 4.8)):
    if isinstance(im, torch.Tensor):
        im = im.to('cpu').detach().cpu().numpy()
    if rgb_axis is not None:
        im = np.moveaxis(im, rgb_axis, -1)
        im = rgb2gray(im)

    plt.figure(dpi=dpi, figsize=figsize)
    norm_obj = Normalize(vmin=im.min(), vmax=im.max())
    plt.imshow(im, norm=norm_obj, cmap=cmap)
    plt.colorbar()
    plt.show()
    plt.close('all')


def imsshow(imgs, titles=None, num_col=5, dpi=100, cmap=None, is_colorbar=False, is_ticks=False):
    '''
    assume imgs's shape is (Nslice, Nx, Ny)
    '''
    num_imgs = len(imgs)
    num_row = math.ceil(num_imgs / num_col)
    fig_width = num_col * 3
    if is_colorbar:
        fig_width += num_col * 1.5
    fig_height = num_row * 3
    fig = plt.figure(dpi=dpi, figsize=(fig_width, fig_height))
    for i in range(num_imgs):
        ax = plt.subplot(num_row, num_col, i + 1)
        im = ax.imshow(imgs[i], cmap=cmap)
        if titles:
            plt.title(titles[i])
        if is_colorbar:
            cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.01, ax.get_position().height])
            plt.colorbar(im, cax=cax)
        if not is_ticks:
            ax.set_xticks([])
            ax.set_yticks([])
    plt.show()
    plt.close('all')


def make_grid_and_show(ims, nrow=5, cmap=None):
    if isinstance(ims, np.ndarray):
        ims = torch.from_numpy(ims)

    B, C, H, W = ims.shape
    grid_im = torchvision.utils.make_grid(ims, nrow=nrow)
    fig_h, fig_w = nrow * 2 + 1, (B / nrow) + 1
    imgshow(grid_im, cmap=cmap, rgb_axis=0, dpi=200, figsize=(fig_h, fig_w))


def int2preetyStr(num: int):
    s = str(num)
    remain_len = len(s)
    while remain_len - 3 > 0:
        s = s[:remain_len - 3] + ',' + s[remain_len - 3:]
        remain_len -= 3
    return s


def compute_num_params(module, is_trace=False):
    print(int2preetyStr(sum([p.numel() for p in module.parameters()])))
    if is_trace:
        for item in [f"[{int2preetyStr(info[1].numel())}] {info[0]}:{tuple(info[1].shape)}"
                     for info in module.named_parameters()]:
            print(item)


def tonp(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    else:
        return x


def pseudo2real(x):
    """
    :param x: [..., C=2, H, W]
    :return: [..., H, W]
    """
    return (x[..., 0, :, :] ** 2 + x[..., 1, :, :] ** 2) ** 0.5


def complex2pseudo(x):
    """
    :param x: [..., H, W] Complex
    :return: [...., C=2, H, W]
    """
    if isinstance(x, np.ndarray):
        return np.stack([x.real, x.imag], axis=-3)
    elif isinstance(x, torch.Tensor):
        return torch.stack([x.real, x.imag], dim=-3)
    else:
        raise RuntimeError("Unsupported type.")


def pseudo2complex(x):
    """
    :param x:  [..., C=2, H, W]
    :return: [..., H, W] Complex
    """
    return x[..., 0, :, :] + x[..., 1, :, :] * 1j


# ================================
# Preprocessing
# ================================
def minmax_normalize(x, eps=1e-8):
    min = x.min()
    max = x.max()
    return (x - min) / (max - min + eps)




# ======================================
# Metrics
# ======================================
def compute_mse(x, y):
    """
    REQUIREMENT: `x` and `y` can be any shape, but their shape have to be same
    """
    assert x.dtype == y.dtype and x.shape == y.shape, \
        'x and y is not compatible to compute MSE metric'

    if isinstance(x, np.ndarray):
        mse = np.mean(np.abs(x - y) ** 2)

    elif isinstance(x, torch.Tensor):
        mse = torch.mean(torch.abs(x - y) ** 2)

    else:
        raise RuntimeError(
            'Unsupported object type'
        )
    return mse


def compute_psnr(reconstructed_im, target_im, peak='normalized', is_minmax=False):
    '''
    Image must be of either Integer [0, 255] or Float value [0,1]
    :param peak: 'max' or 'normalize', max_intensity will be the maximum value of target_im if peek == 'max.
          when peek is 'normalized', max_intensity will be the maximum value depend on data representation (in this
          case, we assume your input should be normalized to [0,1])
    REQUIREMENT: `x` and `y` can be any shape, but their shape have to be same
    '''
    assert target_im.dtype == reconstructed_im.dtype and target_im.shape == reconstructed_im.shape, \
        'target_im and reconstructed_im is not compatible to compute PSNR metric'
    assert peak in {'max', 'normalized'}, \
        'peak mode is not supported'

    eps = 1e-8  # to avoid math error in log(x) when x=0

    if is_minmax:
        reconstructed_im = minmax_normalize(reconstructed_im, eps)
        target_im = minmax_normalize(target_im, eps)

    if isinstance(target_im, np.ndarray):
        max_intensity = 255 if target_im.dtype == np.uint8 else 1.0
        max_intensity = np.max(target_im).item() if peak == 'max' else max_intensity
        psnr = 20 * math.log10(max_intensity) - 10 * np.log10(compute_mse(reconstructed_im, target_im) + eps)

    elif isinstance(target_im, torch.Tensor):
        max_intensity = 255 if target_im.dtype == torch.uint8 else 1.0
        max_intensity = torch.max(target_im).item() if peak == 'max' else max_intensity
        psnr = 20 * math.log10(max_intensity) - 10 * torch.log10(compute_mse(reconstructed_im, target_im) + eps)

    else:
        raise RuntimeError(
            'Unsupported object type'
        )
    return psnr

def axes_to_last_order(ndim, axes_idx):
    """
    Generate axes order of move the axes specifed by `axes_idx` to last in a n-dimension axes,
    return indices can used to permute(array, indices) and inverse_indices can used
    to restore the data by permute(array, inverse_indices)
    eg.
        (Nb, Nx, Ny, Ntime) --move(2,3)--> (Nb, Ntime, Nx, Ny)
        (..., Nx, Ny, Ntime, Nslice) --move(-4,-3)--> (..., Ntime, Nslice, Nx, Ny)
        shape (1,2,3,4,5) --move(3,2)--> shape (1,2,5,4,3)
    """
    # convert reversed index to ordinal index
    axes_need_move = [(axis if axis >= 0 else ndim + axis) for axis in axes_idx]

    # generate forward-indices
    forward_indices = tuple([axis for axis in range(ndim) if axis not in axes_need_move] + axes_need_move)

    # generate inverse-indices
    inverse_indices = np.argsort(forward_indices).tolist()

    return forward_indices, inverse_indices


def compute_sim(reconstructed_im, target_im, image_axes=(-2, -1)):
    """
    Compute structural similarity index between two batches using skimage library,
    which only accept 2D-image input. We have to specify where is image's axes.

    WARNING: this method using skimage's implementation, DOES NOT SUPPORT GRADIENT
    """

    assert target_im.dtype == reconstructed_im.dtype and target_im.shape == reconstructed_im.shape, \
        'target_im and reconstructed_im is not compatible to compute SSIM metric'

    if isinstance(target_im, np.ndarray):
        pass
    elif isinstance(target_im, torch.Tensor):
        target_im = target_im.detach().to('cpu').numpy()
        reconstructed_im = reconstructed_im.detach().to('cpu').numpy()
    else:
        raise RuntimeError(
            'Unsupported object type'
        )
    # b c h w 
    axes_indices, _ = axes_to_last_order(target_im.ndim, image_axes)

    rec_im_seq = np.transpose(reconstructed_im, axes_indices).reshape(-1, target_im.shape[-2], target_im.shape[-1])
    target_im_seq = np.transpose(target_im, axes_indices).reshape(-1, target_im.shape[-2], target_im.shape[-1])
    sequence_length = len(target_im_seq)

    ssim_acc = sum(
        [structural_similarity(target_im_seq[i, :, :], rec_im_seq[i, :, :],
                               gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
                                data_range= target_im_seq[i, :, :].max() - target_im_seq[i, :, :].min())
         for i in range(sequence_length)]
    )

    return ssim_acc / sequence_length


    #former code begin at #b c h w
    # eps = 1e-8  # to avoid math error in log(x) when x=0

    # if is_minmax:
    #     reconstructed_im = minmax_normalize(reconstructed_im, eps)
    #     target_im = minmax_normalize(target_im, eps)
  
    # target_im1=target_im#.transpose(0,2,3,1)
    # reconstructed_im1=reconstructed_im#.transpose(0,2,3,1)
  
    # for i in range(target_im1.shape[0]):
    #     ssim_value = structural_similarity(target_im1[i], reconstructed_im1[i], \
    #         gaussian_weights=True, sigma=1.5, use_sample_covariance=True,multichannel=True,channel_axis=-1)

    # return ssim_value
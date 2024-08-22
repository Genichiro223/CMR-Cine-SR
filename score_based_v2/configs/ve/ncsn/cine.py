from configs.default_cine import get_default_configs

def get_config():
    config = get_default_configs()
    
    # training
    training = config.training
    training.sde = 'vesde'
    training.continuous = True
    
    # sampling
    sampling = config.sampling
    sampling.method = 'pc'
    sampling.predictor = 'reverse_diffusion'
    sampling.corrector = 'langevin'
    sampling.n_steps_each = 100
    sampling.snr = 0.316
    
    # model
    model = config.model
    model.name = 'ncsnv2_128'
    model.scale_by_sigma = False
    model.sigma_max = 1
    model.num_scales = 10
    model.ema_rate = 0.999
    model.normalization = 'InstanceNorm++'
    model.nonlinearity = 'elu'
    model.nf = 128
    model.interpolation = 'bilinear'
    model.type = "simple"
    model.in_channels = 2
    model.out_ch = 1
    model.ch = 128
    model.ch_mult = [1, 1, 2, 2, 4, 4]
    model.num_res_blocks = 2
    model.attn_resolutions = [16, ]
    model.dropout = 0.1
    model.var_type = 'fixedsmall'  # 方差选择
    model.ema_rate =  0.999
    model.ema = True
    model.resamp_with_conv = True
    
    # optim
    optim = config.optim
    optim.weight_decay = 0
    optim.optimizer = 'Adam'
    optim.lr = 1e-3
    optim.beta1 = 0.9
    optim.amsgrad = False
    optim.eps = 1e-8
    optim.warmup = 0
    optim.grad_clip = -1.

    return config
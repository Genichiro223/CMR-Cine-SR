import math
import torch
import torch.nn as nn
# from attention import SpatialTransformer
from einops import rearrange, repeat, pack , unpack
from  einops.layers.torch import Rearrange
import torch
import torch.nn as nn
from torch import nn, einsum
from einops import rearrange, repeat, pack , unpack
import sys, os
from functools import wraps, partial


def _many(fn):
    @wraps(fn)
    def inner(tensors, pattern, **kwargs):
        return (fn(tensor, pattern, **kwargs) for tensor in tensors)
    return inner

rearrange_many = _many(rearrange)

def exists(x):
    return x is not None

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

def nonlinearity(x):
    return x*torch.sigmoid(x)

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = dict(tuple(zip(self.from_einops.split(' '), shape)))
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        x = self.fn(x, **kwargs)   
        x = rearrange(x, f'{self.to_einops} -> {self.from_einops}', **reconstitute_kwargs)
        return x 

class SpatialTemporalConv(nn.Module): 
    def __init__(self,
                 in_channels,
                 out_channels=None,
                 # conv_shortcut=False
                 ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=1)
        # self.use_conv_shortcut = conv_shortcut
        # if self.in_channels != self.out_channels:
        #     if self.use_conv_shortcut:
        #         self.conv_shortcut = nn.Conv3d(in_channels,
        #                                        out_channels,
        #                                        kernel_size=(1, 3, 3),
        #                                        stride=1,
        #                                        padding=(0, 1, 1))
        #     else:
        #         self.nin_shortcut = nn.Conv3d(in_channels,
        #                                        out_channels,
        #                                        kernel_size=(1, 1, 1),
        #                                        stride=1,
        #                                        padding=(0, 0, 0))
    def forward(self, x):
        b, c, t, h, w = x.shape
        h = x
        h = rearrange(h, 'b c t h w -> (b t) c h w')
        h = self.conv1(h)
        h = rearrange(h, '(b t) c h w -> b c t h w', t=t)
        h = self.conv2(h)
        
        # if self.in_channels != self.out_channels:
        #     if self.use_conv_shortcut:
        #         x = self.conv_shortcut(x)
        #     else:
        #         x = self.nin_shortcut(x)
        
        return h 

    
class SpatialTemporalResBlock(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 dropout,
                 temb_channels=512, 
                 conv_shortcut=False):
        super().__init__()
        # channels definition
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.use_conv_shortcut = conv_shortcut
        
        
        self.dropout = nn.Dropout(dropout)
        
        self.temb_projection = nn.Linear(temb_channels, out_channels)
        self.norm1 = Normalize(in_channels)
        self.st_conv1 = SpatialTemporalConv(in_channels=in_channels,
                                            out_channels=out_channels,
                                            )
        self.norm2 = Normalize(out_channels)
        self.st_conv2 = SpatialTemporalConv(in_channels=out_channels,
                                            out_channels=out_channels,
                                            )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv3d(in_channels,
                                               out_channels,
                                               kernel_size=(1, 3, 3),
                                               stride=1,
                                               padding=(0, 1, 1))
            else:
                self.nin_shortcut = nn.Conv3d(in_channels,
                                               out_channels,
                                               kernel_size=(1, 1, 1),
                                               stride=1,
                                               padding=(0, 0, 0))
                
    def forward(self, x, temb):
        h = x  # the x should be like (b, c, frame, h, w)
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.st_conv1(h)  # the shape of h should be like (b, c, frame, h, w)
        
        h = h + self.temb_projection(nonlinearity(temb))[:, :, None, None, None]
        
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.st_conv2(h)
        
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return h +x
    
class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):  
        b, c, t, h, w = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t = t)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)
    def forward(self, x):
        b, c, t, h, w = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
            #x = self.dwt(x)[0]
        x = rearrange(x, '(b t) c h w -> b c t h w', t = t)
        return x


class SpatialAttention(nn.Module):
    def  __init__(self, in_channels, heads=8):
        super().__init__()
        self.heads = heads
        #hidden_dim = dim_head * heads
        dim_head = in_channels // heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Conv2d(in_channels, in_channels*3 , kernel_size=1, bias=False)
        self.to_out = nn.Conv2d(in_channels, in_channels, 1)
        
    def forward(self, x):
        b, c, t, h, w = x.shape  
        x = rearrange(x, 'b c f h w -> (b f) c h w')  
        qkv = self.to_qkv(x).chunk(3, dim=1)  
        q, k, v = rearrange_many(qkv, 'b (h c) x y -> b h c (x y)', h=self.heads) 
        q = q.softmax(dim = -2) 
        k = k.softmax(dim = -1)
        
        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)  
        
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)  
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        out = self.to_out(out)
        return rearrange(out, '(b f) c h w -> b c f h w', b=b)
    
class TemporalAttention(nn.Module):
    def __init__(self, in_channels, heads=8, rotary_emb=None):
        super().__init__()
        
        dim_head = in_channels // heads
        self.scale = dim_head ** -0.5
        self.heads = heads  
        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(in_channels, in_channels * 3, bias = False)
        self.to_out = nn.Linear(in_channels, in_channels, bias = False)

    def forward(self, x, pos_bias = None, focus_present_mask = None):
        n, device = x.shape[-2], x.device

        qkv = self.to_qkv(x).chunk(3, dim = -1)

        if exists(focus_present_mask) and focus_present_mask.all():
            # if all batch samples are focusing on present
            # it would be equivalent to passing that token's values through to the output
            values = qkv[-1]
            return self.to_out(values)

        # split out heads
        q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h = self.heads)

        # scale
        q = q * self.scale
        # rotate positions into queries and keys for time attention
        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # similarity

        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)

        # relative positional bias

        if exists(pos_bias):
            sim = sim + pos_bias

        if exists(focus_present_mask) and not (~focus_present_mask).all():
            attend_all_mask = torch.ones((n, n), device = device, dtype = torch.bool)
            attend_self_mask = torch.eye(n, device = device, dtype = torch.bool)

            mask = torch.where(
                rearrange(focus_present_mask, 'b -> b 1 1 1 1'),
                rearrange(attend_self_mask, 'i j -> 1 1 1 i j'),
                rearrange(attend_all_mask, 'i j -> 1 1 1 i j'),
            )

            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # numerical stability

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        # aggregate values

        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '... h n d -> ... n (h d)')
        return self.to_out(out)
    
class SpatialTemporalAttention(nn.Module):
    def __init__(self, num_head, in_channels, rotary_emb=None):
        super(SpatialTemporalAttention, self).__init__()
        self.num_head = num_head
        self.in_channels = in_channels
        self.rotary_emb = rotary_emb
        self.attention1 = SpatialAttention(in_channels=self.in_channels, heads=self.num_head)
        self.attention2 = EinopsToAndFrom('b c f h w', 
                                          'b (h w) f c', 
                                          TemporalAttention(in_channels, 
                                                            heads = num_head,  
                                                            rotary_emb = self.rotary_emb))
    
    def forward(self, x):
        h = x
        h = self.attention1(h)
        h = self.attention2(h)
        return x + h
    
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)  # 通道基数 输出通道数， 通道乘子
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        in_channels = config.model.in_channels
        resolution = config.data.image_size
        resamp_with_conv = config.model.resamp_with_conv
        num_timesteps = config.diffusion.num_diffusion_timesteps
        
        if config.model.type == 'bayesian':
            self.logvar = nn.Parameter(torch.zeros(num_timesteps))
        
        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.num_head = config.model.num_head
        
        
        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch,
                            self.temb_ch),
            torch.nn.Linear(self.temb_ch,
                            self.temb_ch),
        ])

        # downsampling
        # self.conv_in = torch.nn.Conv2d(in_channels,
        #                                self.ch,
        #                                kernel_size=3,
        #                                stride=1,
        #                                padding=1)
        self.conv_in = torch.nn.Conv3d(in_channels,
                                       self.ch,
                                       kernel_size=(1, 3, 3),
                                       stride=1,
                                       padding=(0, 1,1))

        curr_res = resolution
        in_ch_mult = (1,)+ch_mult

        self.down = nn.ModuleList()

        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(SpatialTemporalResBlock(
                                        in_channels=block_in,
                                        out_channels=block_out,
                                        temb_channels=self.temb_ch,
                                        dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(SpatialTemporalAttention(num_head=self.num_head, 
                                                         in_channels=block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = SpatialTemporalResBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = SpatialTemporalAttention(num_head=self.num_head, 
                                                         in_channels=block_in)
        self.mid.block_2 = SpatialTemporalResBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        
        
        # self.con_embedding = ConditionEmbedding(in_channels=128, 
        #                                         out_channels=block_in,  # 512
        #                                         temb_channels=self.temb_ch,   # 512
        #                                         dropout=dropout)
        
        # self.spatial_transformer = SpatialTransformer(in_channels=block_in,
        #                                               n_heads=8,
        #                                               d_head=64,
        #                                               context_dim=512
        #                                               )
        
        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(SpatialTemporalResBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(SpatialTemporalAttention(num_head=self.num_head, 
                                                         in_channels=block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        # self.conv_out = torch.nn.Conv2d(block_in,
        #                                 out_ch,
        #                                 kernel_size=3,
        #                                 stride=1,
        #                                 padding=1)
        self.conv_out = torch.nn.Conv3d(block_in,
                                        out_ch,
                                        kernel_size=(1, 3, 3),
                                        stride=1,
                                        padding=(0, 1, 1))

    def forward(self, x, t):
        # the shape of x should be like (b 2 t h w)
        assert x.shape[-1] == x.shape[-2] == self.resolution
        # x, y = x[:,0,...], x[:,1,...]
        # x, y = x.unsqueeze(1).type(torch.float), y.unsqueeze(1).type(torch.float)
    
  
        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        
        # condition embedding 
        # in_y = self.conv_in(y)  # [B, 128, 128, 128]   the target resolution is B, 512, 4, 4
        # condition_embedding = self.con_embedding(in_y, temb)
        
        
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        
        # comput the cross-attention between the condition embedding and the feature map
        #print('before spatial transformer', h.shape, condition_embedding.shape)
        # h = self.spatial_transformer(h, condition_embedding) 
        #print('after spatial transformer', h.shape)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

if __name__ == "__main__":
    import yaml
    from argparse import Namespace
    module_path = os.path.abspath(os.path.join('..', 'einops_exts'))
    if module_path not in sys.path:
        sys.path.append(module_path)
        
    def dict2namespace(config):
        namespace = Namespace()
        for key, value in config.items():
            if isinstance(value, dict):
                new_value = dict2namespace(value)
            else:
                new_value = value
            setattr(namespace, key, new_value)
        return namespace

    with open('/data/liaohx/Cine_Super_Resolution/Models/DDIM_wavelet/configs/cine.yml', "r") as f:
        config = yaml.safe_load(f)

    new_config = dict2namespace(config)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    x = torch.randn(1, 2, 30, 128, 128).to(device)
    
    n = x.shape[0]
    t = torch.randint(low=0, high=2000, size=(n // 2 + 1,))
    t = torch.cat([t, 2000 - t - 1], dim=0)[:n].to(device)
    net = Model(new_config).to(device)
    print(x.device, t.device)
    y = net(x, t)
    print(y.shape)
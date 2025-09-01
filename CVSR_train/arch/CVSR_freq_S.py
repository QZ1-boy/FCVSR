import torch
import torch.nn as nn
# from basicsr.models.archs import recons_video81 as recons_video
# from basicsr.models.archs import flow_pwc82 as flow_pwc
import numpy as np
from torch.nn import functional as F
import torch.utils.checkpoint as checkpoint
import numbers
from einops import rearrange
import torch.nn.init as init

try:
    from torch import irfft
    from torch import rfft
except ImportError:
    from torch.fft import irfft2
    from torch.fft import rfft2
    def rfft(x, d):
        t = rfft2(x, dim = (-d))
        return torch.stack((t.real, t.imag), -1)
    def irfft(x, d, signal_sizes):
        return irfft2(torch.complex(x[:,:,0], x[:,:,1]), s = signal_sizes, dim = (-d))


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

def make_model(opt):
    device = 'cuda'
    load_flow_net = True
    load_recons_net = False
    flow_pretrain_fn = opt['pretrain_models_dir'] + 'network-default.pytorch'
    recons_pretrain_fn = ''
    is_mask_filter = True
    return GShiftNet()
class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
class CALayer2(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer2, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
def make_layer(block, num_blocks, **kwarg):
    """Make layers by stacking the same blocks.
    Args:
        block (nn.module): nn.module class for basic block.
        num_blocks (int): number of blocks.
    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_blocks):
        layers.append(block(**kwarg))
    return nn.Sequential(*layers)
class ResidualBlockNoBN(nn.Module):
    def __init__(self, mid_channels=64, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=False)
        self.relu = nn.PReLU() # nn.ReLU(inplace=True)
        # if res_scale == 1.0:
        #     self.init_weights()
        self.CA = CALayer(mid_channels, 4, bias=False)

    def init_weights(self):
        for m in [self.conv1, self.conv2]:
            default_init_weights(m, 0.1)

    def forward(self, x):
        identity = x
        out = self.CA(self.conv2(self.relu(self.conv1(x))))
        return identity + out * self.res_scale

class ResidualBlocksWithInputConv(nn.Module):
    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()
        main = []
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False))
        # main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        main.append(nn.PReLU())
        # residual blocks
        main.append(
            make_layer(
                ResidualBlockNoBN, num_blocks, mid_channels=out_channels))
        self.main = nn.Sequential(*main)

    def forward(self, feat):
        return self.main(feat)
def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)
    
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res
class RepConv(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(RepConv, self).__init__()
        self.conv_1 = nn.Conv2d(n_feat, n_feat, kernel_size, bias=bias, padding=kernel_size//2, groups=n_feat//8)
        self.conv_2 = nn.Conv2d(n_feat, n_feat, 3, bias=bias, padding=1, groups=n_feat//8)
    def forward(self, x):
        res_1 = self.conv_1(x)
        res_2 = self.conv_2(x)
        return res_1 + res_2 + x
class RepConv2(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(RepConv2, self).__init__()
        #self.conv_1 = nn.Conv2d(n_feat, n_feat, kernel_size, bias=bias, padding=kernel_size//2, groups=n_feat//8)
        self.conv_2 = nn.Conv2d(n_feat, n_feat, 3, bias=bias, padding=1, groups=n_feat)
    def forward(self, x):
        #res_1 = self.conv_1(x)
        res_2 = self.conv_2(x)
        return res_2 + x
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
class SimpleGate2(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * torch.sigmoid(x2)
class CAB1(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB1, self).__init__()
        modules_body = []
        scale_factor = 1
        n_scale_feat = int(scale_factor * n_feat)
        self.norm = LayerNorm2d(n_feat)
        modules_body.append(conv(n_feat, n_scale_feat*2, 1, bias=bias))
        # modules_body.append(nn.GELU())
        # modules_body.append(nn.PReLU())
        modules_body.append(RepConv2(n_scale_feat*2, kernel_size, bias))
        # modules_body.append(nn.Conv2d(n_scale_feat*2, n_scale_feat*2, 3, bias=bias, padding=1, groups=n_scale_feat*2))
        modules_body.append(SimpleGate())
        # modules_body.append(CALayer2(n_scale_feat, reduction, bias=bias))
        modules_body.append(RepConv(n_scale_feat, kernel_size, bias))
        modules_body.append(conv(n_scale_feat, 2*n_scale_feat, 1, bias=bias))
        modules_body.append(SimpleGate2())
        modules_body.append(CALayer2(n_feat, reduction, bias=bias))
        modules_body.append(conv(n_scale_feat, n_feat, 1, bias=bias))
        # self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)
        self.beta = nn.Parameter(torch.zeros((1, n_feat, 1, 1)), requires_grad=True)

    def forward(self, x):
        res = self.body(self.norm(x))
        # res = self.CA(res)
        res = x + res * self.beta
        return res
class CAB2(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act, add_channel=0):
        super(CAB2, self).__init__()
        modules_body = []
        scale_factor = 1
        self.n_feat = n_feat
        self.add_channel = add_channel
        n_scale_feat = int(scale_factor * n_feat)
        self.conv1 = nn.Conv2d(self.add_channel, self.add_channel, 3, bias=bias, padding=1, groups=self.add_channel)
        self.norm = LayerNorm2d(self.add_channel + n_feat)
        modules_body.append(conv(n_feat + self.add_channel, n_scale_feat*2, 1, bias=bias))
        modules_body.append(RepConv2(n_scale_feat*2, kernel_size, bias))
        modules_body.append(SimpleGate())
        modules_body.append(RepConv(n_scale_feat, kernel_size, bias))
        modules_body.append(conv(n_scale_feat, 2*n_scale_feat, 1, bias=bias))
        modules_body.append(SimpleGate2())
        modules_body.append(CALayer2(n_feat, reduction, bias=bias))
        modules_body.append(conv(n_scale_feat, n_feat, 1, bias=bias))
        self.body = nn.Sequential(*modules_body)
        self.beta = nn.Parameter(torch.zeros((1, n_feat, 1, 1)), requires_grad=True)

    def forward(self, x_input):
        shortcut, hw = x_input[:,0:self.n_feat], x_input[:,self.n_feat:]
        hw = self.conv1(hw)
        res = self.body(self.norm(torch.cat((shortcut, hw), dim=1)))
        res = shortcut + res * self.beta
        return res
class PixelShufflePack(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor,
                 upsample_kernel):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        self.upsample_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels * scale_factor * scale_factor,
            self.upsample_kernel,
            padding=(self.upsample_kernel - 1) // 2)
        # self.init_weights()

    def init_weights(self):
        default_init_weights(self, 1)

    def forward(self, x):
        x = self.upsample_conv(x)
        x = F.pixel_shuffle(x, self.scale_factor)
        return x
## Original Resolution Block (ORB)
class CABs(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super(CABs, self).__init__()
        modules_body = []
        modules_body = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_cab)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=True)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)

class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, 3, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1+x
        return x1, img
def conv5x5(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=True)
# RDB-based RNN cell
class shallow_cell(nn.Module):
    def __init__(self, n_features):
        super(shallow_cell, self).__init__()
        self.n_feats = n_features
        act = nn.PReLU()
        bias = False
        reduction = 4
        self.shallow_feat = nn.Sequential(conv(3, self.n_feats, 3, bias=bias),
                                           CAB(self.n_feats, 3, reduction, bias=bias, act=act))

    def forward(self,x):
        feat = self.shallow_feat(x)
        return feat
class DownSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(DownSample, self).__init__()
        # self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
        #                           nn.Conv2d(in_channels, in_channels + s_factor, 1, stride=1, padding=0, bias=False))
        # self.down = nn.Sequential(nn.Conv2d(in_channels, in_channels + s_factor, kernel_size=3, stride=2, padding=1, bias=True),
        #                    nn.PReLU())
        self.down = nn.Conv2d(in_channels, in_channels + s_factor, kernel_size=3, stride=2, padding=1, bias=True)
    def forward(self, x):
        x = self.down(x)
        return x
class SkipUpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                 nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias




class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode='bilinear'):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)



class Block(nn.Module):
    def __init__(self,
               num_residual_units,
               kernel_size,
               width_multiplier=1,
               group=4):
        super(Block, self).__init__()
        body = []
        conv = nn.Conv2d(
                num_residual_units,
                int(num_residual_units * width_multiplier),
                kernel_size,
                padding=kernel_size // 2)
        body.append(conv)
        body.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        conv = nn.Conv2d(
                int(num_residual_units * width_multiplier),
                num_residual_units,
                kernel_size,
                padding=kernel_size // 2)
        body.append(conv)
        initialize_weights(body, 0.1)
        self.body = nn.Sequential(*body)

        down = []
        down.append(nn.Conv2d(num_residual_units, num_residual_units, 1))
        down.append(Interpolate(scale_factor=0.5))
        self.down = nn.Sequential(*down)

        up = []
        up.append(nn.Conv2d(num_residual_units, num_residual_units, 1))
        up.append(Interpolate(scale_factor=2.0))
        self.up = nn.Sequential(*up)
        initialize_weights([self.up, self.down], 0.1)

    def forward(self, x_list):
        # x_list : L1 L2 L3
        # down_res_list : L1 L1/2 L2/2 
        # up_res_list: L2*2 L3*2 L3
        res_list = [self.body(x) for x in x_list]
        down_res_list = [res_list[0]] + [self.down(x) for x in res_list[:-1]]
        up_res_list = [self.up(x) for x in res_list[1:]] + [res_list[-1]]
        x_list = [
            x + r + d + u
            for x, r, d, u in zip(x_list, res_list, down_res_list, up_res_list)
        ]
        return x_list



class SCGroup(nn.Module):
    def __init__(self, nf=64, back_RBs=3):
        super(SCGroup, self).__init__()
        self.nf = nf
        self.conv = nn.Conv2d(nf, nf, 3, padding=1)
        body = []
        for _ in range(back_RBs):
            body.append(
                Block(    ### Change 
                    nf,
                    kernel_size=3,
                    width_multiplier=4
                ))
        self.body = nn.Sequential(*body)
    
    def forward(self, x_list):
        res_list = self.body(x_list)
        res_list = [self.conv(x) for x in res_list]
        x_list = [
            x + r
            for x, r in zip(x_list, res_list)
        ]
        return x_list



class SCNet(nn.Module):
    def __init__(self, nf=64, SCGroupN=4):
        super(SCNet, self).__init__()
        self.nf = nf
        body = []
        for _ in range(SCGroupN):
            body.append(SCGroup(nf=nf))
        self.body = nn.Sequential(*body)
    
    def forward(self, x_list):
        res_list = self.body(x_list)
        x_list = [
            x + r
            for x, r in zip(x_list, res_list)
        ]
        return x_list





class FSAS(nn.Module):
    def __init__(self, dim, bias=False):
        super(FSAS, self).__init__()
        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=bias)
        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.norm = LayerNorm(dim * 2, LayerNorm_type='WithBias')
        self.patch_size = 8

    def forward(self, x):
        hidden = self.to_hidden(x)
        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)
        q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        q_fft = torch.fft.rfft2(q_patch.float())
        k_fft = torch.fft.rfft2(k_patch.float())
        out = q_fft * k_fft
        out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))
        out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                        patch2=self.patch_size)
        out = self.norm(out)
        output = v * out
        output = self.project_out(output)

        return output



class FSAS_freq(nn.Module):
    def __init__(self, dim, bias=False, add_channel=None):
        super(FSAS_freq, self).__init__()
        self.add_channel = add_channel
        self.n_feat = dim
        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=bias)
        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.norm = LayerNorm(dim * 2, LayerNorm_type='WithBias')
        self.patch_size = 8
        self.conv1 = nn.Conv2d(dim+ self.add_channel, dim + self.add_channel, 3, bias=bias, padding=1, groups=dim+self.add_channel)
    def forward(self, x):
        # print('x',x.shape,self.n_feat+self.add_channel )
        hw, shortcut = x[:,0:self.n_feat], x[:,self.n_feat:]
        hidden = self.to_hidden(hw)
        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)
        q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        v_patch = rearrange(v, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        q_fft = torch.fft.rfft2(q_patch.float())
        k_fft = torch.fft.rfft2(k_patch.float())
        v_fft = torch.fft.rfft2(v_patch.float())
        out1 = q_fft * k_fft
        out2 = v_fft * k_fft
        out = out1 * out2
        out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))
        out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                        patch2=self.patch_size)
        out = self.norm(out)
        # output = v * out
        output = self.project_out(out) + hw
        return output



class FFT_spital_module(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.spatial_branch = nn.Sequential(
                nn.Conv2d(dim, dim // 2, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 2, dim // 2, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 2, dim, 3, 1, 1))
        self.conv = nn.Sequential(  nn.Conv2d(dim, dim, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv1_1 =  nn.Conv2d(dim*2, dim, 1, 1, 0)
        self.norm = 'backward'
        self.fft_branch = nn.Sequential(
                nn.Conv2d(dim*2, dim*2, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim*2 , dim*2 , 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True))

    def forward(self, x):
        spatial_output =  self.spatial_branch(x)
        x = self.conv(x)
        _, _, H, W = x.shape
        #print(x[0].shape)
        dim = 1
        #print( '============== ' , input)
        input_fft=torch.fft.rfft2(x, norm=self.norm) 
        y_imag = input_fft.imag
        #print('xxxxxx ' ,y_imag.shape)
        y_real = input_fft.real
        y_f = torch.cat([y_real, y_imag], dim=dim)   
        #print('=========' , y_f.shape)     
        fft_output = self.fft_branch(y_f)
        y_real, y_imag = torch.chunk(fft_output, 2, dim=dim)
        fft_output = torch.complex(y_real, y_imag)
        fft_output = torch.fft.irfft2(fft_output, s=(H, W), norm=self.norm)

        final_out =self.conv1_1(torch.cat((fft_output , spatial_output) , 1))
        return final_out

class TFDC(nn.Module):
    def __init__(self, dim, bias=False, add_channel=None):
        super(TFDC, self).__init__()
        base_filter = 64
        self.dim = dim
        self.norm = 'backward'
        
        self.conv3 = self.conv2 = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=bias),
        )
        self.conv4 = self.conv2 = nn.Sequential(
                nn.Conv2d(2*dim, 2*dim, kernel_size=3, padding=1, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(2*dim, 2*dim, kernel_size=3, padding=1, bias=bias),
        )

        # self.avg_diff = nn.AvgPool2d(kernel_size=2, stride=2)  # 池化降采样2倍
        self.sigmoid = nn.Sigmoid()
        act = nn.PReLU()
        self.CAB2 = CAB2(dim//2, 5, reduction=4, bias=False, act=act, add_channel=dim//2)
        self.conv7 = nn.Conv2d(dim//2, dim, kernel_size=3, padding=1, bias=bias)
        # self.conv8 = nn.Conv2d(3*dim//2, dim, kernel_size=3, padding=1, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        x1 = x[:, 0:1*self.dim,:,:]
        x2 = x[:, 1*self.dim: 2*self.dim,:,:]
        x3 = x[:, 2*self.dim:3*self.dim,:,:]

        x1_3 = self.conv3(x[:, 0:1*self.dim,:,:])
        x2_3 = self.conv3(x[:, 1*self.dim: 2*self.dim,:,:])
        x3_3 = self.conv3(x[:, 2*self.dim:3*self.dim,:,:])
        x1_fft = torch.fft.rfft2(x1_3, norm=self.norm)
        x2_fft = torch.fft.rfft2(x2_3, norm=self.norm)
        x3_fft = torch.fft.rfft2(x3_3, norm=self.norm)
        x1_imag = x1_fft.imag
        x1_real = x1_fft.real
        x1_f = torch.cat([x1_imag, x1_real], dim=1)
        x2_imag = x2_fft.imag
        x2_real = x2_fft.real
        x2_f = torch.cat([x2_imag, x2_real], dim=1)
        x3_imag = x3_fft.imag
        x3_real = x3_fft.real
        x3_f = torch.cat([x3_imag, x3_real], dim=1)
        diff_21 = x1_f - x2_f
        diff_23 = x3_f - x2_f

        enhance_f3 = self.conv4(diff_21)
        enhance_b3 = self.conv4(diff_23)

        f3 = self.sigmoid(self.conv4(diff_21 + enhance_f3 ))
        b3 = self.sigmoid(self.conv4(diff_23 + enhance_b3 ))
        outfreq = x2_f * f3 + x2_f * b3 + x2_f

        y_real, y_imag = torch.chunk(outfreq, 2, dim=1)
        fft_output = torch.complex(y_real, y_imag)
        fft_output = torch.fft.irfft2(fft_output, s=(H, W), norm=self.norm)
        out_3 = self.conv7(self.CAB2(fft_output)) # ) + x2
        out = out_3 + x2

        return out


class Encoder(nn.Module):
    def __init__(self, n_features, kernel_size=3, reduction=4, bias=False, scale_unetfeats=48):
        super(Encoder, self).__init__()
        n_feat = n_features
        scale_unetfeats = int(n_feat/2)
        act = nn.PReLU()
        self.encoder_level1 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level2 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]
        self.encoder_level3 = [CAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)

        self.down12 = DownSample(n_feat, scale_unetfeats)
        self.down23 = DownSample(n_feat + scale_unetfeats, scale_unetfeats)


    def forward(self, x, encoder_outs=None, decoder_outs=None):
        enc1 = self.encoder_level1(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc1 = enc1 + self.skip_enc1(encoder_outs[0]) + self.skip_dec1(decoder_outs[0])

        x = self.down12(enc1)

        enc2 = self.encoder_level2(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc2 = enc2 + self.skip_enc2(encoder_outs[1]) + self.skip_dec2(decoder_outs[1])

        x = self.down23(enc2)

        enc3 = self.encoder_level3(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc3 = enc3 + self.skip_enc3(encoder_outs[2]) + self.skip_dec3(decoder_outs[2])

        return [enc1, enc2, enc3]


def generate_kernels(h=11,l=80,n=10):
    kernels = torch.zeros(l,1,h,h).to(torch.device('cuda'))
    n2 = 2
    n1 = n-2*n2
    kernels[0*n2:1*n2,:,0,0] = 1
    kernels[1*n2:2*n2,:,0,h//4] = 1
    kernels[2*n2:3*n2,:,0,h//2] = 1
    kernels[3*n2:4*n2,:,0,3*h//4] = 1
    kernels[4*n2:5*n2,:,0,h-1] = 1
    kernels[5*n2:6*n2,:,h-1,0] = 1
    kernels[6*n2:7*n2,:,h-1,h//4] = 1
    kernels[7*n2:8*n2,:,h-1,h//2] = 1
    kernels[8*n2:9*n2,:,h-1,3*h//4] = 1
    kernels[9*n2:10*n2,:,h-1,h-1] = 1
    kernels[10*n2:11*n2,:,h//4,0] = 1
    kernels[11*n2:12*n2,:,h//4,h-1] = 1
    kernels[12*n2:13*n2,:,h//2,0] = 1
    kernels[13*n2:14*n2,:,h//2,h-1] = 1
    kernels[14*n2:15*n2,:,3*h//4,0] = 1
    kernels[15*n2:16*n2,:,3*h//4,h-1] = 1
    kernels[16*n2+0*n1:16*n2+1*n1,:,h//4,h//4] = 1
    kernels[16*n2+1*n1:16*n2+2*n1,:,h//4,h//2] = 1
    kernels[16*n2+2*n1:16*n2+3*n1,:,h//4,3*h//4] = 1 
    kernels[16*n2+3*n1:16*n2+4*n1,:,h//2,h//4] = 1
    kernels[16*n2+4*n1:16*n2+5*n1,:,h//2,3*h//4] = 1
    kernels[16*n2+5*n1:16*n2+6*n1,:,3*h//4,h//4] = 1
    kernels[16*n2+6*n1:16*n2+7*n1,:,3*h//4,h//2] = 1
    kernels[16*n2+7*n1:16*n2+8*n1,:,3*h//4,3*h//4] = 1
    return kernels


class FourierUnit1(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        # bn_layer not used
        super(FourierUnit1, self).__init__()
        self.groups = groups
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        batch, c, h, w = x.size()
        r_size = x.size()
        # (batch, c, h, w/2+1, 2)
        ffted = torch.fft.rfft2(x)
        # ffted = torch.rfft(x, signal_ndim=2, normalized=True)
        # ffted = torch.fft.rfft(x)
        # (batch, c, 2, h, w/2+1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        # output = torch.irfft(ffted, signal_ndim=2, signal_sizes=r_size[2:], normalized=True)
        output = torch.fft.ifft2(ffted, dim=(-2, -1))
        return output


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduction = 2
        bias = False
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(out_channels* 2, out_channels* 2 // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels* 2 // reduction, out_channels* 2, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        batch, c, h, w = x.size()
        r_size = x.size()
        # (batch, c, 2, h, w/2+1)
        ffted = torch.fft.rfftn(x, s=(h, w), dim=(2, 3), norm='ortho')
        ffted = torch.cat([ffted.real, ffted.imag], dim=1)
        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))
        # attention 
        ffted = self.avg_pool(ffted)
        # print('ffted',ffted.shape)
        ffted = self.conv_du(ffted)
        ffted = torch.tensor_split(ffted, 2, dim=1)
        ffted = torch.complex(ffted[0], ffted[1]) # (batch,c, t, h, w/2+1, 2)
        output = torch.fft.irfftn(ffted, s=(h, w), dim=(2, 3), norm='ortho')
        return output


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = nn.Conv2d(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale


##########################################################################
##---------- Selective Kernel Feature Fusion (SKFF) ----------
class SKFF(nn.Module):
    def __init__(self, in_channels, height=1,reduction=8,bias=False):
        super(SKFF, self).__init__()
        
        self.height = height
        d = max(int(in_channels/reduction),4)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.LeakyReLU(0.2))

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1,bias=bias))
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats =  inp_feats[0].shape[1]
        
        # inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats[0] - inp_feats[1]
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])
        
        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        # stx()
        attention_vectors = self.softmax(attention_vectors)
        
        feats_V = torch.sum(inp_feats*attention_vectors, dim=1)
        
        return feats_V        


class Spa_freqblock(nn.Module):
    def __init__(self, dim, bias=False, add_channel=None):
        super(Spa_freqblock, self).__init__()
        self.FourierUnit1 = FourierUnit(in_channels=dim, out_channels=dim)
        self.SpatialAttention1 = SpatialAttention()
        self.skff1 = SKFF(dim)

        self.FourierUnit2 = FourierUnit(in_channels=dim, out_channels=dim)
        self.SpatialAttention2 = SpatialAttention()
        self.skff2 = SKFF(dim)

        self.FourierUnit3 = FourierUnit(in_channels=dim, out_channels=dim)
        self.SpatialAttention3 = SpatialAttention()
        self.skff3 = SKFF(dim)

        # self.FourierUnit4 = FourierUnit(in_channels=dim, out_channels=dim)
        # self.SpatialAttention4 = SpatialAttention()
        # self.skff4 = SKFF(dim)

    def forward(self, x):
        freq_x1 = self.FourierUnit1(x)
        spa_x1 = self.SpatialAttention1(x)
        output1 = self.skff1([freq_x1, spa_x1])

        freq_x2 = self.FourierUnit2(freq_x1 + output1) 
        spa_x2 = self.SpatialAttention2(spa_x1 + output1) 
        output2 = self.skff2([freq_x2, spa_x2])

        freq_x3 = self.FourierUnit3(freq_x2 + output2) 
        spa_x3 = self.SpatialAttention3(spa_x2 + output2) 
        output3 = self.skff3([freq_x3, spa_x3])

        # freq_x4 = self.FourierUnit4(freq_x3 + output3) 
        # spa_x4 = self.SpatialAttention4(spa_x3 + output3) 
        # output = self.skff4([freq_x4, spa_x4]) 
        output = output3 + x

        return output


class Encoder_shift_block(nn.Module):
    def __init__(self, n_features, kernel_size, reduction, bias=False, scale_unetfeats=48):
        super(Encoder_shift_block, self).__init__()
        n_feat = n_features
        scale_unetfeats = int(n_feat/2)
        act = nn.PReLU()
        number = n_feat // 2 // 8
        self.number = number
        #  modify CAB2 and CAB1
        self.encoder_level1 = [CAB2(n_feat, 5, reduction, bias=bias, act=act, add_channel=8*self.number), CAB1(n_feat, 5, reduction, bias=bias, act=act)]
        self.encoder_level1_1 = [CAB2(n_feat, 5, reduction, bias=bias, act=act, add_channel=8*self.number), CAB1(n_feat, 5, reduction, bias=bias, act=act)]
        self.encoder_level1_2 = [CAB2(n_feat, 5, reduction, bias=bias, act=act, add_channel=8*self.number) , CAB1(n_feat, 5, reduction, bias=bias, act=act)]
        self.encoder_level1_3 = [CAB2(n_feat, 5, reduction, bias=bias, act=act, add_channel=8*self.number), CAB1(n_feat, 5, reduction, bias=bias, act=act)]
        self.encoder_level1_4 = [CAB2(n_feat, 5, reduction, bias=bias, act=act, add_channel=8*self.number) , CAB1(n_feat, 5, reduction, bias=bias, act=act)]
        self.encoder_level1_5 = [CAB2(n_feat, 5, reduction, bias=bias, act=act, add_channel=8*self.number), CAB1(n_feat, 5, reduction, bias=bias, act=act)]
        self.encoder_level1_6 = [CAB2(n_feat, 5, reduction, bias=bias, act=act, add_channel=8*self.number) , CAB1(n_feat, 5, reduction, bias=bias, act=act)]
        self.encoder_level1_7 = [CAB2(n_feat, 5, reduction, bias=bias, act=act, add_channel=8*self.number), CAB1(n_feat, 5, reduction, bias=bias, act=act)]
        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level1_1 = nn.Sequential(*self.encoder_level1_1)
        self.encoder_level1_2 = nn.Sequential(*self.encoder_level1_2)
        self.encoder_level1_3 = nn.Sequential(*self.encoder_level1_3)
        self.encoder_level1_4 = nn.Sequential(*self.encoder_level1_4)
        self.encoder_level1_5 = nn.Sequential(*self.encoder_level1_5)
        self.encoder_level1_6 = nn.Sequential(*self.encoder_level1_6)
        self.encoder_level1_7 = nn.Sequential(*self.encoder_level1_7)

    def spatial_shift2(self, hw):
        n2 = (self.number-1)//2
        n1 = self.number-2*n2
        s = 4
        s_list = []
        _, _, H, W = hw.shape
        s_out = torch.zeros_like(hw)
        s_out[:,0*n2:1*n2,s*2:,s*2:] = hw[:,0*n2:1*n2,:-s*2,:-s*2]
        s_out[:,1*n2:2*n2,s*2:,s:] = hw[:,1*n2:2*n2,:-s*2,:-s]
        s_out[:,2*n2:3*n2,s*2:,0:] = hw[:,2*n2:3*n2,:-s*2,:]
        s_out[:,3*n2:4*n2,s*2:,0:-s] = hw[:,3*n2:4*n2,:-s*2,s:]
        s_out[:,4*n2:5*n2,s*2:,0:-s*2] = hw[:,4*n2:5*n2,:-s*2,s*2:]

        s_out[:,5*n2:6*n2,0:-s*2,s*2:] = hw[:,5*n2:6*n2,s*2:,:-s*2]
        s_out[:,6*n2:7*n2,0:-s*2,s:] = hw[:,6*n2:7*n2,s*2:,:-s]
        s_out[:,7*n2:8*n2,0:-s*2,0:] = hw[:,7*n2:8*n2,s*2:,:]
        s_out[:,8*n2:9*n2,0:-s*2,0:-s] = hw[:,8*n2:9*n2,s*2:,s:]
        s_out[:,9*n2:10*n2,0:-s*2,0:-s*2] = hw[:,9*n2:10*n2,s*2:,s*2:]

        s_out[:,10*n2:11*n2,s:,s*2:] = hw[:,10*n2:11*n2,  :-s,:-s*2]
        s_out[:,11*n2:12*n2,s:,0:-s*2] = hw[:,11*n2:12*n2,:-s,s*2:]
        s_out[:,12*n2:13*n2,:,s*2:] = hw[:,12*n2:13*n2,  :,:-s*2]
        s_out[:,13*n2:14*n2,:,0:-s*2] = hw[:,13*n2:14*n2,:,s*2:]
        s_out[:,14*n2:15*n2,0:-s,s*2:] = hw[:,14*n2:15*n2,  s:,:-s*2]
        s_out[:,15*n2:16*n2,0:-s,0:-s*2] = hw[:,15*n2:16*n2,s:,s*2:]
        s_out[:,16*n2+0*n1:16*n2+1*n1,s:,s:] = hw[:,16*n2+0*n1:16*n2+1*n1,:-s,:-s]
        s_out[:,16*n2+1*n1:16*n2+2*n1,s:,0:] = hw[:,16*n2+1*n1:16*n2+2*n1,:-s,:]
        s_out[:,16*n2+2*n1:16*n2+3*n1,s:,0:-s] = hw[:,16*n2+2*n1:16*n2+3*n1,:-s,s:]
        s_out[:,16*n2+3*n1:16*n2+4*n1,:,s:] = hw[:,16*n2+3*n1:16*n2+4*n1,:,:-s]
        s_out[:,16*n2+4*n1:16*n2+5*n1,:,0:-s] = hw[:,16*n2+4*n1:16*n2+5*n1,:,s:]
        s_out[:,16*n2+5*n1:16*n2+6*n1,0:-s,s:] = hw[:,16*n2+5*n1:16*n2+6*n1,s:,:-s]
        s_out[:,16*n2+6*n1:16*n2+7*n1,0:-s,0:] = hw[:,16*n2+6*n1:16*n2+7*n1,s:,:]
        s_out[:,16*n2+7*n1:16*n2+8*n1,0:-s,0:-s] = hw[:,16*n2+7*n1:16*n2+8*n1,s:,s:]
        return s_out 
    def channel_shift(self, x, div=2, reverse=False):
        B, C, H, W = x.shape
        slice_c = C // div
        if reverse:
            slice_c = -slice_c
        y1 = x.view(1,B*C,H,W)
        y1 = torch.roll(y1, slice_c,1).view(B,C,H,W)
        kernel_size = 5
        if reverse == False:
            y = torch.cat((x[0:1], y1[1:]), dim=0)
            hw = y[:,0:8*self.number,...]
            # other = y[:,slice_c:,...]
        else:
            y = torch.cat((y1[0:-1],x[-1:]), dim=0)
            hw = y[:,-8*self.number:,...]
            # other = y[:,0:-slice_c,...]
        # hw_shifts = self.shift_conv1(hw)
        hw = self.spatial_shift2(hw)
        return torch.cat((y, hw), dim=1)

    def forward(self, x, reverse=0):
        x = self.channel_shift(x)
        x = self.encoder_level1(x)
        x = self.channel_shift(x, reverse=True)
        x = self.encoder_level1_1(x)
        x = self.channel_shift(x)
        x = self.encoder_level1_2(x)
        x = self.channel_shift(x, reverse=True)
        x = self.encoder_level1_3(x)
        x = self.channel_shift(x)
        x = self.encoder_level1_4(x)
        x = self.channel_shift(x, reverse=True)
        x = self.encoder_level1_5(x)
        x = self.channel_shift(x)
        x = self.encoder_level1_6(x)
        x = self.channel_shift(x, reverse=True)
        x = self.encoder_level1_7(x)
        return x


class Encoder_shift_block_1(nn.Module):
    def __init__(self, n_features, kernel_size, reduction, bias=False, scale_unetfeats=48):
        super(Encoder_shift_block_1, self).__init__()
        n_feat = n_features
        scale_unetfeats = int(n_feat/2)
        act = nn.PReLU()
        number = n_feat // 2 // 8
        self.number = number
        #  modify CAB2 and CAB1
        self.encoder_level1 = [FSAS_freq(n_feat, add_channel=8*self.number), CAB1(n_feat, 5, reduction, bias=bias, act=act)]
        self.encoder_level1_1 = [FSAS_freq(n_feat,add_channel=8*self.number), CAB1(n_feat, 5, reduction, bias=bias, act=act)]
        self.encoder_level1_2 = [FSAS_freq(n_feat,add_channel=8*self.number), CAB1(n_feat, 5, reduction, bias=bias, act=act)]
        self.encoder_level1_3 = [FSAS_freq(n_feat,add_channel=8*self.number), CAB1(n_feat, 5, reduction, bias=bias, act=act)]
        self.encoder_level1_4 = [FSAS_freq(n_feat,add_channel=8*self.number), CAB1(n_feat, 5, reduction, bias=bias, act=act)]
        self.encoder_level1_5 = [FSAS_freq(n_feat,add_channel=8*self.number), CAB1(n_feat, 5, reduction, bias=bias, act=act)]
        self.encoder_level1_6 = [FSAS_freq(n_feat,add_channel=8*self.number), CAB1(n_feat, 5, reduction, bias=bias, act=act)]
        self.encoder_level1_7 = [FSAS_freq(n_feat,add_channel=8*self.number), CAB1(n_feat, 5, reduction, bias=bias, act=act)]
        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level1_1 = nn.Sequential(*self.encoder_level1_1)
        self.encoder_level1_2 = nn.Sequential(*self.encoder_level1_2)
        self.encoder_level1_3 = nn.Sequential(*self.encoder_level1_3)
        self.encoder_level1_4 = nn.Sequential(*self.encoder_level1_4)
        self.encoder_level1_5 = nn.Sequential(*self.encoder_level1_5)
        self.encoder_level1_6 = nn.Sequential(*self.encoder_level1_6)
        self.encoder_level1_7 = nn.Sequential(*self.encoder_level1_7)

    def spatial_shift2(self, hw):
        n2 = (self.number-1)//2
        n1 = self.number-2*n2
        s = 4
        s_list = []
        _, _, H, W = hw.shape
        s_out = torch.zeros_like(hw)
        s_out[:,0*n2:1*n2,s*2:,s*2:] = hw[:,0*n2:1*n2,:-s*2,:-s*2]
        s_out[:,1*n2:2*n2,s*2:,s:] = hw[:,1*n2:2*n2,:-s*2,:-s]
        s_out[:,2*n2:3*n2,s*2:,0:] = hw[:,2*n2:3*n2,:-s*2,:]
        s_out[:,3*n2:4*n2,s*2:,0:-s] = hw[:,3*n2:4*n2,:-s*2,s:]
        s_out[:,4*n2:5*n2,s*2:,0:-s*2] = hw[:,4*n2:5*n2,:-s*2,s*2:]

        s_out[:,5*n2:6*n2,0:-s*2,s*2:] = hw[:,5*n2:6*n2,s*2:,:-s*2]
        s_out[:,6*n2:7*n2,0:-s*2,s:] = hw[:,6*n2:7*n2,s*2:,:-s]
        s_out[:,7*n2:8*n2,0:-s*2,0:] = hw[:,7*n2:8*n2,s*2:,:]
        s_out[:,8*n2:9*n2,0:-s*2,0:-s] = hw[:,8*n2:9*n2,s*2:,s:]
        s_out[:,9*n2:10*n2,0:-s*2,0:-s*2] = hw[:,9*n2:10*n2,s*2:,s*2:]

        s_out[:,10*n2:11*n2,s:,s*2:] = hw[:,10*n2:11*n2,  :-s,:-s*2]
        s_out[:,11*n2:12*n2,s:,0:-s*2] = hw[:,11*n2:12*n2,:-s,s*2:]
        s_out[:,12*n2:13*n2,:,s*2:] = hw[:,12*n2:13*n2,  :,:-s*2]
        s_out[:,13*n2:14*n2,:,0:-s*2] = hw[:,13*n2:14*n2,:,s*2:]
        s_out[:,14*n2:15*n2,0:-s,s*2:] = hw[:,14*n2:15*n2,  s:,:-s*2]
        s_out[:,15*n2:16*n2,0:-s,0:-s*2] = hw[:,15*n2:16*n2,s:,s*2:]
        s_out[:,16*n2+0*n1:16*n2+1*n1,s:,s:] = hw[:,16*n2+0*n1:16*n2+1*n1,:-s,:-s]
        s_out[:,16*n2+1*n1:16*n2+2*n1,s:,0:] = hw[:,16*n2+1*n1:16*n2+2*n1,:-s,:]
        s_out[:,16*n2+2*n1:16*n2+3*n1,s:,0:-s] = hw[:,16*n2+2*n1:16*n2+3*n1,:-s,s:]
        s_out[:,16*n2+3*n1:16*n2+4*n1,:,s:] = hw[:,16*n2+3*n1:16*n2+4*n1,:,:-s]
        s_out[:,16*n2+4*n1:16*n2+5*n1,:,0:-s] = hw[:,16*n2+4*n1:16*n2+5*n1,:,s:]
        s_out[:,16*n2+5*n1:16*n2+6*n1,0:-s,s:] = hw[:,16*n2+5*n1:16*n2+6*n1,s:,:-s]
        s_out[:,16*n2+6*n1:16*n2+7*n1,0:-s,0:] = hw[:,16*n2+6*n1:16*n2+7*n1,s:,:]
        s_out[:,16*n2+7*n1:16*n2+8*n1,0:-s,0:-s] = hw[:,16*n2+7*n1:16*n2+8*n1,s:,s:]
        return s_out 
    def channel_shift(self, x, div=2, reverse=False):
        B, C, H, W = x.shape
        slice_c = C // div
        if reverse:
            slice_c = -slice_c
        y1 = x.view(1,B*C,H,W)
        y1 = torch.roll(y1, slice_c,1).view(B,C,H,W)
        kernel_size = 5
        if reverse == False:
            y = torch.cat((x[0:1], y1[1:]), dim=0)
            hw = y[:,0:8*self.number,...]
            # other = y[:,slice_c:,...]
        else:
            y = torch.cat((y1[0:-1],x[-1:]), dim=0)
            hw = y[:,-8*self.number:,...]
            # other = y[:,0:-slice_c,...]
        # hw_shifts = self.shift_conv1(hw)
        hw = self.spatial_shift2(hw)
        return torch.cat((y, hw), dim=1)

    def forward(self, x, reverse=0):
        x = self.channel_shift(x)
        x = self.encoder_level1(x)
        x = self.channel_shift(x, reverse=True)
        x = self.encoder_level1_1(x)
        x = self.channel_shift(x)
        x = self.encoder_level1_2(x)
        x = self.channel_shift(x, reverse=True)
        x = self.encoder_level1_3(x)
        x = self.channel_shift(x)
        x = self.encoder_level1_4(x)
        x = self.channel_shift(x, reverse=True)
        x = self.encoder_level1_5(x)
        x = self.channel_shift(x)
        x = self.encoder_level1_6(x)
        x = self.channel_shift(x, reverse=True)
        x = self.encoder_level1_7(x)
        return x


class Encoder2(nn.Module):
    def __init__(self, n_features, kernel_size=3, reduction=4, bias=False, scale_unetfeats=48):
        super(Encoder2, self).__init__()
        n_feat = n_features
        scale_unetfeats = 0
        act = nn.PReLU()
        n_feat0 = 24
        self.act = act
        self.encoder_level1 = CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.concat = conv(3*n_feat0, n_feat, kernel_size, bias=bias)
        self.decoder_level1 = Encoder_shift_block_1(n_feat, kernel_size, reduction, bias)  #  Encoder_shift_block_1
        self.skip_conv = CAB(n_feat, kernel_size, reduction, bias=bias, act=act) 
        self.out_conv = CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.conv_hr0 = conv(n_feat*2, n_feat, kernel_size, bias=True)
        div = 4
        self.slice_c =  n_feat // div

    def channel_shift(self, x, div=2, reverse=False):
        B, C, H, W = x.shape
        slice_c = C // div
        if reverse:
            slice_c = -slice_c
        # for kk in range(1, B):
        #     x[kk,-slice_c:] = x[kk-1,-slice_c:]
        y = x.view(1,B*C,H,W)
        return torch.roll(y, slice_c,1).view(B,C,H,W)
    def forward(self, x, reverse=False):
        x = self.concat(x)
        shortcut = x 
        enc1 = self.encoder_level1(x)
        dec1 = self.decoder_level1(enc1)
        dec11_out = self.conv_hr0(torch.cat((dec1, self.skip_conv(shortcut)), dim=1))
        dec11_out = self.out_conv(dec11_out)
        return dec11_out


class Decoder(nn.Module):
    def __init__(self, n_features, kernel_size=3, reduction=4, bias=False, scale_unetfeats=48):
        super(Decoder, self).__init__()
        n_feat = n_features
        scale_unetfeats = int(n_feat/2)
        act = nn.PReLU()
        # self.concat = conv(n_feat*2, n_feat, kernel_size, bias=bias)
        self.decoder_level1 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        # self.decoder_level1 += [conv(n_feat, 80, kernel_size, bias=bias)]
        self.decoder_level2 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]
        self.decoder_level3 = [CAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.skip_attn1 = CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.skip_attn2 = CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act)

        self.up21 = SkipUpSample(n_feat, scale_unetfeats)
        self.up32 = SkipUpSample(n_feat + scale_unetfeats, scale_unetfeats)

    def forward(self, outs):
        enc1, enc2, enc3 = outs
        dec3 = self.decoder_level3(enc3)

        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)

        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)
        

        return [dec1, dec2, dec3]


class TFR_UNet(nn.Module):
    def __init__(self, n_feat0, n_feat, kernel_size, reduction, act, bias, scale_unetfeats):
        super(TFR_UNet, self).__init__()
        scale_unetfeats = 12
        self.encoder_level1 = [CAB(n_feat0, kernel_size, reduction, bias=bias, act=act) for _ in range(1)]
        self.encoder_level2 = [CAB(n_feat0 + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in
                               range(3)]
        self.encoder_level3 = [CAB(n_feat0 + 2*scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in
                               range(3)]
        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)
        self.down12 = DownSample(n_feat0, scale_unetfeats)
        self.down23 = DownSample(n_feat0+scale_unetfeats, scale_unetfeats)

        self.decoder_level1 = [CAB(n_feat0, kernel_size, reduction, bias=bias, act=act) for _ in range(1)]
        self.decoder_level2 = [CAB(n_feat0+scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in
                               range(3)]
        self.decoder_level3 = [CAB(n_feat0+scale_unetfeats*2, kernel_size, reduction, bias=bias, act=act) for _ in
                               range(3)]
        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.skip_attn1 = CAB(n_feat0, kernel_size, reduction, bias=bias, act=act)
        self.skip_attn2 = CAB(n_feat0+scale_unetfeats, kernel_size, reduction, bias=bias, act=act)
        self.up21 = SkipUpSample(n_feat0, scale_unetfeats)
        self.up32 = SkipUpSample(n_feat0 + scale_unetfeats, scale_unetfeats)
    def forward(self, x):
        shortcut = x 
        enc1 = self.encoder_level1(x)
        x = self.down12(enc1)
        enc2 = self.encoder_level2(x)
        x = self.down23(enc2)
        enc3 = self.encoder_level3(x)

        dec3 = self.decoder_level3(enc3)
        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)
        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)
        return dec1


class GShiftNet(nn.Module):
    def __init__(self, n_features=64, future_frames=2, past_frames=2):
        super(GShiftNet, self).__init__()
        self.n_feats = n_features
        self.num_ff = future_frames
        self.num_fb = past_frames
        self.device = torch.device('cuda')
        self.n_feats0 = 64
        # self.feat_extract = nn.Sequential(nn.Conv2d(7, 7*self.n_feats0, 3, 1, 1),
        #     CAB(7*self.n_feats0, 3, 4, bias=False, act=nn.PReLU()))
        self.feat_extract = nn.Sequential(nn.Conv2d(7, 7*self.n_feats0, 3, 1, 1))
        self.lrelu = nn.PReLU() 
        self.TFDC = TFDC(dim=self.n_feats0)
        # self.orb1 = TFR_UNet(7*self.n_feats0, 7*self.n_feats0, kernel_size=3, reduction=4, act=nn.PReLU(), bias=False, scale_unetfeats=0)
        # self.recorb1 = TFR_UNet(self.n_feats0, self.n_feats0, kernel_size=3, reduction=4, act=nn.PReLU(), bias=False, scale_unetfeats=0)
        self.rconcat1 = nn.Conv2d(self.n_feats0, self.n_feats0, 3, stride=2, padding=1, bias=True)
        self.rconcat2 = nn.Conv2d(self.n_feats0, self.n_feats0, 3, stride=2, padding=1, bias=True)
        self.recorb1 = SCNet(nf=self.n_feats0, SCGroupN=3) 
        self.recorb0 = nn.Conv2d(self.n_feats0, self.n_feats0, 3, 1, 1, bias=True)
        #### upsampling
        #### fea pyramid fuse 
        self.upconv1_L2 = nn.Conv2d(self.n_feats0, self.n_feats0, 1, 1, 0, bias=True)
        self.upconv1_L2_2 = nn.Conv2d(self.n_feats0 + self.n_feats0//4, self.n_feats0, 1, 1, 0, bias=True)
        self.upconv1_L3 = nn.Conv2d(self.n_feats0, self.n_feats0, 1, 1, 0, bias=True)

        self.upconv1 = nn.Conv2d(self.n_feats0, self.n_feats * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(self.n_feats, self.n_feats * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_last0 = nn.Conv2d(self.n_feats, 1, 3, 1, 1, bias=True)
        self.Spa_freqblock0 = Spa_freqblock(self.n_feats0)
        self.upconv_fuse = nn.Conv2d(self.n_feats0 + self.n_feats0 // 4 + self.n_feats0 // 16, self.n_feats0, 3, 1, 1, bias=True)  

    def forward(self, x):
        batch_size, frames, channels, height, width = x.shape
        shortcut = x
        x = x.reshape(batch_size,-1, height, width)    
        # Feature extraction  
        x1 = self.feat_extract(x)
        sam_features = x1 
        f1 = sam_features[:,0:3*self.n_feats0,:,...]
        f2 = sam_features[:,3*self.n_feats0:4*self.n_feats0,:,...]
        f3 = sam_features[:,4*self.n_feats0:,:,...]
        # Temporal-frequency difference compensation block  TFDC module
        TFD_outs1 = self.TFDC(f1) 
        TFD_outs3 = self.TFDC(f3)
        cat_feat = torch.cat([TFD_outs1,f2,TFD_outs3],dim=1)
        TFD_outs2 = self.TFDC(cat_feat)
        # Spatial-frequency interactive enhancement module
        decoder_outs = self.Spa_freqblock0(TFD_outs2)
        decoder_outs1 = self.rconcat1(decoder_outs)
        decoder_outs2 = self.rconcat2(decoder_outs1)
        decoder_list = [decoder_outs,decoder_outs1,decoder_outs2]
        out_list = self.recorb1(decoder_list)
        out_L3 = self.lrelu(self.upconv1_L3(out_list[2]))
        out_L3_1 = self.pixel_shuffle(out_L3)
        out_L3_2 = self.pixel_shuffle(out_L3_1)
        out_L2 = self.lrelu(self.upconv1_L2(out_list[1]))
        out_L2 = self.pixel_shuffle(out_L2 + self.upconv1_L2_2(torch.cat([out_L2, out_L3_1],1)))
        out_fuse = torch.cat([out_list[0], out_L2, out_L3_2], 1)
        out_fuse = self.recorb0(self.upconv_fuse(out_fuse))

        out = self.lrelu(self.pixel_shuffle(self.upconv1(out_fuse)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.conv_last0(out)
        base = F.interpolate(shortcut[:,frames//2,:,:,:], scale_factor=4, mode='bilinear')
        out = out + base 
        return out




class GShiftNet_S(nn.Module):
    def __init__(self, n_features=64, future_frames=2, past_frames=2):
        super(GShiftNet_S, self).__init__()
        self.n_feats = n_features
        self.num_ff = future_frames
        self.num_fb = past_frames
        self.device = torch.device('cuda')
        self.n_feats0 = 64
        # self.feat_extract = nn.Sequential(nn.Conv2d(7, 7*self.n_feats0, 3, 1, 1),
        #     CAB(7*self.n_feats0, 3, 4, bias=False, act=nn.PReLU()))
        self.feat_extract = nn.Sequential(nn.Conv2d(7, 7*self.n_feats0, 3, 1, 1))
        self.lrelu = nn.PReLU() 
        self.TFDC = TFDC(dim=self.n_feats0)
        # self.orb1 = TFR_UNet(7*self.n_feats0, 7*self.n_feats0, kernel_size=3, reduction=4, act=nn.PReLU(), bias=False, scale_unetfeats=0)
        # self.recorb1 = TFR_UNet(self.n_feats0, self.n_feats0, kernel_size=3, reduction=4, act=nn.PReLU(), bias=False, scale_unetfeats=0)
        self.rconcat1 = nn.Conv2d(self.n_feats0, self.n_feats0, 3, stride=2, padding=1, bias=True)
        self.rconcat2 = nn.Conv2d(self.n_feats0, self.n_feats0, 3, stride=2, padding=1, bias=True)
        self.recorb1 = SCNet(nf=self.n_feats0, SCGroupN=3) 
        self.recorb0 = nn.Conv2d(self.n_feats0, self.n_feats0, 3, 1, 1, bias=True)
        #### upsampling
        #### fea pyramid fuse 
        self.upconv1_L2 = nn.Conv2d(self.n_feats0, self.n_feats0, 1, 1, 0, bias=True)
        self.upconv1_L2_2 = nn.Conv2d(self.n_feats0 + self.n_feats0//4, self.n_feats0, 1, 1, 0, bias=True)
        self.upconv1_L3 = nn.Conv2d(self.n_feats0, self.n_feats0, 1, 1, 0, bias=True)

        self.upconv1 = nn.Conv2d(self.n_feats0, self.n_feats * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(self.n_feats, self.n_feats * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_last0 = nn.Conv2d(self.n_feats, 1, 3, 1, 1, bias=True)
        self.Spa_freqblock0 = Spa_freqblock(self.n_feats0)
        self.upconv_fuse = nn.Conv2d(self.n_feats0 + self.n_feats0 // 4 + self.n_feats0 // 16, self.n_feats0, 3, 1, 1, bias=True)  

    def forward(self, x):
        batch_size, frames, channels, height, width = x.shape
        shortcut = x
        x = x.reshape(batch_size,-1, height, width)    
        # Feature extraction  
        x1 = self.feat_extract(x)
        sam_features = x1 
        f1 = sam_features[:,0:3*self.n_feats0,:,...]
        f2 = sam_features[:,3*self.n_feats0:4*self.n_feats0,:,...]
        f3 = sam_features[:,4*self.n_feats0:,:,...]
        # Temporal-frequency difference compensation block  TFDC module
        TFD_outs1 = self.TFDC(f1) 
        TFD_outs3 = self.TFDC(f3)
        cat_feat = torch.cat([TFD_outs1,f2,TFD_outs3],dim=1)
        TFD_outs2 = self.TFDC(cat_feat)
        # Spatial-frequency interactive enhancement module
        decoder_outs = self.Spa_freqblock0(TFD_outs2)
        decoder_outs1 = self.rconcat1(decoder_outs)
        decoder_outs2 = self.rconcat2(decoder_outs1)
        decoder_list = [decoder_outs,decoder_outs1,decoder_outs2]
        out_list = self.recorb1(decoder_list)
        out_L3 = self.lrelu(self.upconv1_L3(out_list[2]))
        out_L3_1 = self.pixel_shuffle(out_L3)
        out_L3_2 = self.pixel_shuffle(out_L3_1)
        out_L2 = self.lrelu(self.upconv1_L2(out_list[1]))
        out_L2 = self.pixel_shuffle(out_L2 + self.upconv1_L2_2(torch.cat([out_L2, out_L3_1],1)))
        out_fuse = torch.cat([out_list[0], out_L2, out_L3_2], 1)
        out_fuse = self.recorb0(self.upconv_fuse(out_fuse))

        out = self.lrelu(self.pixel_shuffle(self.upconv1(out_fuse)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.conv_last0(out)
        base = F.interpolate(shortcut[:,frames//2,:,:,:], scale_factor=4, mode='bilinear')
        out = out + base 
        return out

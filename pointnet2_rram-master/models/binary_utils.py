import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Hardtanh, BatchNorm1d as BN
from torch.nn.modules.utils import _single
from torch.autograd import Function
from torch.nn import Parameter
import math
# from torch_geometric.data import DataLoader
# from torch_geometric.nn import PointConv, fps, radius, global_max_pool
import numpy as np


activations = {
    'ReLU': ReLU,
    'Hardtanh': Hardtanh
}


def add_noise(weight, noise):
    # one = 1.cuda()
    new_w = weight + weight * noise * torch.randn_like(weight)
    # new_w = nn.
    # return nn.Parameter(new_w, requires_grad=False).to(weight.device)
    return new_w.to(weight.device)


def gen_noise(weight, noise):
    new_w = weight * noise * torch.randn_like(weight)
    return new_w.to(weight.device)


class MeanShift(torch.nn.Module):

    def __init__(self, channels):
        super(MeanShift, self).__init__()
        self.register_buffer('median', torch.zeros((1, channels)))
        self.register_buffer("num_track", torch.LongTensor([0]))

    def forward(self, x):
        if self.training:
            median = torch.sort(x, dim=0)[0][x.shape[0] // 2].view(1, -1)
            self.median.mul_(self.num_track)
            self.median.add_(median)
            self.median.div_(self.num_track + 1)
            self.num_track.add_(1)
            self.median.detach_()
            self.num_track.detach_()
            x = x - self.median
        else:
            x = x - self.median
        return x


class BinaryQuantize(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output
        grad_input[input[0].gt(1)] = 0
        grad_input[input[0].lt(-1)] = 0
        return grad_input


class TernaryQuantize(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # get output
        ctx_max, ctx_min = torch.max(input), torch.min(input)
        lower_interval = ctx_min + (ctx_max - ctx_min) / 3
        higher_interval = ctx_max - (ctx_max - ctx_min) / 3
        out = torch.where(input < lower_interval, torch.tensor(-1.).to(input.device, input.dtype), input)
        out = torch.where(input > higher_interval, torch.tensor(1.).to(input.device, input.dtype), out)
        out = torch.where((input >= lower_interval) & (input <= higher_interval), torch.tensor(0.).to(input.device, input.dtype), out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output
        grad_input[input[0].gt(1)] = 0
        grad_input[input[0].lt(-1)] = 0
        return grad_input


class BinaryQuantizeIdentity(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output
        return grad_input


class BinaryQuantizeIRNet(Function):
    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        grad_input = k * t * (1 - torch.pow(torch.tanh(input * t), 2)) * grad_output
        return grad_input, None, None


class BiLinearLSR(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=False, binary_act=True):
        super(BiLinearLSR, self).__init__(in_features, out_features, bias=bias)
        self.binary_act = binary_act

        # must register a nn.Parameter placeholder for model loading
        # self.register_parameter('scale', None) doesn't register None into state_dict
        # so it leads to unexpected key error when loading saved model
        # hence, init scale with Parameter
        # however, Parameter(None) actually has size [0], not [] as a scalar
        # hence, init it using the following trick
        self.register_parameter('scale', Parameter(torch.Tensor([0.0]).squeeze()))

    def reset_scale(self, input):
        bw = self.weight
        ba = input
        bw = bw - bw.mean()
        self.scale = Parameter((F.linear(ba, bw).std() / F.linear(torch.sign(ba), torch.sign(bw)).std()).float().to(ba.device))
        # corner case when ba is all 0.0
        if torch.isnan(self.scale):
            self.scale = Parameter((bw.std() / torch.sign(bw).std()).float().to(ba.device))

    def forward(self, input):
        bw = self.weight
        ba = input
        bw = bw - bw.mean()

        if self.scale.item() == 0.0:
            self.reset_scale(input)

        bw = BinaryQuantize().apply(bw)
        bw = bw * self.scale
        if self.binary_act:
            ba = BinaryQuantize().apply(ba)
        output = F.linear(ba, bw)
        return output


class BiLinearXNOR(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, binary_act=True):
        super(BiLinearXNOR, self).__init__(in_features, out_features, bias=bias)
        self.binary_act = binary_act

    def forward(self, input):
        bw = self.weight
        ba = input
        bw = bw - bw.mean(-1).view(-1, 1)
        sw = bw.abs().mean(-1).view(-1, 1).detach()
        bw = BinaryQuantize().apply(bw)
        bw = bw * sw
        if self.binary_act:
            sa = ba.abs().mean(-1).view(-1, 1).detach()
            ba = BinaryQuantize().apply(ba)
            ba = ba * sa
        output = F.linear(ba, bw, self.bias)
        return output


class BiLinearBiReal(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, binary_act=True):
        super(BiLinearBiReal, self).__init__(in_features, out_features, bias=bias)
        self.binary_act = binary_act

    def forward(self, input):
        x = input
        out_forward = torch.sign(input)
        out_e_total = 0
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        input = out_forward.detach() - out3.detach() + out3
        real_weights = self.weight
        scaling_factor = torch.mean(abs(real_weights),dim=1,keepdim=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        # y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)
        output = F.linear(input, binary_weights)
        return output


class BiLinearIRNet(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, binary_act=True):
        super(BiLinearIRNet, self).__init__(in_features, out_features, bias=bias)
        self.k = Parameter(torch.tensor([10], requires_grad=False).float().cuda())
        self.t = Parameter(torch.tensor([0.1], requires_grad=False).float().cuda())
        self.binary_act = binary_act

    def forward(self, input):
        bw = self.weight
        ba = input
        bw = bw - bw.mean(-1).view(-1, 1)
        bw = bw / bw.std(-1).view(-1, 1)
        sw = torch.pow(torch.tensor([2] * bw.size(0)).cuda().float(), (torch.log(bw.abs().mean(-1)) / math.log(2)).round().float()).view(-1, 1).detach()
        k, t = self.k, self.t
        bw = BinaryQuantizeIRNet().apply(bw, k, t)
        bw = bw * sw
        if self.binary_act:
            ba = BinaryQuantizeIRNet().apply(ba, k, t)
        output = F.linear(ba, bw, self.bias)
        return output


def sparse_matrix_C(nrows, ncols, sparsity, distribution):
    # Create sparse matrix

    # Type of distribution
    if distribution == 'uniform':
        W = 2 * torch.rand(nrows, ncols) - 1
    elif distribution == 'normal':
        W = torch.randn(nrows, ncols)
    elif distribution == 'experimental':
        h_mean = 2117
        h_sigma = 901
        l_mean = 896
        l_sigma = 561

        # Ideal tenary weight matrix
        W_ternary = np.random.rand(nrows, ncols)
        W_ternary = -1 * (W_ternary < 0.5) + 1 * (W_ternary > 0.5)

        # Simulate positive and negative conductance matrix of real RRAM
        # W_ternary == -1: G_pos == 0, G_neg == 1
        # W_ternary == 0: G_pos == 0, G_neg == 0
        # W_ternary == +1: G_pos == 1, G_neg == 0
        G_pos = (W_ternary == 1) * np.random.normal(loc=h_mean, scale=h_sigma, size=(nrows, ncols)) + (
                W_ternary == -1) * np.random.normal(loc=l_mean, scale=l_sigma, size=(nrows, ncols))
        G_neg = (W_ternary == -1) * np.random.normal(loc=h_mean, scale=h_sigma, size=(nrows, ncols)) + (
                W_ternary == 1) * np.random.normal(loc=l_mean, scale=l_sigma, size=(nrows, ncols))

        # Final weight
        W = torch.Tensor((G_pos - G_neg) / (h_mean - l_mean))
    else:
        raise ValueError('distribution not supported')

    # Sparsity
    if sparsity > 0:
        connectivity_mask = torch.rand(nrows, ncols) > sparsity
        W = W * connectivity_mask

    return W


class TriLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, binary_act=True):
        super(TriLinear, self).__init__(in_features, out_features, bias=bias)
        self.output_ = None

    def forward(self, input):
        tw = self.weight
        ta = input
        tw = TernaryQuantize().apply(tw)
        output = F.linear(ta, tw, self.bias)
        self.output_ = output
        return output


class BiLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, binary_act=False):
        super(BiLinear, self).__init__(in_features, out_features, bias=True)
        self.binary_act = binary_act
        self.output_ = None

    def forward(self, input):
        bw = self.weight
        ba = input
        bw = BinaryQuantize().apply(bw)
        if self.binary_act:
            ba = BinaryQuantize().apply(ba)
        output = F.linear(ba, bw, self.bias)
        self.output_ = output
        return output


class NoiseBiLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, binary_act=False, noise=0):
        super(NoiseBiLinear, self).__init__(in_features, out_features, bias=True)
        self.binary_act = binary_act
        self.output_ = None

        self.out_features
        self.noise = noise

    def forward(self, input):
        bw = self.weight
        ba = input
        bw = BinaryQuantize().apply(bw)
        if self.binary_act:
            ba = BinaryQuantize().apply(ba)
        if not self.noise:
            output = F.linear(ba, bw, self.bias)
        else:
            output = F.linear(ba, bw, self.bias) + self.noised_forward(input, bw)
        self.output_ = output
        return output

    def noised_forward(self, x, w):
        batch_size, n_points = x.size 
        origin_weight = w
        x_new = torch.zeros(batch_size, self.out_features).to(x.device)
        
        for i in range(x.shape[0]):
            noise_weight = gen_noise(origin_weight, self.noise).detach()   # detach from computational graph
            x_i = F.linear(x[i,:], noise_weight, self.bias)
            x_new[i, :] = x_i.squeeze()
            del noise_weight, x_i
        
        return x_new


biLinears = {
    'BiLinear': BiLinear,
    'BiLinearXNOR': BiLinearXNOR,
    'BiLinearABC': BiLinearXNOR,
    'BiLinearIRNet': BiLinearIRNet,
    'BiLinearLSR': BiLinearLSR
}


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])


def BiMLP(channels, batch_norm=True, activation=ReLU, BiLinear=BiLinear):
    return Seq(*[
        Seq(BiLinear(channels[i - 1], channels[i]), activation(), BN(channels[i]))
        for i in range(1, len(channels))
    ])


def BwMLP(channels, batch_norm=True, activation='ReLU', bilinear='BiLinear'):
    return Seq(*[
        Seq(biLinears[bilinear](channels[i - 1], channels[i], binary_act=False), activations[activation](), BN(channels[i]))
        for i in range(1, len(channels))
    ])


def FirstBiMLP(channels, batch_norm=True, activation='ReLU', bilinear='BiLinear'):
    part1 = [Seq(Lin(channels[0], channels[1]), activations[activation](), BN(channels[1]))]
    part2 = [
        Seq(biLinears[bilinear](channels[i - 1], channels[i]), activations[activation](), BN(channels[i]))
        for i in range(2, len(channels))
    ]
    obj = part1 + part2
    return Seq(*obj)


class BiConv1d(torch.nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(BiConv1d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)

    def forward(self, input):

        bw = self.weight
        ba = input
        bw = bw - bw.mean()
        bw = BinaryQuantize().apply(bw)
        # ba = BinaryQuantize().apply(ba)

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv1d(F.pad(ba, expanded_padding, mode='circular'),
                            bw, self.bias, self.stride,
                            _single(0), self.dilation, self.groups)
        return F.conv1d(ba, bw, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class BiConv1dXNOR(torch.nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(BiConv1dXNOR, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)

    def forward(self, input):

        bw = self.weight
        # ba = input
        bw = bw - bw.mean()
        sw = bw.abs().view(bw.size(0), bw.size(1), -1).mean(-1).view(bw.size(0), bw.size(1), 1).detach()
        # sa = ba.abs().view(ba.size(0), ba.size(1), -1).mean(-1).view(ba.size(0), ba.size(1), 1).detach()
        bw = BinaryQuantize().apply(bw)
        # ba = BinaryQuantize().apply(ba)
        bw = bw * sw
        # ba = ba * sa

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv1d(F.pad(ba, expanded_padding, mode='circular'),
                            bw, self.bias, self.stride,
                            _single(0), self.dilation, self.groups)
        return F.conv1d(ba, bw, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class BiConv2dLSR(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=False, padding_mode='zeros'):
        super(BiConv2dLSR, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)
        self.register_parameter('scale', None)

    def reset_scale(self, input):
        bw = self.weight
        ba = input
        bw = bw - bw.mean()
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[0] + 1) // 2, self.padding[0] // 2)
            self.scale = Parameter((F.conv1d(F.pad(ba, expanded_padding, mode='circular'),\
                                  bw, self.bias, self.stride,\
                                  _single(0), self.dilation, self.groups).std() / \
                F.conv2d(torch.sign(F.pad(ba, expanded_padding, mode='circular')),\
                         torch.sign(bw), self.bias, self.stride,\
                         _single(0), self.dilation, self.groups).std()).float().to(ba.device))
        else:
            self.scale = Parameter((F.conv2d(ba, bw, self.bias, self.stride, self.padding, self.dilation, self.groups).std() \
                                   / F.conv2d(torch.sign(ba), torch.sign(bw), self.bias, self.stride, self.padding, self.dilation, self.groups).std()).float().to(ba.device))

    def forward(self, input):
        bw = self.weight
        ba = input
        bw = bw - bw.mean()
        if self.scale is None:
            self.reset_scale(input)
        bw = BinaryQuantize().apply(bw)
        # ba = BinaryQuantize().apply(ba)

        bw = bw * self.scale

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(ba, expanded_padding, mode='circular'),
                            bw, self.bias, self.stride,
                            _single(0), self.dilation, self.groups)
        return F.conv2d(ba, bw, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class NoiseBiConv2dLSR(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', noise=0):
        super(NoiseBiConv2dLSR, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)
        self.register_parameter('scale', None)
        self.noise = noise
        # self.conv = nn.Conv2d(in)
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.sample_noise = sample_noise

    def reset_scale(self, input):
        bw = self.weight
        ba = input
        bw = bw - bw.mean()
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[0] + 1) // 2, self.padding[0] // 2)
            self.scale = Parameter((F.conv1d(F.pad(ba, expanded_padding, mode='circular'),\
                                  bw, self.bias, self.stride,\
                                  _single(0), self.dilation, self.groups).std() / \
                F.conv2d(torch.sign(F.pad(ba, expanded_padding, mode='circular')),\
                         torch.sign(bw), self.bias, self.stride,\
                         _single(0), self.dilation, self.groups).std()).float().to(ba.device))
        else:
            self.scale = Parameter((F.conv2d(ba, bw, self.bias, self.stride, self.padding, self.dilation, self.groups).std() \
                                   / F.conv2d(torch.sign(ba), torch.sign(bw), self.bias, self.stride, self.padding, self.dilation, self.groups).std()).float().to(ba.device))


    def forward(self, input):
        bw = self.weight
        ba = input
        bw = bw - bw.mean()
        if self.scale is None:
            self.reset_scale(input)
        bw = BinaryQuantize().apply(bw)
        # ba = BinaryQuantize().apply(ba)

        bw = bw * self.scale

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[0] + 1) // 2, self.padding[0] // 2)
            output = F.conv2d(F.pad(ba, expanded_padding, mode='circular'),
                            bw, self.bias, self.stride,
                            _single(0), self.dilation, self.groups)
        else:
            output = F.conv2d(ba, bw, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

        if self.noise:
            output = output + self.noised_forward(input, bw)

        return output

    def noised_forward(self, x, weight):
        x = x.detach()

        batch_size, in_features, nsamples, npoints = x.size()
        x = x.reshape(-1, in_features, 1, 1)

        origin_weight = weight
        x_new = torch.zeros(x.shape[0], self.out_channels, 1, 1)

        for i in range(x.shape[0]):
            noise_weight = gen_noise(weight, self.noise).detach()
            noise_weight = noise_weight.squeeze()
            # x_i =  x[i, :, :, :].squeeze(-1)
            x_i =  x[i, :, :, :].unsqueeze(0)
            # x_i = torch.matmul(noise_weight, x_i)
            x_i = F.conv2d(x_i, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            x_new[i, :, :, :] = x_i.squeeze(0)
            del noise_weight, x_i
        x_new = x_new.reshape(batch_size, self.out_channels, nsamples, npoints)
        return x_new.to(x.device).detach()


class TriConv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(TriConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)

    def forward(self, input):

        bw = self.weight
        ba = input
        bw = bw - bw.mean()
        bw = TernaryQuantize().apply(bw)
        # ba = TernaryQuantize().apply(ba)

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(ba, expanded_padding, mode='circular'),
                            bw, self.bias, self.stride,
                            _single(0), self.dilation, self.groups)
        return F.conv2d(ba, bw, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

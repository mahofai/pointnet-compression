
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


from models.noise_layers import NoiseModule, NoiseConv, NoiseLinear
from models.binary_utils import MeanShift, BiLinear, BiLinearXNOR, BiLinearIRNet, BiLinearLSR, BiLinearBiReal


biLinears = {
    'BiLinear': BiLinear,
    'BiLinearXNOR': BiLinearXNOR,
    'BiLinearIRNet': BiLinearIRNet,
    'BiLinearLSR': BiLinearLSR
}


seg_classes = {
    'Airplane': [0, 1, 2, 3],
    'Bag': [4, 5],
    'Cap': [6, 7],
    'Car': [8, 9, 10, 11],
    'Chair': [12, 13, 14, 15],
    'Earphone': [16, 17, 18],
    'Guitar': [19, 20, 21],
    'Knife': [22, 23],
    'Lamp': [24, 25, 26, 27],
    'Laptop': [28, 29],
    'Motorbike': [30, 31, 32, 33, 34, 35],
    'Mug': [36, 37],
    'Pistol': [38, 39, 40],
    'Rocket': [41, 42, 43],
    'Skateboard': [44, 45, 46],
    'Table': [47, 48, 49],
}


offset_map = {
    1024: -3.2041,
    2048: -3.4025,
    4096: -3.5836
}


class Conv1d(nn.Module):
    def __init__(self, inplane, outplane, Linear):
        super().__init__()
        self.lin = Linear(inplane, outplane)

    def forward(self, x):
        B, C, N = x.shape
        x = x.permute(0, 2, 1).contiguous().view(-1, C)
        x = self.lin(x).view(B, N, -1).permute(0, 2, 1).contiguous()
        return x


class BiSTN3d(nn.Module):
    def __init__(self, channel, Linear=BiLinear, pool='max', affine=True, bi_first=False):
        super(BiSTN3d, self).__init__()
        if bi_first:
            self.conv1 = Conv1d(channel, 64, Linear)
        else:
            self.conv1 = Conv1d(channel, 64, nn.Linear)
        self.conv2 = Conv1d(64, 128, Linear)
        self.conv3 = Conv1d(128, 1024, Linear)
        self.fc1 = Linear(1024, 512)
        self.fc2 = Linear(512, 256)
        self.fc3 = Linear(256, 9)

        self.bn1 = nn.BatchNorm1d(64, affine=affine)
        self.bn2 = nn.BatchNorm1d(128, affine=affine)
        self.bn3 = nn.BatchNorm1d(1024, affine=affine)
        self.bn4 = nn.BatchNorm1d(512, affine=affine)
        self.bn5 = nn.BatchNorm1d(256, affine=affine)
        self.pool = pool

    def forward(self, x):

        batchsize, D, N = x.size()
        x = F.hardtanh(self.bn1(self.conv1(x)))
        x = F.hardtanh(self.bn2(self.conv2(x)))

        if self.pool == 'max':
            x = F.hardtanh(self.bn3(self.conv3(x)))
            x = torch.max(x, 2, keepdim=True)[0]
        elif self.pool == 'mean':
            x = F.hardtanh(self.bn3(self.conv3(x)))
            x = torch.mean(x, 2, keepdim=True)
        elif self.pool == 'ema-max':
            x = self.bn3(self.conv3(x)) + offset_map[N]
            x = torch.max(x, 2, keepdim=True)[0]
            x = x.view(-1, 1024)
        x = x.view(-1, 1024)

        x = F.hardtanh(self.bn4(self.fc1(x)))
        x = F.hardtanh(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)

        return x


class BiSTNkd(nn.Module):
    def __init__(self, k=64, Linear=BiLinear, pool='max', affine=True, bi_first=False):
        super(BiSTNkd, self).__init__()
        self.conv1 = Conv1d(k, 64, Linear)
        self.conv2 = Conv1d(64, 128, Linear)
        self.conv3 = Conv1d(128, 1024, Linear)
        self.fc1 = Linear(1024, 512)
        self.fc2 = Linear(512, 256)
        self.fc3 = Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64, affine=affine)
        self.bn2 = nn.BatchNorm1d(128, affine=affine)
        self.bn3 = nn.BatchNorm1d(1024, affine=affine)
        self.bn4 = nn.BatchNorm1d(512, affine=affine)
        self.bn5 = nn.BatchNorm1d(256, affine=affine)
        self.k = k
        self.pool = pool

    def forward(self, x):
        batchsize, D, N = x.size()
        x = F.hardtanh(self.bn1(self.conv1(x)))
        x = F.hardtanh(self.bn2(self.conv2(x)))
        if self.pool == 'max':
            x = F.hardtanh(self.bn3(self.conv3(x)))
            x = torch.max(x, 2, keepdim=True)[0]
        elif self.pool == 'mean':
            x = F.hardtanh(self.bn3(self.conv3(x)))
            x = torch.mean(x, 2, keepdim=True)
        elif self.pool == 'ema-max':
            x = self.bn3(self.conv3(x)) + offset_map[N]
            x = torch.max(x, 2, keepdim=True)[0]
            x = x.view(-1, 1024)
        x = x.view(-1, 1024)

        x = F.hardtanh(self.bn4(self.fc1(x)))
        x = F.hardtanh(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class BiPointNetEncoder(nn.Module):
    def __init__(self, Linear, global_feat=True, feature_transform=False, channel=3, pool='max', affine=True, tnet=True, bi_first=False, use_bn=True):
        super(BiPointNetEncoder, self).__init__()
        self.tnet = tnet
        if self.tnet:
            self.stn = BiSTN3d(channel, Linear, pool=pool, affine=affine, bi_first=bi_first)
        if bi_first:
            self.conv1 = Conv1d(channel, 64, Linear)
        else:
            self.conv1 = Conv1d(channel, 64, nn.Linear)
        self.conv2 = Conv1d(64, 128, Linear)
        self.conv3 = Conv1d(128, 1024, Linear)
        self.bn1 = nn.BatchNorm1d(64, affine=affine)
        self.bn2 = nn.BatchNorm1d(128, affine=affine)
        self.bn3 = nn.BatchNorm1d(1024, affine=affine)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.tnet and self.feature_transform:
            self.fstn = BiSTNkd(k=64, Linear=Linear, pool=pool, affine=affine, bi_first=bi_first)
        self.pool = pool
        self.use_bn = use_bn

    def forward(self, x):
        B, D, N = x.size()
        if self.tnet:
            trans = self.stn(x)
        else:
            trans = None

        x = x.transpose(2, 1)
        if D == 6:
            x, feature = x.split(3, dim=2)
        elif D == 9:
            x, feature = x.split([3, 6], dim=2)
        if self.tnet:
            x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        if self.use_bn:
            x = F.hardtanh(self.bn1(self.conv1(x)))
        else:
            x = F.hardtanh(self.conv1(x))

        if self.tnet and self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        if self.use_bn:
            x = F.hardtanh(self.bn2(self.conv2(x)))
            x = self.bn3(self.conv3(x))
        else:
            x = F.hardtanh(self.conv2(x))
            x = self.conv3(x)

        if self.pool == 'max':
            x = torch.max(x, 2, keepdim=True)[0]
        elif self.pool == 'mean':
            x = torch.mean(x, 2, keepdim=True)
        elif self.pool == 'ema-max':
            if self.use_bn:
                x = torch.max(x, 2, keepdim=True)[0] + offset_map[N]
            else:
                x = torch.max(x, 2, keepdim=True)[0] - 0.3
            x = x.view(-1, 1024)
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss

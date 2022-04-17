import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation
from noise_layers import NoiseModule, NoiseConv, NoiseLinear,NoiseConv1d
from binary_utils import NoiseBiLinearLSR, BiLinearLSR, NoiseTriLinear, NoiseBiLinear, BiMLP, BiLinear, NoiseTriConv1d,NoiseBiConv1dLSR
from kmeans import kmeans_clustering, KmeansLinear, KmeansConv1d

class Conv1dLSR(nn.Module):
    def __init__(self, inplane, outplane):
        super().__init__()
        self.lin = NoiseBiLinearLSR(inplane, outplane)

    def forward(self, x):
        B, C, N = x.shape
        x = x.permute(0, 2, 1).contiguous().view(-1, C)
        x = self.lin(x).view(B, N, -1).permute(0, 2, 1).contiguous()
        return x

class get_model(nn.Module):
    def __init__(self, num_classes, c_prune_rate=1, noise=0, compression='full',pool='ema-max',bits=1,drop=0):
        super(get_model, self).__init__()
        #mlp1 = [32, 32, 64]
        #mlp2 = [64, 64, 128]
        #mlp3 = [128, 128, 256]
        #mlp4 = [256, 256, 512]
        mlp1 = [64, 64, 64]
        mlp2 = [128, 128, 128, 128]
        mlp3 = [256, 256, 256, 256]
        mlp4 = [512]
        if c_prune_rate != 1:
            mlp1 = [int(c / c_prune_rate) for c in mlp1]
            mlp2 = [int(c / c_prune_rate) for c in mlp2]
            mlp3 = [int(c / c_prune_rate) for c in mlp3]
            mlp4 = [int(c / c_prune_rate) for c in mlp4] 

        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 9 + 3, mlp1, False, noise=noise, compression=compression,drop=drop)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, int(64/c_prune_rate) + 3, mlp2, False, noise=noise, compression=compression,bits=bits,drop=drop)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, int(128/c_prune_rate) + 3, mlp3, False, noise=noise, compression=compression,bits=bits,drop=drop)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, int(256/c_prune_rate) + 3, mlp4, False, noise=noise, compression=compression,bits=bits,drop=drop)
        self.fp4 = PointNetFeaturePropagation( int(768 / c_prune_rate), [int(256 / c_prune_rate), int(256 / c_prune_rate)],compression=compression,bits=bits,drop=drop)
        self.fp3 = PointNetFeaturePropagation( int(384 / c_prune_rate), [int(256 / c_prune_rate), int(256 / c_prune_rate)],compression=compression,bits=bits,drop=drop)
        self.fp2 = PointNetFeaturePropagation( int(320 / c_prune_rate), [int(256 / c_prune_rate), int(128 / c_prune_rate)],compression=compression,bits=bits,drop=drop)
        self.fp1 = PointNetFeaturePropagation( int(128 / c_prune_rate), [int(128 / c_prune_rate), int(128 / c_prune_rate), int(128 / c_prune_rate)],compression=compression,bits=bits,drop=drop)


        if compression == 'full':
            self.conv1 = NoiseConv1d(int(128 / c_prune_rate), int(128 / c_prune_rate), 1,noise=noise,)
            self.conv2 = NoiseConv1d(int(128 / c_prune_rate), num_classes, 1,noise=noise,)

        elif compression == 'binary':
            print("BiConv1d")
            self.conv1 = Conv1dLSR(int(128 / c_prune_rate), int(128 / c_prune_rate))
            self.conv2 = Conv1dLSR(int(128 / c_prune_rate), num_classes)

        elif compression == 'ternary':
            print("tenConv1d")
            self.conv1 =  NoiseTriConv1d(int(128 / c_prune_rate), int(128 / c_prune_rate), 1, noise=noise,)
            self.conv2 =  NoiseTriConv1d(int(128 / c_prune_rate), num_classes, 1,noise=noise,)

        elif compression == 'kmeans':
            print("kmeansConv1d")
            self.conv1 =  KmeansConv1d(int(128 / c_prune_rate), int(128 / c_prune_rate), 1,bits=bits,)
            self.conv2 =  KmeansConv1d(int(128 / c_prune_rate), num_classes, 1,bits=bits,)

        #self.conv1 = nn.Conv1d(int(128 / c_prune_rate), int(128 / c_prune_rate), 1)
        self.bn1 = nn.BatchNorm1d(int(128 / c_prune_rate))
        self.drop1 = nn.Dropout(0.5)
        #self.conv2 = nn.Conv1d(int(128 / c_prune_rate), num_classes, 1)


        self.noise = noise
        self.c_prune_rate = c_prune_rate
        self.dropAll=  nn.Dropout(drop)
        

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.dropAll(self.drop1(F.relu(self.bn1(self.conv1(l0_points)))))
        x = self.dropAll(self.conv2(x))
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l4_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
    def forward(self, pred, target, trans_feat, weight):
        total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss

if __name__ == '__main__':
    import  torch
    model = get_model(13)
    xyz = torch.rand(6, 9, 2048)
    (model(xyz))

import torch.nn as nn
import torch
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation
from binary_utils import NoiseBiLinearLSR,BiLinearLSR, NoiseTriLinear, NoiseBiLinear, BiMLP, BiLinear, NoiseTriConv1d,BiConv1dLSR,NoiseBiConv1dLSR,NoiseTriConv1dLSR,NoiseBiConv1d
from noise_layers import NoiseModule, NoiseConv1d, NoiseLinear
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

class Conv1d(nn.Module):
    def __init__(self, inplane, outplane):
        super().__init__()
        self.lin = NoiseBiLinear(inplane, outplane)

    def forward(self, x):
        B, C, N = x.shape
        x = x.permute(0, 2, 1).contiguous().view(-1, C)
        x = self.lin(x).view(B, N, -1).permute(0, 2, 1).contiguous()
        return x

class get_model(nn.Module):
    def __init__(self, num_classes, normal_channel=False,c_prune_rate=1, noise=0,compression='full',pool='ema_max',bits=1):
        super(get_model, self).__init__()
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel

        #mlp1 = [64, 64, 128]
        #mlp2 = [128, 128, 256]
        #mlp3 = [256, 512, 1024]
        mlp1 = [128, 128, 128,128]
        mlp2 = [256, 256, 256, 256, 256]
        mlp3 = [1024]

        mlp4 = [256, 256]
        mlp5 = [256, 128]
        mlp6 = [128, 128, 128]

        if c_prune_rate != 1:
            mlp1 = [int(c / c_prune_rate) for c in mlp1]
            mlp2 = [int(c / c_prune_rate) for c in mlp2]
            mlp3 = [int(c / c_prune_rate) for c in mlp3]
            mlp4 = [int(c / c_prune_rate) for c in mlp4]
            mlp5 = [int(c / c_prune_rate) for c in mlp5]
            mlp6 = [int(c / c_prune_rate) for c in mlp6]
        '''
        for i in range(len(mlp1)):
          if mlp1[i]%2>0:
            mlp1[i] = mlp1[i]-1
        for i in range(len(mlp2)):
          if mlp2[i]%2>0:
            mlp2[i] = mlp2[i]-1
        for i in range(len(mlp3)):
          if mlp3[i]%2>0:
            mlp3[i] = mlp3[i]+1
        for i in range(len(mlp4)):
          if mlp4[i]%2>0:
            mlp4[i] = mlp4[i]-1
        for i in range(len(mlp5)):
          if mlp5[i]%2>0:
            mlp5[i] = mlp5[i]-1
        '''
        

        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=6+additional_channel, mlp=mlp1, group_all=False, compression=compression,noise=noise,bits=bits)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel= int(128 / c_prune_rate) + 3, mlp=mlp2, group_all=False, compression=compression,noise=noise,bits=bits)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=int(256 / c_prune_rate) + 3, mlp=mlp3, group_all=True, compression=compression,noise=noise,bits=bits)
        #self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256], compression=compression)
        #self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128], compression=compression)
        #self.fp1 = PointNetFeaturePropagation(in_channel=128+16+6+additional_channel, mlp=[128, 128, 128], compression=compression)
        self.fp3 = PointNetFeaturePropagation(in_channel= int(1280 / c_prune_rate), mlp=mlp4, compression=compression, noise=noise,bits=bits)
        self.fp2 = PointNetFeaturePropagation(in_channel= int(384 / c_prune_rate), mlp=mlp5, compression=compression, noise=noise,bits=bits)
        self.fp1 = PointNetFeaturePropagation(in_channel=int(128 / c_prune_rate)+16+6+additional_channel, mlp=mlp6, compression=compression,noise=noise,bits=bits)
 
        
        if compression == 'full':
            #self.conv1 = nn.Conv1d(128, 128, 1)
            #self.conv2 = nn.Conv1d(128, num_classes, 1)
            self.conv1 = NoiseConv1d(int(128 / c_prune_rate), int(128 / c_prune_rate), 1, noise=noise,)
            self.conv2 = NoiseConv1d(int(128 / c_prune_rate), num_classes, 1, noise=noise,)
        if compression == 'binary':
            #self.conv1 = BiConv1dLSR(128, 128, 1)
            #self.conv2 = BiConv1dLSR(128, num_classes, 1)
            self.conv1 = Conv1d(int(128 / c_prune_rate), int(128 / c_prune_rate))
            self.conv2 = Conv1d(int(128 / c_prune_rate), num_classes)
        elif compression == 'ternary':
            #self.conv1 = TriConv1d(128, 128, 1)
            #self.conv2 = TriConv1d(128, num_classes, 1)          
            self.conv1 = NoiseTriConv1d(int(128 / c_prune_rate), int(128 / c_prune_rate), 1,noise=noise)
            self.conv2 = NoiseTriConv1d(int(128 / c_prune_rate), num_classes, 1,noise=noise)
        elif compression == 'ternaryLSR':
            #self.conv1 = TriConv1d(128, 128, 1)
            #self.conv2 = TriConv1d(128, num_classes, 1)          
            self.conv1 = NoiseTriConv1d(int(128 / c_prune_rate), int(128 / c_prune_rate), 1,noise=noise)
            self.conv2 = NoiseTriConv1d(int(128 / c_prune_rate), num_classes, 1,noise=noise)
        elif compression == 'kmeans':
            #self.conv1 = TriConv1d(128, 128, 1)
            #self.conv2 = TriConv1d(128, num_classes, 1)          
            self.conv1 = KmeansConv1d(int(128 / c_prune_rate), int(128 / c_prune_rate), 1,bits=bits)
            self.conv2 = KmeansConv1d(int(128 / c_prune_rate), num_classes, 1,bits=bits)
        #self.bn1 = nn.BatchNorm1d(128)
        self.bn1 = nn.BatchNorm1d(int(128 / c_prune_rate))
        self.drop1 = nn.Dropout(0.5)
        


    def forward(self, xyz, cls_label):
        # Set Abstraction layers
        B,C,N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        cls_label_one_hot = cls_label.view(B,16,1).repeat(1,1,N)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([cls_label_one_hot,l0_xyz,l0_points],1), l1_points)
        # FC layers
        feat =  F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
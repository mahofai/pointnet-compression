import torch.nn as nn
import torch
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation
from noise_layers import NoiseModule, NoiseConv, NoiseLinear


class get_model(nn.Module):
    def __init__(self, num_classes, normal_channel=False, c_prune_rate=1, noise=0):
        super(get_model, self).__init__()
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel

        mlp1=[64, 64, 128]
        mlp2=[128, 128, 256]
        mlp3=[256, 512, 1024]

        if c_prune_rate != 1:
            mlp1 = [int(c / c_prune_rate) for c in mlp1]
            mlp2 = [int(c / c_prune_rate) for c in mlp2]
            mlp3 = [int(c / c_prune_rate) for c in mlp3]
            # mlp4 = [int(c / c_prune_rate) for c in mlp4]
        #self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=6+additional_channel, mlp=[64, 64, 128], group_all=False)
        #self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        #self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=6+additional_channel, mlp = mlp1, group_all=False, noise=noise)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=int(128/c_prune_rate) + 3, mlp = mlp2, group_all=False, noise=noise)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=int(256/c_prune_rate) + 3, mlp = mlp3, group_all=True, noise=noise)

        #self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        #self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        #self.fp1 = PointNetFeaturePropagation(in_channel=128+16+6+additional_channel, mlp=[128, 128, 128])
        self.fp3 = PointNetFeaturePropagation(in_channel=int(1280/c_prune_rate), mlp=[int(256/c_prune_rate), int(256/c_prune_rate)])
        self.fp2 = PointNetFeaturePropagation(in_channel=int(384/c_prune_rate), mlp=[int(256/c_prune_rate), int(128/c_prune_rate)])
        self.fp1 = PointNetFeaturePropagation(in_channel=int(128/c_prune_rate)+16+6+additional_channel, mlp=[int(128/c_prune_rate), int(128/c_prune_rate), int(128/c_prune_rate)])
        self.conv1 = nn.Conv1d(int(128/c_prune_rate), int(128/c_prune_rate), 1)
        self.bn1 = nn.BatchNorm1d(int(128/c_prune_rate))
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(int(128/c_prune_rate), num_classes, 1)

        self.noise = noise
        self.c_prune_rate = c_prune_rate

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
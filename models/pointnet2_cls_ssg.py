import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction
from noise_layers import NoiseModule, NoiseConv, NoiseLinear


class get_model(NoiseModule):
    def __init__(self, num_class, normal_channel=True, c_prune_rate=1, noise=0, compression='full'):
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        # mlp1 = [64, 64, 128]
        # mlp2 = [128, 128, 256]
        # mlp3 = [256, 512, 1024]

        # try recurrent
        mlp1 = [128, 128, 128, 128]
        mlp2 = [256, 256, 256, 256, 256]
        mlp3 = [1024]
        # mlp4 = [1024]
        # mlp3 = [512, 512, 512]
        # mlp4 = [1024, 1024, 1024, 1024, 1024, 1024]
        
        if c_prune_rate != 1:
            mlp1 = [int(c / c_prune_rate) for c in mlp1]
            mlp2 = [int(c / c_prune_rate) for c in mlp2]
            mlp3 = [int(c / c_prune_rate) for c in mlp3]
            # mlp4 = [int(c / c_prune_rate) for c in mlp4]
        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=mlp1, group_all=False, noise=noise)
        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64, in_channel=int(128/c_prune_rate) + 3, mlp=mlp2, group_all=False, noise=noise)
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None, in_channel=int(256/c_prune_rate) + 3, mlp=mlp3, group_all=True, noise=noise)
        
        # try recurrent
        # self.sa4 = PointNetSetAbstraction(
        #     npoint=None, radius=None, nsample=None, in_channel=int(512/c_prune_rate) + 3, mlp=mlp4, group_all=True, noise=noise)

        # self.fc1 = nn.Linear(int(1024 / c_prune_rate), int(512 / c_prune_rate))
        self.fc1 = NoiseLinear(int(1024 / c_prune_rate), int(512 / c_prune_rate), noise=noise)
        self.bn1 = nn.BatchNorm1d(int(512 / c_prune_rate))
        self.drop1 = nn.Dropout(0.4)
        # self.fc2 = nn.Linear(int(512 / c_prune_rate), int(256 / c_prune_rate))
        self.fc2 = NoiseLinear(int(512 / c_prune_rate), int(256 / c_prune_rate), noise=noise)
        self.bn2 = nn.BatchNorm1d(int(256 / c_prune_rate))
        self.drop2 = nn.Dropout(0.4)
        # self.fc3 = nn.Linear(int(256 / c_prune_rate), num_class)
        self.fc3 = NoiseLinear(int(256 / c_prune_rate), num_class, noise=noise)

        self.noise = noise
        self.c_prune_rate = c_prune_rate

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None

        # if self.noise:
        #     fc1_weight_ori = self.fc1.weight
        #     fc2_weight_ori = self.fc2.weight
        #     fc3_weight_ori = self.fc3.weight
        #     self.fc1.weight = self.add_noise(self.fc1.weight.data, self.noise)
        #     self.fc2.weight = self.add_noise(self.fc2.weight.data, self.noise)
        #     self.fc3.weight = self.add_noise(self.fc3.weight.data, self.noise)
        # with torch.no_grad():
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # try recurrent
        # l3_xyz, l3_points = self.sa4(l3_xyz, l3_points)

        x = l3_points.view(B, int(1024 / self.c_prune_rate))
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)
        return x, l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss

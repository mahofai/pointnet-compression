import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer
from noise_layers import NoiseModule, NoiseConv, NoiseLinear

class get_model(NoiseModule):
    def __init__(self, k=40, normal_channel=True,c_prune_rate=1, noise=0, compression='full'):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3

        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel, c_prune_rate=c_prune_rate,)
        #self.fc1 = nn.Linear(1024, 512)
        #self.fc2 = nn.Linear(512, 256)
        #self.fc3 = nn.Linear(256, k)
        
        self.fc1 = NoiseLinear(int(1024/c_prune_rate), int(512/c_prune_rate), noise=noise)
        self.fc2 = NoiseLinear(int(512/c_prune_rate), int(256/c_prune_rate), noise=noise)
        self.fc3 = NoiseLinear(int(256/c_prune_rate), k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(int(512/c_prune_rate))
        self.bn2 = nn.BatchNorm1d(int(256/c_prune_rate))
        self.relu = nn.ReLU()

        self.noise = noise
        self.c_prune_rate = c_prune_rate

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x, trans_feat

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss

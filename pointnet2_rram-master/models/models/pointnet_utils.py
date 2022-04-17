import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from noise_layers import NoiseModule, NoiseConv, NoiseLinear

offset_map = {
    1024: -3.2041,
    2048: -3.4025,
    4096: -3.5836
}


class STN3d(nn.Module):
    def __init__(self, channel, c_prune_rate=1):
        super(STN3d, self).__init__()
        #self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        #self.conv2 = torch.nn.Conv1d(64, 128, 1)
        #self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        #self.fc1 = nn.Linear(1024, 512)
        #self.fc2 = nn.Linear(512, 256)
        #self.fc3 = nn.Linear(256, 9)

        self.conv1 = torch.nn.Conv1d(channel, int(64/c_prune_rate), 1)
        self.conv2 = torch.nn.Conv1d(int(64/c_prune_rate), int(128/c_prune_rate), 1)
        self.conv3 = torch.nn.Conv1d(int(128/c_prune_rate), int(1024/c_prune_rate), 1)
        self.fc1 = nn.Linear(int(1024/c_prune_rate), int(512/c_prune_rate))
        self.fc2 = nn.Linear(int(512/c_prune_rate), int(256/c_prune_rate))
        self.fc3 = nn.Linear(int(256/c_prune_rate), 9)

        self.relu = nn.ReLU()

        #self.bn1 = nn.BatchNorm1d(64)
        #self.bn2 = nn.BatchNorm1d(128)
        #self.bn3 = nn.BatchNorm1d(1024)
        #self.bn4 = nn.BatchNorm1d(512)
        #self.bn5 = nn.BatchNorm1d(256)

        self.bn1 = nn.BatchNorm1d(int(64/c_prune_rate))
        self.bn2 = nn.BatchNorm1d(int(128/c_prune_rate))
        self.bn3 = nn.BatchNorm1d(int(1024/c_prune_rate))
        self.bn4 = nn.BatchNorm1d(int(512/c_prune_rate))
        self.bn5 = nn.BatchNorm1d(int(256/c_prune_rate))

        self.c_prune_rate = c_prune_rate

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))+ offset_map[N]
        x = torch.max(x, 2, keepdim=True)[0]
        #x = x.view(-1, 1024)
        x = x.view(-1, int(1024/self.c_prune_rate))

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64,c_prune_rate=1):
        super(STNkd, self).__init__()
        #self.conv1 = torch.nn.Conv1d(k, 64, 1)
        #self.conv2 = torch.nn.Conv1d(64, 128, 1)
        #self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        #self.fc1 = nn.Linear(1024, 512)
        #self.fc2 = nn.Linear(512, 256)
        #self.fc3 = nn.Linear(256, k * k)
        self.conv1 = torch.nn.Conv1d(k, int(64/c_prune_rate), 1)
        self.conv2 = torch.nn.Conv1d(int(64/c_prune_rate), int(128/c_prune_rate), 1)
        self.conv3 = torch.nn.Conv1d(int(128/c_prune_rate), int(1024/c_prune_rate), 1)
        self.fc1 = nn.Linear(int(1024/c_prune_rate), int(512/c_prune_rate))
        self.fc2 = nn.Linear(int(512/c_prune_rate), int(256/c_prune_rate))
        self.fc3 = nn.Linear(int(256/c_prune_rate), k * k)

        self.relu = nn.ReLU()

        #self.bn1 = nn.BatchNorm1d(64)
        #self.bn2 = nn.BatchNorm1d(128)
        #self.bn3 = nn.BatchNorm1d(1024)
        #self.bn4 = nn.BatchNorm1d(512)
        #self.bn5 = nn.BatchNorm1d(256)
        self.bn1 = nn.BatchNorm1d(int(64/c_prune_rate))
        self.bn2 = nn.BatchNorm1d(int(128/c_prune_rate))
        self.bn3 = nn.BatchNorm1d(int(1024/c_prune_rate))
        self.bn4 = nn.BatchNorm1d(int(512/c_prune_rate))
        self.bn5 = nn.BatchNorm1d(int(256/c_prune_rate))

        self.k = k
        self.c_prune_rate = c_prune_rate

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        #x = x.view(-1, 1024)
        x = x.view(-1, int(1024/self.c_prune_rate))

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3, c_prune_rate=1, noise=0):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel,c_prune_rate=c_prune_rate)
        #self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        #self.conv2 = torch.nn.Conv1d(64, 128, 1)
        #self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.conv1 = torch.nn.Conv1d(channel, int(64/c_prune_rate), 1)
        self.conv2 = torch.nn.Conv1d(int(64/c_prune_rate), int(128/c_prune_rate), 1)
        self.conv3 = torch.nn.Conv1d(int(128/c_prune_rate), int(1024/c_prune_rate), 1)

        #self.bn1 = nn.BatchNorm1d(64)
        #self.bn2 = nn.BatchNorm1d(128)
        #self.bn3 = nn.BatchNorm1d(1024)
        self.bn1 = nn.BatchNorm1d(int(64/c_prune_rate))
        self.bn2 = nn.BatchNorm1d(int(128/c_prune_rate))
        self.bn3 = nn.BatchNorm1d(int(1024/c_prune_rate))        
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=int(64/c_prune_rate),c_prune_rate=c_prune_rate)

        self.c_prune_rate = c_prune_rate
        self.noise = noise

    def forward(self, x):
        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        #x = x.view(-1, 1024)
        x = x.view(-1, int(1024/self.c_prune_rate))
        if self.global_feat:
            return x, trans, trans_feat
        else:
            #x = x.view(-1, 1024, 1).repeat(1, 1, N)
            x = x.view(-1, int(1024/self.c_prune_rate), 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss

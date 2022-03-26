import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction
from noise_layers import NoiseModule, NoiseConv, NoiseLinear 

class get_model(nn.Module):
    def __init__(self, num_class, normal_channel=True,c_prune_rate=1, noise=0):
        super(get_model, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel

        mlp_list_1 = [[32, 32, 64], [64, 64, 128], [64, 96, 128]]
        mlp_list_2 = [[64, 64, 128], [128, 128, 256], [128, 128, 256]]
        mlp_list_3 = [256, 512, 1024]

        if c_prune_rate != 1:
            temp_array = []
            for mlp in mlp_list_1:
              mlp = [int(c / c_prune_rate) for c in mlp]
              temp_array.append(mlp)

            mlp_list_1 = temp_array
            temp_array = []

            
            for mlp in mlp_list_2:
              mlp = [int(c / c_prune_rate) for c in mlp]
              temp_array.append(mlp)
 
            mlp_list_2 = temp_array
            temp_array = []    
            
            mlp_list_3 = [int(c / c_prune_rate) for c in mlp_list_3]

            print(mlp_list_1)
            print(mlp_list_2) 
            print(mlp_list_3) 
            
        #self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        #self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        #self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel, mlp_list_1, )
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], int(320/c_prune_rate), mlp_list_2, )
        self.sa3 = PointNetSetAbstraction(None, None, None, int(640/c_prune_rate)+3 , mlp_list_3, True, )

        #self.fc1 = nn.Linear(1024, 512)
        self.fc1 = nn.Linear(int(1024 / c_prune_rate), int(512 / c_prune_rate), )
        #self.bn1 = nn.BatchNorm1d(512)
        self.bn1 = nn.BatchNorm1d(int(512 / c_prune_rate))
        self.drop1 = nn.Dropout(0.4)
        #self.fc2 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(int(512 / c_prune_rate), int(256 / c_prune_rate), )
        #self.bn2 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(int(256 / c_prune_rate))
        self.drop2 = nn.Dropout(0.5)
        #self.fc3 = nn.Linear(256, num_class)
        self.fc3 = nn.Linear(int(256 / c_prune_rate), num_class, )

        self.c_prune_rate = c_prune_rate

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        #x = l3_points.view(B, 1024)
        x = l3_points.view(B, int(1024 / self.c_prune_rate))
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)


        return x,l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss



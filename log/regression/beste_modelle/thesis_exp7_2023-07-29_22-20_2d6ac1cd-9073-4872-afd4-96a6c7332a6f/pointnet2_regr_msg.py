import torch.nn as nn
import torch
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction
from torchmetrics.functional import r2_score
import matplotlib.pyplot as plt

def plot_pc_tensor(tensor, label=""):
    plt.figure()
    coordinates = tensor[0, :2, :].detach().numpy()
    plt.scatter(coordinates[0], coordinates[1], label=label)
    plt.legend()

class get_model(nn.Module):
    def __init__(self, normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320, [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 128)
        # This part is added to make it regression instead of classification
        self.bn3 = nn.BatchNorm1d(128)
        self.drop3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(128, 1)


    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        # plot_pc_tensor(xyz, label="before forward")
        l1_xyz, l1_points = self.sa1(xyz, norm)
        # plot_pc_tensor(l1_xyz, label="after sa1 xyz")
        # plot_pc_tensor(l1_points, label="after sa1 points")
        plt.show()
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.drop3(F.relu(self.bn3(self.fc3(x))))
        x = self.fc4(x)

        return x

class get_loss(nn.Module):
    def __init__(self, loss_fn):
        super(get_loss, self).__init__()
        self.loss_fn = loss_fn

    def forward(self, pred, target):
        if self.loss_fn == "mae":
            pred = torch.reshape(pred, target.shape)
            total_loss = F.l1_loss(pred, target)
        if self.loss_fn == "mse":
            pred = torch.reshape(pred, target.shape)
            total_loss = F.mse_loss(pred, target)
        if self.loss_fn == "r2":
            pred = torch.reshape(pred, target.shape)
            if len(pred) == 1:
                total_loss = abs(pred - target)
                total_loss = torch.tensor([total_loss])
            else:
                total_loss = r2_score(pred, target)
                total_loss = - total_loss

        return total_loss
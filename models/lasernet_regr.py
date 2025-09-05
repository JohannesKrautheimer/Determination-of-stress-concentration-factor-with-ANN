import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import r2_score

class get_model(nn.Module):
    def __init__(self, input_seq_length):
        super(get_model, self).__init__()
        self.input_seq_length = input_seq_length
        self.c1 = nn.Conv1d(in_channels=2, out_channels=32, kernel_size=11)
        self.bn1 = nn.BatchNorm1d(32)
        self.c2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5)
        self.bn2 = nn.BatchNorm1d(64)
        self.c3 = nn.Conv1d(in_channels=64, out_channels=256, kernel_size=5)
        self.bn3 = nn.BatchNorm1d(256)
        # Determine in_features for fc1-layer
        self.in_features_fc1 = input_seq_length - self.c1.kernel_size[0] + 1
        self.in_features_fc1 = self.in_features_fc1 - self.c2.kernel_size[0] + 1
        self.in_features_fc1 = int(self.in_features_fc1 / 2)
        self.in_features_fc1 = self.in_features_fc1 - self.c3.kernel_size[0] + 1
        self.in_features_fc1 = int(self.in_features_fc1 / 2)
        self.in_features_fc1 *= self.c3.out_channels
        self.fc1 = nn.Linear(self.in_features_fc1, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 1)
        self.max_pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, xy):
        B, _, _ = xy.shape
        # 1st Conv Layer
        x = self.c1(xy)
        x = self.bn1(x)
        x = F.relu(x)
        # 2nd Conv Layer
        x = self.c2(x)
        x = self.bn2(x)
        x = F.relu(x)
        ###
        x = self.max_pool(x)
        ###
        # 3rd Conv Layer
        x = self.c3(x)
        x = self.bn3(x)
        x = F.relu(x)
        ###
        x = self.max_pool(x)
        x = x.view(B, self.in_features_fc1)
        ###
        x = self.fc1(x)
        x = self.bn4(x)
        x = F.relu(x)
        ###
        x = self.fc2(x)

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
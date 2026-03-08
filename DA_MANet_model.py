"""
定义了1D-MCNN模型
"""

import torch.nn as nn
import torch
from timm.models.layers import DropPath
from MA import *
class MCNN_1D(nn.Module):
    def __init__(self, num_classes, features, dropout = 0):
        super(MCNN_1D, self).__init__()
        self.conv = nn.ModuleList([
            nn.Conv1d(1, 16, 1, padding=0),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Conv1d(3, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Conv1d(5, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU()
        ])

        self.output_layer = nn.Sequential(
            # Add dropout here
            nn.Flatten(),
            nn.Linear(48 * features, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, num_classes)

        )

    def forward(self, x):
        convs = []
        for i in range(0, len(self.conv), 3):
            conv_layer = self.conv[i]
            batch_norm = self.conv[i + 1]
            activation = self.conv[i + 2]
            #将不同尺度数据放入不同的卷积层中进行处理
            conv = activation(batch_norm(conv_layer(x[i // 3])))
            #处理后放入convs列表中
            convs.append(conv)
        x = torch.cat(convs, dim=1)
        #print(x.shape)
        x = self.output_layer(x)
        #print(x.shape)
        return x





class MA(nn.Module):
    def __init__(self, channels, factor=8):
        super(MA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class MACNN_1D(nn.Module):
    def __init__(self, num_classes, features, dropout = 0):
        super(MACNN_1D, self).__init__()
        self.size_1 = 16
        self.size_2 = 16
        self.size_3 = 16
        self.size = self.size_1 + self.size_2 + self.size_3
        self.conv = nn.ModuleList([
            nn.Conv1d(1, self.size_1, 1, padding=0),
            nn.BatchNorm1d(self.size_1),
            nn.LeakyReLU(),
            nn.Conv1d(3, self.size_2, 3, padding=1),
            nn.BatchNorm1d(self.size_2),
            nn.LeakyReLU(),
            nn.Conv1d(5, self.size_3, 3, padding=1),
            nn.BatchNorm1d(self.size_3),
            nn.LeakyReLU()
        ])

        self.scc = ScConv(self.size)


        self.output_layer_scc = nn.Sequential(
            # Add dropout here
            nn.Flatten(),
            nn.Linear(self.size * features, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, num_classes)
        )

        self.output_layer = nn.Sequential(
            # Add dropout here
            nn.Flatten(),
            nn.Linear(self.size * features, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, num_classes)
        )

        self.fc = nn.Linear(2 * num_classes, num_classes)



    def forward(self, x):
        convs = []
        for i in range(0, len(self.conv), 3):
            conv_layer = self.conv[i]
            batch_norm = self.conv[i + 1]
            activation = self.conv[i + 2]
            #将不同尺度数据放入不同的卷积层中进行处理
            conv = activation(batch_norm(conv_layer(x[i // 3])))
            #处理后放入convs列表中
            convs.append(conv)
        #print(convs[0].shape)
        #print(convs[1].shape)
        #print(convs[2].shape)
        x = torch.cat(convs, dim=1)

        x_2d = x.unsqueeze(-1)  # (S, 48, 11) -> (S, 48, 11, 1)
        #x_2d = x.unflatten(1, (8, 8))

        x_ma = self.scc(x_2d)

        x_res = x_ma.squeeze(-1)  # (S, 48, 11, 1) -> (S, 48, 11)



        #print(x.shape)
        x_res = self.output_layer_scc(x_res)  # (S, 5)
        #x = self.output_layer(x)  # (S, 5)

        #x = torch.cat([x, x_res], dim=1)  # (S, 10)
        #x = self.fc(x)  # (S, 5)


        #print(x.shape)
        return x_res
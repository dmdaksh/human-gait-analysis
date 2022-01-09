import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1_acc = nn.Conv1d(in_channels=3,
                                   out_channels=16,
                                   kernel_size=3,
                                   stride=1,
                                   dilation=2)
        self.dropout2d1_acc = nn.Dropout2d(p=0.125)
        self.conv1_gyr = nn.Conv1d(in_channels=3,
                                   out_channels=16,
                                   kernel_size=3,
                                   stride=1,
                                   dilation=2)
        self.dropout2d1_gyr = nn.Dropout2d(p=0.125)
        self.conv1_mag = nn.Conv1d(in_channels=3,
                                   out_channels=16,
                                   kernel_size=3,
                                   stride=1,
                                   dilation=2)
        self.dropout2d1_mag = nn.Dropout2d(p=0.125)

        self.conv2_acc = nn.Conv1d(in_channels=16,
                                   out_channels=64,
                                   kernel_size=3,
                                   stride=1,
                                   dilation=2)
        self.dropout2d2_acc = nn.Dropout2d(p=0.25)
        self.conv2_gyr = nn.Conv1d(in_channels=16,
                                   out_channels=64,
                                   kernel_size=3,
                                   stride=1,
                                   dilation=2)
        self.dropout2d2_gyr = nn.Dropout2d(p=0.25)
        self.conv2_mag = nn.Conv1d(in_channels=16,
                                   out_channels=64,
                                   kernel_size=3,
                                   stride=1,
                                   dilation=2)
        self.dropout2d2_mag = nn.Dropout2d(p=0.25)

        self.conv3_acc = nn.Conv1d(in_channels=64,
                                   out_channels=256,
                                   kernel_size=5,
                                   stride=1,
                                   dilation=2)
        self.dropout2d3_acc = nn.Dropout2d(p=0.5)
        self.conv3_gyr = nn.Conv1d(in_channels=64,
                                   out_channels=256,
                                   kernel_size=5,
                                   stride=1,
                                   dilation=2)
        self.dropout2d3_gyr = nn.Dropout2d(p=0.5)
        self.conv3_mag = nn.Conv1d(in_channels=64,
                                   out_channels=256,
                                   kernel_size=5,
                                   stride=1,
                                   dilation=2)
        self.dropout2d3_mag = nn.Dropout2d(p=0.5)

        self.flatten = nn.Flatten(start_dim=1)

        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features=24576, out_features=4096)

        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=4096, out_features=256)

        self.fc3 = nn.Linear(in_features=256, out_features=9)

    def forward(self, acc, gyr, mag):
        # input (batch*3*300)
        acc = self.dropout2d1_acc(
            F.gelu(F.max_pool1d(self.conv1_acc(acc), kernel_size=2, stride=2)))
        gyr = self.dropout2d1_gyr(
            F.gelu(F.max_pool1d(self.conv1_gyr(gyr), kernel_size=2, stride=2)))
        mag = self.dropout2d1_mag(
            F.gelu(F.max_pool1d(self.conv1_mag(mag), kernel_size=2, stride=2)))

        # input [dilation:2 = (batch*16*296) (batch*16*148)]
        acc = self.dropout2d2_acc(
            F.gelu(F.max_pool1d(self.conv2_acc(acc), kernel_size=2, stride=2)))
        gyr = self.dropout2d2_gyr(
            F.gelu(F.max_pool1d(self.conv2_gyr(gyr), kernel_size=2, stride=2)))
        mag = self.dropout2d2_mag(
            F.gelu(F.max_pool1d(self.conv2_mag(mag), kernel_size=2, stride=2)))

        # input [dilation:2 = (batch*64*144) (batch*64*72)]
        acc = self.dropout2d3_acc(
            F.gelu(F.max_pool1d(self.conv3_acc(acc), kernel_size=2, stride=2)))
        gyr = self.dropout2d3_gyr(
            F.gelu(F.max_pool1d(self.conv3_gyr(gyr), kernel_size=2, stride=2)))
        mag = self.dropout2d3_mag(
            F.gelu(F.max_pool1d(self.conv3_mag(mag), kernel_size=2, stride=2)))
        # output [dilation:2 = (batch*256*64) (batch*256*32)]

        # concatenating acc, gyr, mag cnn models
        x = torch.cat((acc, gyr, mag), dim=1)
        # output [dilation:2 = (batch*768*32)]

        # flattening cnn
        x = self.flatten(x)

        x = self.dropout1(F.gelu(self.fc1(x)))

        x = self.dropout2(F.gelu(self.fc2(x)))

        # output
        x = self.fc3(x)

        return x


class Transformer(nn.Module):

    def __init__(self):
        pass

    def forward(self, x):
        return x


class CNNOld(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=3,
                               out_channels=16,
                               kernel_size=3,
                               stride=1,
                               dilation=2)
        self.dropout2d1 = nn.Dropout2d(p=0.125)
        # self.batchnorm1   =   nn.BatchNorm1d(num_features=8)
        self.conv1_1 = nn.Conv1d(in_channels=16,
                                 out_channels=16,
                                 kernel_size=3,
                                 stride=1,
                                 dilation=2)
        self.dropout2d1_1 = nn.Dropout2d(p=0.125)
        # self.batchnorm1_1 =   nn.BatchNorm1d(num_features=16)

        self.conv2 = nn.Conv1d(in_channels=16,
                               out_channels=64,
                               kernel_size=5,
                               stride=2,
                               dilation=2)
        self.dropout2d2 = nn.Dropout2d(p=0.25)
        # self.batchnorm2     = nn.BatchNorm1d(num_features=32)
        self.conv2_1 = nn.Conv1d(in_channels=64,
                                 out_channels=64,
                                 kernel_size=5,
                                 stride=2,
                                 dilation=2)
        self.dropout2d2_1 = nn.Dropout2d(p=0.25)
        # self.batchnorm2_1   = nn.BatchNorm1d(num_features=64)

        self.conv3 = nn.Conv1d(in_channels=64,
                               out_channels=256,
                               kernel_size=5,
                               stride=2,
                               dilation=2)
        self.dropout2d3 = nn.Dropout2d(p=0.5)
        # self.batchnorm3     = nn.BatchNorm1d(num_features=128)
        self.conv3_1 = nn.Conv1d(in_channels=256,
                                 out_channels=256,
                                 kernel_size=5,
                                 stride=2,
                                 dilation=2)
        # self.batchnorm3_1   = nn.BatchNorm1d(num_features=256)
        self.dropout2d3_1 = nn.Dropout2d(p=0.5)

        self.flatten = nn.Flatten(start_dim=1)

        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features=6912, out_features=1024)  # 6912

        self.dropout2 = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(in_features=1024, out_features=128)

        self.dropout3 = nn.Dropout(p=0.125)
        self.fc3 = nn.Linear(in_features=128, out_features=9)

    def forward(self, acc, gyr, mag):
        # input (batch*1*300)
        acc = self.dropout2d1(
            F.gelu(F.max_pool1d(self.conv1(acc), kernel_size=2, stride=1)))
        gyr = self.dropout2d1(
            F.gelu(F.max_pool1d(self.conv1(gyr), kernel_size=2, stride=1)))
        mag = self.dropout2d1(
            F.gelu(F.max_pool1d(self.conv1(mag), kernel_size=2, stride=1)))
        # (batch*16*298) [dilation:2 = (batch*16*296) (batch*16*295)]

        acc = self.dropout2d1_1(
            F.gelu(F.max_pool1d(self.conv1_1(acc), kernel_size=2, stride=1)))
        gyr = self.dropout2d1_1(
            F.gelu(F.max_pool1d(self.conv1_1(gyr), kernel_size=2, stride=1)))
        mag = self.dropout2d1_1(
            F.gelu(F.max_pool1d(self.conv1_1(mag), kernel_size=2, stride=1)))
        # (batch*16*298) [dilation:2 = (batch*16*291) (batch*16*290)]

        acc = self.dropout2d2(
            F.gelu(F.max_pool1d(self.conv2(acc), kernel_size=2, stride=1)))
        gyr = self.dropout2d2(
            F.gelu(F.max_pool1d(self.conv2(gyr), kernel_size=2, stride=1)))
        mag = self.dropout2d2(
            F.gelu(F.max_pool1d(self.conv2(mag), kernel_size=2, stride=1)))
        # (batch*16*298) [dilation:2 = (batch*64*141[.5]) (batch*64*140)]

        acc = self.dropout2d2_1(
            F.gelu(F.max_pool1d(self.conv2_1(acc), kernel_size=2, stride=1)))
        gyr = self.dropout2d2_1(
            F.gelu(F.max_pool1d(self.conv2_1(gyr), kernel_size=2, stride=1)))
        mag = self.dropout2d2_1(
            F.gelu(F.max_pool1d(self.conv2_1(mag), kernel_size=2, stride=1)))
        # (batch*16*298) [dilation:2 = (batch*64*66[.5]) (batch*64*65)]

        acc = self.dropout2d3(
            F.gelu(F.max_pool1d(self.conv3(acc), kernel_size=2, stride=1)))
        gyr = self.dropout2d3(
            F.gelu(F.max_pool1d(self.conv3(gyr), kernel_size=2, stride=1)))
        mag = self.dropout2d3(
            F.gelu(F.max_pool1d(self.conv3(mag), kernel_size=2, stride=1)))
        # (batch*16*298) [dilation:2 = (batch*256*29) (batch*256*28)]

        acc = self.dropout2d3_1(
            F.gelu(F.max_pool1d(self.conv3_1(acc), kernel_size=2, stride=1)))
        gyr = self.dropout2d3_1(
            F.gelu(F.max_pool1d(self.conv3_1(gyr), kernel_size=2, stride=1)))
        mag = self.dropout2d3_1(
            F.gelu(F.max_pool1d(self.conv3_1(mag), kernel_size=2, stride=1)))
        # (batch*16*298) [dilation:2 = (batch*256*10[.5]) (batch*256*9)]

        # concatenating acc, gyr, mag cnn models
        x = torch.cat((acc, gyr, mag), dim=1)
        # [dilation:2 = (batch*768*9)]

        # flattening cnn
        x = self.flatten(x)

        x = self.dropout1(F.gelu(self.fc1(x)))

        x = self.dropout2(F.gelu(self.fc2(x)))

        # output
        x = self.fc3(x)

        return x


class CNNAlternate(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1_acc = nn.Conv1d(in_channels=3,
                                   out_channels=16,
                                   kernel_size=3,
                                   stride=1,
                                   dilation=2)
        self.conv1_1_acc = nn.Conv1d(in_channels=16,
                                     out_channels=16,
                                     kernel_size=3,
                                     stride=1,
                                     dilation=2)
        self.dropout2d1_acc = nn.Dropout2d(p=0.125)
        self.conv1_gyr = nn.Conv1d(in_channels=3,
                                   out_channels=16,
                                   kernel_size=3,
                                   stride=1,
                                   dilation=2)
        self.conv1_1_gyr = nn.Conv1d(in_channels=16,
                                     out_channels=16,
                                     kernel_size=3,
                                     stride=1,
                                     dilation=2)
        self.dropout2d1_gyr = nn.Dropout2d(p=0.125)
        self.conv1_mag = nn.Conv1d(in_channels=3,
                                   out_channels=16,
                                   kernel_size=3,
                                   stride=1,
                                   dilation=2)
        self.conv1_1_mag = nn.Conv1d(in_channels=16,
                                     out_channels=16,
                                     kernel_size=3,
                                     stride=1,
                                     dilation=2)
        self.dropout2d1_mag = nn.Dropout2d(p=0.125)

        self.conv2_acc = nn.Conv1d(in_channels=16,
                                   out_channels=64,
                                   kernel_size=3,
                                   stride=1,
                                   dilation=2)
        self.conv2_1_acc = nn.Conv1d(in_channels=64,
                                     out_channels=64,
                                     kernel_size=3,
                                     stride=1,
                                     dilation=2)
        self.dropout2d2_acc = nn.Dropout2d(p=0.125)
        self.conv2_gyr = nn.Conv1d(in_channels=16,
                                   out_channels=64,
                                   kernel_size=3,
                                   stride=1,
                                   dilation=2)
        self.conv2_1_gyr = nn.Conv1d(in_channels=64,
                                     out_channels=64,
                                     kernel_size=3,
                                     stride=1,
                                     dilation=2)
        self.dropout2d2_gyr = nn.Dropout2d(p=0.125)
        self.conv2_mag = nn.Conv1d(in_channels=16,
                                   out_channels=64,
                                   kernel_size=3,
                                   stride=1,
                                   dilation=2)
        self.conv2_1_mag = nn.Conv1d(in_channels=64,
                                     out_channels=64,
                                     kernel_size=3,
                                     stride=1,
                                     dilation=2)
        self.dropout2d2_mag = nn.Dropout2d(p=0.125)

        self.conv3_acc = nn.Conv1d(in_channels=64,
                                   out_channels=256,
                                   kernel_size=5,
                                   stride=1,
                                   dilation=2)
        self.conv3_1_acc = nn.Conv1d(in_channels=256,
                                     out_channels=256,
                                     kernel_size=5,
                                     stride=1,
                                     dilation=2)
        self.dropout2d3_acc = nn.Dropout2d(p=0.25)
        self.conv3_gyr = nn.Conv1d(in_channels=64,
                                   out_channels=256,
                                   kernel_size=5,
                                   stride=1,
                                   dilation=2)
        self.conv3_1_gyr = nn.Conv1d(in_channels=256,
                                     out_channels=256,
                                     kernel_size=5,
                                     stride=1,
                                     dilation=2)
        self.dropout2d3_gyr = nn.Dropout2d(p=0.25)
        self.conv3_mag = nn.Conv1d(in_channels=64,
                                   out_channels=256,
                                   kernel_size=5,
                                   stride=1,
                                   dilation=2)
        self.conv3_1_mag = nn.Conv1d(in_channels=256,
                                     out_channels=256,
                                     kernel_size=5,
                                     stride=1,
                                     dilation=2)
        self.dropout2d3_mag = nn.Dropout2d(p=0.25)

        self.flatten = nn.Flatten(start_dim=1)

        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features=19968, out_features=4096)

        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=4096, out_features=256)

        self.fc3 = nn.Linear(in_features=256, out_features=9)

    def forward(self, acc, gyr, mag):
        # input (batch*3*300)
        acc = self.dropout2d1_acc(
            F.gelu(
                F.max_pool1d(self.conv1_1_acc(self.conv1_acc(acc)),
                             kernel_size=2,
                             stride=2)))
        gyr = self.dropout2d1_gyr(
            F.gelu(
                F.max_pool1d(self.conv1_1_gyr(self.conv1_gyr(gyr)),
                             kernel_size=2,
                             stride=2)))
        mag = self.dropout2d1_mag(
            F.gelu(
                F.max_pool1d(self.conv1_1_mag(self.conv1_mag(mag)),
                             kernel_size=2,
                             stride=2)))

        # input [dilation:2 = (batch*16*296) (batch*16*292) (batch*16*146)]
        acc = self.dropout2d2_acc(
            F.gelu(
                F.max_pool1d(self.conv2_1_acc(self.conv2_acc(acc)),
                             kernel_size=2,
                             stride=2)))
        gyr = self.dropout2d2_gyr(
            F.gelu(
                F.max_pool1d(self.conv2_1_gyr(self.conv2_gyr(gyr)),
                             kernel_size=2,
                             stride=2)))
        mag = self.dropout2d2_mag(
            F.gelu(
                F.max_pool1d(self.conv2_1_mag(self.conv2_mag(mag)),
                             kernel_size=2,
                             stride=2)))

        # input [dilation:2 = (batch*64*142) (batch*16*138) (batch*64*69)]
        acc = self.dropout2d3_acc(
            F.gelu(
                F.max_pool1d(self.conv3_1_acc(self.conv3_acc(acc)),
                             kernel_size=2,
                             stride=2)))
        gyr = self.dropout2d3_gyr(
            F.gelu(
                F.max_pool1d(self.conv3_1_gyr(self.conv3_gyr(gyr)),
                             kernel_size=2,
                             stride=2)))
        mag = self.dropout2d3_mag(
            F.gelu(
                F.max_pool1d(self.conv3_1_mag(self.conv3_mag(mag)),
                             kernel_size=2,
                             stride=2)))
        # output [dilation:2 = (batch*256*61) (batch*256*53) (batch*256*26+[0.5])]

        # concatenating acc, gyr, mag cnn models
        x = torch.cat((acc, gyr, mag), dim=1)
        # output [dilation:2 = (batch*768*26)]

        # flattening cnn
        x = self.flatten(x)

        x = self.dropout1(F.gelu(self.fc1(x)))

        x = self.dropout2(F.gelu(self.fc2(x)))

        # output
        x = self.fc3(x)

        return x

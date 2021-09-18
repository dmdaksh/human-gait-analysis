import torch
import torch.nn as nn
import torch.nn.functional as F
from gait import log
import pdb


logger = log.get_logger(__name__)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1          = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3, stride=1, dilation=2)
        self.dropout2d1     = nn.Dropout2d(p=0.25)
        # self.batchnorm1   =   nn.BatchNorm1d(num_features=8)
        self.conv1_1        = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1, dilation=2)
        self.dropout2d1_1   = nn.Dropout2d(p=0.25)
        # self.batchnorm1_1 =   nn.BatchNorm1d(num_features=16)

        self.conv2          = nn.Conv1d(in_channels=16, out_channels=64, kernel_size=5, stride=2, dilation=2)
        self.dropout2d2     = nn.Dropout2d(p=0.25)
        # self.batchnorm2     = nn.BatchNorm1d(num_features=32)
        self.conv2_1        = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, stride=2, dilation=2)
        self.dropout2d2_1   = nn.Dropout2d(p=0.25)
        # self.batchnorm2_1   = nn.BatchNorm1d(num_features=64)
        
        self.conv3          = nn.Conv1d(in_channels=64, out_channels=256, kernel_size=5, stride=2, dilation=2)
        self.dropout2d3     = nn.Dropout2d(p=0.5)
        # self.batchnorm3     = nn.BatchNorm1d(num_features=128)
        self.conv3_1        = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=5, stride=2, dilation=2)
        # self.batchnorm3_1   = nn.BatchNorm1d(num_features=256)
        self.dropout2d3_1   = nn.Dropout2d(p=0.5)

        self.flatten = nn.Flatten(start_dim=1)
        
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features=6912, out_features=1024) # 6912
        
        self.dropout2 = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(in_features=1024, out_features=128)
        
        self.dropout3 = nn.Dropout(p=0.25)
        self.fc3 = nn.Linear(in_features=128, out_features=9)

    
    def forward(self, acc, gyr, mag):
        # input (batch*1*300)
        acc = self.dropout2d1(F.gelu(F.avg_pool1d(self.conv1(acc), kernel_size=2, stride=1)))
        gyr = self.dropout2d1(F.gelu(F.avg_pool1d(self.conv1(gyr), kernel_size=2, stride=1)))
        mag = self.dropout2d1(F.gelu(F.avg_pool1d(self.conv1(mag), kernel_size=2, stride=1)))
        # (batch*16*298) [dilation:2 = (batch*16*296) (batch*16*295)]
        logger.debug(f'acc.shape: {acc.shape}, gyr.shape: {gyr.shape}, mag.shape {mag.shape}')
        
        acc = self.dropout2d1_1(F.gelu(F.avg_pool1d(self.conv1_1(acc), kernel_size=2, stride=1)))
        gyr = self.dropout2d1_1(F.gelu(F.avg_pool1d(self.conv1_1(gyr), kernel_size=2, stride=1)))
        mag = self.dropout2d1_1(F.gelu(F.avg_pool1d(self.conv1_1(mag), kernel_size=2, stride=1)))
        # (batch*16*298) [dilation:2 = (batch*16*291) (batch*16*290)]
        logger.debug(f'acc.shape: {acc.shape}, gyr.shape: {gyr.shape}, mag.shape {mag.shape}')
        
        acc = self.dropout2d2(F.gelu(F.avg_pool1d(self.conv2(acc), kernel_size=2, stride=1)))
        gyr = self.dropout2d2(F.gelu(F.avg_pool1d(self.conv2(gyr), kernel_size=2, stride=1)))
        mag = self.dropout2d2(F.gelu(F.avg_pool1d(self.conv2(mag), kernel_size=2, stride=1)))
        # (batch*16*298) [dilation:2 = (batch*64*141[.5]) (batch*64*140)]
        logger.debug(f'acc.shape: {acc.shape}, gyr.shape: {gyr.shape}, mag.shape {mag.shape}')

        acc = self.dropout2d2_1(F.gelu(F.avg_pool1d(self.conv2_1(acc), kernel_size=2, stride=1)))
        gyr = self.dropout2d2_1(F.gelu(F.avg_pool1d(self.conv2_1(gyr), kernel_size=2, stride=1)))
        mag = self.dropout2d2_1(F.gelu(F.avg_pool1d(self.conv2_1(mag), kernel_size=2, stride=1)))
        # (batch*16*298) [dilation:2 = (batch*64*66[.5]) (batch*64*65)]
        logger.debug(f'acc.shape: {acc.shape}, gyr.shape: {gyr.shape}, mag.shape {mag.shape}')

        acc = self.dropout2d3(F.gelu(F.avg_pool1d(self.conv3(acc), kernel_size=2, stride=1)))
        gyr = self.dropout2d3(F.gelu(F.avg_pool1d(self.conv3(gyr), kernel_size=2, stride=1)))
        mag = self.dropout2d3(F.gelu(F.avg_pool1d(self.conv3(mag), kernel_size=2, stride=1)))
        # (batch*16*298) [dilation:2 = (batch*256*29) (batch*256*28)]
        logger.debug(f'acc.shape: {acc.shape}, gyr.shape: {gyr.shape}, mag.shape {mag.shape}')

        acc = self.dropout2d3_1(F.gelu(F.avg_pool1d(self.conv3_1(acc), kernel_size=2, stride=1)))
        gyr = self.dropout2d3_1(F.gelu(F.avg_pool1d(self.conv3_1(gyr), kernel_size=2, stride=1)))
        mag = self.dropout2d3_1(F.gelu(F.avg_pool1d(self.conv3_1(mag), kernel_size=2, stride=1)))
        # (batch*16*298) [dilation:2 = (batch*256*10[.5]) (batch*256*9)]
        logger.debug(f'acc.shape: {acc.shape}, gyr.shape: {gyr.shape}, mag.shape {mag.shape}')

        # concatenating acc, gyr, mag cnn models
        x = torch.cat((acc, gyr, mag), dim=1)
        # [dilation:2 = (batch*768*9)]
        logger.debug(f'concat x.shape: {x.shape}')


        # flattening cnn
        x = self.flatten(x)
        logger.debug(f'flatten x.shape: {x.shape}')

        x = self.dropout1(F.gelu(self.fc1(x)))
        logger.debug(f'fc1 x.shape: {x.shape}')

        x = self.dropout2(F.gelu(self.fc2(x)))
        logger.debug(f'fc2 x.shape: {x.shape}')
        
        # output
        x = self.fc3(x)
        logger.debug(f'fc3 x.shape: {x.shape}')

        return x


class Transformer(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        return x
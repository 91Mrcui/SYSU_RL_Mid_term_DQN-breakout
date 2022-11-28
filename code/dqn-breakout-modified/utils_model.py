import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, action_dim, device):
        super(DQN, self).__init__()
        #输入：[1,4,84,84]，4通道
        self.__conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, bias=False)
        self.__conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.__conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)
        self.__fc_v1 = nn.Linear(64*7*7, 256)
        self.__fc_v2 = nn.Linear(256, 1)
        self.__fc_a1 = nn.Linear(64*7*7, 256)
        self.__fc_a2 = nn.Linear(256, action_dim)
        self.__device = device

    def forward(self, x):
        x = x / 255.
        x = F.relu(self.__conv1(x))
        x = F.relu(self.__conv2(x))
        x = F.relu(self.__conv3(x))
        x = x.view(x.size(0), -1)
        v = F.relu(self.__fc_v1(x))
        v = self.__fc_v2(v)
        a = F.relu(self.__fc_a1(x))
        a = self.__fc_a2(a)
        out = v.expand_as(a) + (a - a.mean().expand_as(a))
        return out
    @staticmethod
    def init_weights(module):   #初始化网络权重，静态方法
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            module.bias.data.fill_(0.0)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")

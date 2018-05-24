import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.use_cuda = args.cuda
        self.m = data.m
        self.w = args.window

        self.weight = Parameter(torch.Tensor(self.w, self.m))
        self.bias = Parameter(torch.zeros(self.m))
        nn.init.xavier_normal(self.weight)

        self.output = None;
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid;
        if (args.output_fun == 'tanh'):
            self.output = F.tanh;

    def forward(self, x):
        # x: batch x window (self.P) x #signal (m)
        batch_size = x.size(0);
        x = torch.sum(x * self.weight, dim=1) + self.bias
        if (self.output != None):
            x = self.output(x);
        return x;




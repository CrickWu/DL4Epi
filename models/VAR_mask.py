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

        self.output = None;
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid;
        if (args.output_fun == 'tanh'):
            self.output = F.tanh;

        self.adj = data.adj
        self.mask_mat = Parameter(torch.Tensor(self.w, self.m, self.m))
        self.bias = Parameter(torch.zeros(self.m))

        nn.init.xavier_normal(self.mask_mat)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        # x: batch x window (self.P) x #signal (m)
        masked_adj = self.adj.view(1, self.m, self.m) * self.mask_mat
        masked_adj = masked_adj.view(self.w * self.m, self.m)
        x = x.view(-1, self.w * self.m)
        x = x.matmul(masked_adj)
        x = x + self.bias

        if (self.output != None):
            x = self.output(x)
        return x;




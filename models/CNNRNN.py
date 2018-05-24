import torch
import torch.nn as nn
from torch.nn import Parameter

class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.use_cuda = args.cuda
        self.P = args.window
        self.m = data.m
        self.hidR = args.hidRNN
        self.GRU1 = nn.GRU(self.m, self.hidR)

        self.mask_mat = Parameter(torch.Tensor(self.m, self.m))
        self.adj = data.adj

        self.dropout = nn.Dropout(p=args.dropout)
        self.linear1 = nn.Linear(self.hidR, self.m)
        self.output = None
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid
        if (args.output_fun == 'tanh'):
            self.output = F.tanh

    def forward(self, x):
        # x: batch x window (self.P) x #signal (m)
        # first transform
        masked_adj = self.adj * self.mask_mat
        x = x.matmul(masked_adj)
        # RNN
        # r: window (self.P) x batch x #signal (m)
        r = x.permute(1, 0, 2).contiguous()
        _, r = self.GRU1(r)
        r = self.dropout(torch.squeeze(r, 0))

        res = self.linear1(r)

        if self.output is not None:
            res = self.output(res).float()

        return res

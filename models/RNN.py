import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.use_cuda = args.cuda
        self.P = args.window
        self.m = data.m
        self.hidR = args.hidRNN
        self.GRU1 = nn.GRU(self.m, self.hidR)
        self.dropout = nn.Dropout(p=args.dropout)
        self.linear1 = nn.Linear(self.hidR, self.m)
        self.output = None
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid
        if (args.output_fun == 'tanh'):
            self.output = F.tanh
        self.sigmoid = F.sigmoid

    def forward(self, x):
        # x: batch x window (self.P) x #signal (m)
        # RNN
        # r: #signal (m) x batch x window (self.P)
        r = x.permute(1, 0, 2).contiguous()
        _, r = self.GRU1(r)
        r = self.dropout(torch.squeeze(r, 0))
        res = self.linear1(r)

        if self.output is not None:
            res = self.output(res).float()

        return res

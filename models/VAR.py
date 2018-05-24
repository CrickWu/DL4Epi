import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.use_cuda = args.cuda
        self.m = data.m
        self.w = args.window

        self.linear = nn.Linear(self.m * self.w, self.m);
        self.output = None;
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid;
        if (args.output_fun == 'tanh'):
            self.output = F.tanh;

    def forward(self, x):
        x = x.view(-1, self.m * self.w);
        x = self.linear(x);
        if (self.output != None):
            x = self.output(x);
        return x;




import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.use_cuda = args.cuda
        self.m = data.m
        self.w = args.window

        self.linear = nn.Linear(self.w, 1);
        self.output = None;
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid;
        if (args.output_fun == 'tanh'):
            self.output = F.tanh;

    def forward(self, x):
        batch_size = x.size(0);
        x = x.transpose(2,1).contiguous();
        x = x.view(batch_size * self.m, self.w);
        x = self.linear(x);
        x = x.view(batch_size, self.m);
        if (self.output != None):
            x = self.output(x);
        return x;




import torch
import torch.nn as nn

class Normalize(nn.Module):
    def __init__(self, mean, std, device=None):
        super(Normalize, self).__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mean = torch.tensor(mean).view(1, -1, 1, 1)
        std = torch.tensor(std).view(1, -1, 1, 1)
        self.mean = mean
        self.std = std
        # self.register_buffer('mean', mean)
        # self.register_buffer('std',  std)

    def forward(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

class Denormalize(nn.Module):
    def __init__(self, mean, std, device=None):
        super(Denormalize, self).__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mean = torch.tensor(mean).view(1, -1, 1, 1)
        std = torch.tensor(std).view(1, -1, 1, 1)
        self.mean = mean
        self.std = std
        # self.register_buffer('mean', mean)
        # self.register_buffer('std',  std)

    def forward(self, x):
        return x * self.std.to(x.device) + self.mean.to(x.device)
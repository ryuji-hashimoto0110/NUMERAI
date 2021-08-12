import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseLineModel(nn.Module):
    def __init__(self, input_dim):
        super(BaseLineModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8, 1)
        
    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

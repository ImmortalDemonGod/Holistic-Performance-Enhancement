import torch
import torch.nn as nn

class CustomSigmoidActivation(nn.Module):
    def __init__(self, min_value=-1, max_value=9):
        super(CustomSigmoidActivation, self).__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.scale = max_value - min_value

    def forward(self, x):
        # Apply sigmoid to constrain between 0 and 1, then scale and shift
        return torch.sigmoid(x) * self.scale + self.min_value

import torch.nn as nn
import torch

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.conv2d(x)

            # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-3), x.size(-2), x.size(-1))  # (samples * timesteps, input_size)

        avg_out = torch.mean(x_reshape, dim=1, keepdim=True)
        max_out, _ = torch.max(x_reshape, dim=1, keepdim=True)
        _temp_x = torch.cat([avg_out, max_out], dim=1)
        _temp_x = self.conv1(_temp_x)
        _temp_x = self.sigmoid(_temp_x)
        y = torch.mul(_temp_x, x_reshape)
        y = y.contiguous().view(x.size(0), x.size(1), -1, x.size(-2), x.size(-1))

        return y
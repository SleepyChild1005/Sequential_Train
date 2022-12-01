import torch
import torch.nn as nn

class TimeDistributedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, dropout=False):
        super(TimeDistributedConv2d, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.batch2d = nn.BatchNorm2d(out_channels)
        
        model = [self.conv2d, self.batch2d]
        
        if dropout:
            model.insert(1, nn.Dropout(0.5, inplace=True))
            # model.append()
            # model.append(nn.Dropout(0.5, inplace=True))

        self.seq = nn.Sequential(*model)

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.conv2d(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-3),x.size(-2), x.size(-1))     # (samples * timesteps, input_size)

        y = self.seq(x_reshape)
        y = y.contiguous().view(x.size(0), x.size(1), -1, x.size(-2), x.size(-1))  # (samples, timesteps, output_size)
        return y

class TimeDistributedMaxPool(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super(TimeDistributedMaxPool, self).__init__()
        self.maxpool2d = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        
    def forward(self, x):
        if len(x.size()) <= 2:
            return self.maxpool2d(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-3),x.size(-2), x.size(-1))     # (samples * timesteps, input_size)

        y = self.maxpool2d(x_reshape)
        y = y.contiguous().view(x.size(0), x.size(1), x.size(2), x.size(-2)//2, x.size(-1)//2)  # (samples, timesteps, output_size)
        return y

class TimeDistributedUpsampling(nn.Module):
    def __init__(self, scale_factor=2, mode='nearest'):
        super(TimeDistributedUpsampling, self).__init__()
        self.upsampling = nn.Upsample(scale_factor=scale_factor, mode=mode)
        
    def forward(self, x):
        if len(x.size()) <= 2:
            return self.upsampling(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-3),x.size(-2), x.size(-1))     # (samples * timesteps, input_size)

        y = self.upsampling(x_reshape)
        y = y.contiguous().view(x.size(0), x.size(1), x.size(2), x.size(-2)*2, x.size(-1)*2)  # (samples, timesteps, output_size)
        return y

class TimeDistributedSigmoid(nn.Module):
    def __init__(self, in_channels, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False):
        super(TimeDistributedSigmoid, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-3),x.size(-2), x.size(-1))     # (samples * timesteps, input_size)
        conv_x = self.conv2d(x_reshape)
        y = self.sigmoid(conv_x)
        y = y.contiguous().view(x.size(0), x.size(1), -1, x.size(-2), x.size(-1))  # (samples, timesteps, output_size)
        return y

# nn.ConvTranspose2d(start_neurons * 8, start_neurons * 4, kernel_size=3, stride=2, padding=1, output_padding=1)

class TimeDistributedConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, output_padding=1, stride=1, padding=1, bias=False, dropout=False):
        # output_padding=1
        super(TimeDistributedConvTranspose2d, self).__init__()
        self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, output_padding=output_padding, bias=bias)
        self.batch2d = nn.BatchNorm2d(out_channels)
        model = [self.conv2d, self.batch2d]

        if dropout:
            model.append(nn.Dropout(0.5, inplace=True))

        self.seq = nn.Sequential(*model)
    def forward(self, x):
        if len(x.size()) <= 2:
            return self.conv2d(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-3),x.size(-2), x.size(-1))     # (samples * timesteps, input_size)
        # print('Time_tranpose1', x_reshape.shape)
        y = self.seq(x_reshape)
        # print('Time_tranpose2',y.shape)
        y = y.contiguous().view(x.size(0), x.size(1), -1, x.size(-2)*2, x.size(-1)*2)  # (samples, timesteps, output_size)
        # print('Time_tranpose3',y.shape)
        return y

class TimeDistributedSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(TimeDistributedSpatialAttention, self).__init__()
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

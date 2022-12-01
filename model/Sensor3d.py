import torch
import torch.nn as nn

from .TimeDistributedLayer import TimeDistributedConv2d, TimeDistributedMaxPool, TimeDistributedUpsampling
from .BiConvLSTM import BiConvLSTM
# used in trainer
class DeepSequentialNet(nn.Module):
    # n_class for multiple output (== label numbers)
    def __init__(self, image_size, device, n_class=1):
        super(DeepSequentialNet, self).__init__()
        self.device = device

        self.encoding_block1 = nn.Sequential(
            TimeDistributedConv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ELU(inplace=True),
            TimeDistributedConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ELU(inplace=True)
        )
        self.encoding_block2 = nn.Sequential(
            TimeDistributedConv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ELU(inplace=True),
            TimeDistributedConv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ELU(inplace=True)
        )
        self.encoding_block3 = nn.Sequential(
            TimeDistributedConv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ELU(inplace=True),
            TimeDistributedConv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ELU(inplace=True)
        )

        self.biCLSTM1 = BiConvLSTM(input_size=(image_size//(2**3),image_size//(2**3)), input_dim=256, hidden_dim=512, kernel_size=(3,3), num_layers=3, device=self.device)

        self.decoding_block3 = nn.Sequential(
            TimeDistributedConv2d(512+256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ELU(inplace=True),
            TimeDistributedConv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ELU(inplace=True)
        )
        self.decoding_block2 = nn.Sequential(
            TimeDistributedConv2d(256+128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ELU(inplace=True),
            TimeDistributedConv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ELU(inplace=True)
        )
        self.decoding_block1 = nn.Sequential(
            TimeDistributedConv2d(128+64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ELU(inplace=True)
        )

        self.biCLSTM2 = BiConvLSTM(input_size=(image_size,image_size), input_dim=64, hidden_dim=64, kernel_size=(3,3), num_layers=3, device=self.device)
        
        self.maxpooling = TimeDistributedMaxPool(2, stride=2)
        self.upsampling = TimeDistributedUpsampling(scale_factor=2, mode='nearest')
        
        self.onebyoneConv = nn.Sequential(
            nn.Conv2d(64, n_class, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        ### encoding
        encoded_vol1 = self.encoding_block1(input)
        maxpooled_encoded_vol1 = self.maxpooling(encoded_vol1)
        encoded_vol2 = self.encoding_block2(maxpooled_encoded_vol1)
        maxpooled_encoded_vol2 = self.maxpooling(encoded_vol2)
        encoded_vol3 = self.encoding_block3(maxpooled_encoded_vol2)
        maxpooled_encoded_vol3 = self.maxpooling(encoded_vol3)

        ### lSTM
        lstm_vol1 = self.biCLSTM1(maxpooled_encoded_vol3)

        ### decoding
        up_vol3 = self.upsampling(lstm_vol1)
        concat_vol3 = torch.cat((encoded_vol3, up_vol3), 2)
        decoded_vol3 = self.decoding_block3(concat_vol3)
        up_vol2 = self.upsampling(decoded_vol3)
        concat_vol2 = torch.cat((encoded_vol2, up_vol2), 2)
        decoded_vol2 = self.decoding_block2(concat_vol2)
        up_vol1 = self.upsampling(decoded_vol2)
        concat_vol1 = torch.cat((encoded_vol1, up_vol1), 2)
        decoded_vol1 = self.decoding_block1(concat_vol1)

        ### LSTM
        lstm_vol2 = self.biCLSTM2(decoded_vol1)
        lstm_vol2 = torch.sum(lstm_vol2, 1)

        ### Last PANG!~
        synth_code = self.onebyoneConv(lstm_vol2)

        return synth_code


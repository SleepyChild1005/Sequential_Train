import torch
import torch.nn as nn
from _legacy.model.TimeDistributedLayer import TimeDistributedConv2d, TimeDistributedMaxPool, \
    TimeDistributedConvTranspose2d, TimeDistributedSpatialAttention
from _legacy.model.BiConvLSTM import BiConvLSTM
from _legacy.model.Attention import Attention_block

class SA_Unet_Torch(nn.Module):
    # input_size=(512, 512, 3)?
    # n_class for multiple output (== label numbers)

    def __init__(self, image_size, device, n_class=3, block_size=7, keep_prob=0.9, start_neurons=64):
        # start_neuron = 16 , lr=1e-3
        super(SA_Unet_Torch, self).__init__()
        self.device = device

        self.encode_block1 = nn.Sequential(
            TimeDistributedConv2d(1, start_neurons * 1, kernel_size=3, padding="same", dropout=True),
            nn.ELU(),
            TimeDistributedConv2d(start_neurons * 1, start_neurons * 1, kernel_size=3, padding="same", dropout=True),
            nn.ELU()
        )

        self.encode_block2 = nn.Sequential(
            TimeDistributedMaxPool(2, stride=2),
            TimeDistributedConv2d(start_neurons * 1, start_neurons * 2, kernel_size=3, padding="same", dropout=True),
            nn.ELU(),
            TimeDistributedConv2d(start_neurons * 2, start_neurons * 2, kernel_size=3, padding="same", dropout=True),
            nn.ELU()
        )

        self.encode_block3 = nn.Sequential(
            TimeDistributedMaxPool(2, stride=2),
            TimeDistributedConv2d(start_neurons * 2, start_neurons * 4, kernel_size=3, padding="same", dropout=True),
            nn.ELU(),
            TimeDistributedConv2d(start_neurons * 4, start_neurons * 4, kernel_size=3, padding="same", dropout=True),
            nn.ELU()
        )

        self.biCLSTM1 = BiConvLSTM(input_size=(image_size // (8), image_size // (8)), input_dim=512,
                                   hidden_dim=512, kernel_size=(3, 3), num_layers=3, device=self.device)

        self.spatial_attention = TimeDistributedSpatialAttention()

        self.structured_dropout_convolutional_block_pre = nn.Sequential(
            TimeDistributedMaxPool(2, stride=2),
            TimeDistributedConv2d(start_neurons * 4, start_neurons * 8, kernel_size=3, padding="same", dropout=True),
            nn.ELU()
        )

        self.structured_dropout_convolutional_block_post = nn.Sequential(
            TimeDistributedConv2d(start_neurons * 8, start_neurons * 8, kernel_size=3, padding="same", dropout=True),
            nn.ELU(),
            TimeDistributedConvTranspose2d(start_neurons * 8, start_neurons * 4, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        # uconv3 = concatenate([deconv3, conv3]) <-concat here!

        self.unconv_block3 = nn.Sequential(
            TimeDistributedConv2d(start_neurons * 8, start_neurons * 4, kernel_size=3, padding="same", dropout=True),
            nn.ELU(),
            TimeDistributedConv2d(start_neurons * 4, start_neurons * 4, kernel_size=3, padding="same", dropout=True),
            nn.ELU(),
            TimeDistributedConvTranspose2d(start_neurons * 4, start_neurons * 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

        self.unconv_block2 = nn.Sequential(
            TimeDistributedConv2d(start_neurons * 4, start_neurons * 2, kernel_size=3, padding="same", dropout=True),
            nn.ELU(),
            TimeDistributedConv2d(start_neurons * 2, start_neurons * 2, kernel_size=3, padding="same", dropout=True),
            nn.ELU(),
            TimeDistributedConvTranspose2d(start_neurons * 2, start_neurons * 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        #     uconv1 = concatenate([deconv1, conv1])

        self.biCLSTM2 = BiConvLSTM(input_size=(image_size, image_size), input_dim=64, hidden_dim=64, kernel_size=(3, 3),
                                   num_layers=3, device=self.device)

        self.unconv_block1 = nn.Sequential(
            TimeDistributedConv2d(start_neurons * 2, start_neurons * 1, kernel_size=3, padding="same", dropout=True),
            nn.ELU(),
            TimeDistributedConv2d(start_neurons * 1, start_neurons * 1, kernel_size=3, padding="same", dropout=True),
            nn.ELU()
        )
#######
        self.onebyoneConv = nn.Sequential(
            nn.Conv2d(start_neurons * 1, n_class, kernel_size=1, stride=1, padding="same",bias=False),
            nn.Sigmoid()
        )

        self.Attention1 = Attention_block(F_g=start_neurons * 1, F_l=start_neurons * 1, F_int=start_neurons * 1)
        self.Attention2 = Attention_block(F_g=start_neurons * 2, F_l=start_neurons * 2, F_int=start_neurons * 2)
        self.Attention3 = Attention_block(F_g=start_neurons * 4, F_l=start_neurons * 4, F_int=start_neurons * 4)

    def forward(self, input):
        ### encoding
        conv1 = self.encode_block1(input)     # 1
        conv2 = self.encode_block2(conv1)     # 2
        conv3 = self.encode_block3(conv2)     # 4

        ### decoding
        pre_spat = self.structured_dropout_convolutional_block_pre(conv3)
        # print('pre_spat', pre_spat.shape)
        lstm_vol1 = self.biCLSTM1(pre_spat)
        # print('lstm_vol1',lstm_vol1.shape)
        spat_attention = self.spatial_attention(lstm_vol1)
        # print('spat_attention', spat_attention.shape)
        post_spat = self.structured_dropout_convolutional_block_post(spat_attention)

        # deconv_post = SpatialAttention(deconv_pre)
        # print('post_spat', post_spat.shape)

        # deconv3 = self.structured_dropout_convolutional_block_m(deconv)   # 4
        attention3 = self.Attention3(post_spat, conv3)
        unconv3 = torch.cat((attention3, conv3),2)                       # need to check dim
        # print('unconv3', unconv3.shape)

        deconv2 = self.unconv_block3(unconv3)               # 2
        attention2 = self.Attention2(deconv2, conv2)
        unconv2 = torch.cat((deconv2, conv2),2)           # need to check dim
        # print('unconv2', unconv2.shape)

        deconv1 = self.unconv_block2(unconv2)               # 1
        attention1 = self.Attention1(deconv1, conv1)
        unconv1 = torch.cat((deconv1, conv1),2)
        # print('unconv1', unconv1.shape)

        unconv = self.unconv_block1(unconv1)
        # print('unconv', unconv.shape)

        lstm_vol2 = self.biCLSTM2(unconv)
        lstm_vol2 = torch.sum(lstm_vol2, 1)
        # print('lstm_vol2', lstm_vol2.shape)

        out_tensor = self.onebyoneConv(lstm_vol2)
        # print('out_tensor', out_tensor.shape)
        out_tensor = torch.squeeze(out_tensor, dim=2)
        return out_tensor


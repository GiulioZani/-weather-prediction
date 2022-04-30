import ipdb
import torch
import torch.nn as nn
import torch as t
import torch.nn.functional as f

from main.models.conv.model import GaussianNoise
from .AxialConvLSMT import ConvLSTMCell


class ConvLSTMBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=(3, 3),
        bias=True,
        batch_norm=True,
        dropout=0.2,
        act=nn.LeakyReLU(0.2, inplace=True),
    ) -> None:
        super().__init__()

        self.conv_cell = ConvLSTMCell(
            in_channel, out_channel, kernel_size, bias
        )
        self.norm = nn.BatchNorm2d(out_channel)
        self.dout = nn.Dropout3d(p=dropout)

        self.act = act
        self.batch_norm = batch_norm
        self.dropout = dropout

    def init_hidden(self, x):
        b, _, _, h, w = x.size()
        h_t, c_t = self.conv_cell.init_hidden(batch_size=b, image_size=(h, w))
        return h_t, c_t

    def forward(self, input_tensor, cur_state):

        # h_t, c_t = cur_state
        # ipdb.set_trace()

        h_t, c_t = self.conv_cell(input_tensor, cur_state)
        x = h_t
        if self.batch_norm:
            x = self.norm(h_t)
        if self.dropout:
            x = self.dout(x)
        x = self.act(x)
        return x, c_t


class EncoderDecoderConvLSTM(nn.Module):
    def __init__(self, params, nf=8):
        super(EncoderDecoderConvLSTM, self).__init__()

        self.params = params
        in_chan = 1

        self.conv_lstm_out_chan = 32
        """ ARCHITECTURE 

        # Encoder (ConvLSTM)
        # Encoder Vector (final hidden state of encoder)
        # Decoder (ConvLSTM) - takes Encoder Vector as input
        # Decoder (3D CNN) - produces regression predictions for our model


        """
        self.encoder_1_convlstm = ConvLSTMBlock(
            in_chan, 8, kernel_size=(3, 3), bias=True
        )
        self.encoder_2_convlstm = ConvLSTMBlock(
            8, 16, kernel_size=(3, 3), bias=True
        )
        self.encoder_3_convlstm = ConvLSTMBlock(
            16, 32, kernel_size=(3, 3), bias=True
        )
        self.encoder_4_convlstm = ConvLSTMBlock(
            32, 64, kernel_size=(3, 3), bias=True
        )

        self.decoder_1_convlstm = ConvLSTMBlock(
            64, 64, kernel_size=(3, 3), bias=True
        )
        # self.decoder_2_convlstm = ConvLSTMBlock(
        #     64, 32, kernel_size=(3, 3), bias=True
        # )
        # self.decoder_3_convlstm = ConvLSTMBlock(
        #     32, 16, kernel_size=(3, 3), bias=True
        # )
        # self.decoder_4_convlstm = ConvLSTMBlock(
        #     16, 1, kernel_size=(3, 3), bias=True
        # )

        self.conv_lstms = [
            self.encoder_1_convlstm,
            self.encoder_2_convlstm,
            self.encoder_3_convlstm,
            self.encoder_4_convlstm,
            self.decoder_1_convlstm,
        ]  # self.decoder_2_convlstm, self.decoder_3_convlstm, self.decoder_4_convlstm ]

        self.conv_encoders = [
            self.encoder_1_convlstm,
            self.encoder_2_convlstm,
            self.encoder_3_convlstm,
            self.encoder_4_convlstm,
        ]
        self.conv_decoders = [
            self.decoder_1_convlstm,
        ]  # self.decoder_2_convlstm, self.decoder_3_convlstm, self.decoder_4_convlstm]

        self.decoder_CNN = nn.Sequential(
            nn.Conv3d(
                in_channels=64,
                out_channels=1,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
            )
        )

        self.gaussian_noise = GaussianNoise(0.00001)

    def autoencoder(self, x, seq_len, future_step, h):

        outputs = []

        # encoder
        for t in range(seq_len):

            input_tensor = x[:, t, :, :, :]
            # looping over encoders
            for i in range(len(self.conv_encoders)):
                input_tensor, c = self.conv_encoders[i](
                    input_tensor=input_tensor, cur_state=h[i]
                )
                h[i] = (input_tensor, c)

            # h_t, c_t = self.encoder_1_convlstm(
            #     input_tensor=x[:, t, :, :], cur_state=[h_t, c_t]
            # )  # we could concat to provide skip conn here
            # h_t2, c_t2 = self.encoder_2_convlstm(
            #     input_tensor=h_t, cur_state=[h_t2, c_t2]
            # )  # we could concat to provide skip conn here

            # encoder_vector = h_t4
            # outputs += [h_t4]  # predictions

        # encoder_vector
        encoder_vector = h[len(self.conv_encoders) - 1][0]

        # decoder
        for t in range(future_step):

            input_tensor = encoder_vector
            # looping over decoders
            # ipdb.set_trace()
            for i in range(len(self.conv_decoders)):
                input_tensor, c = self.conv_decoders[i](
                    input_tensor=input_tensor,
                    cur_state=h[i + len(self.conv_encoders)],
                )
                h[i + len(self.conv_encoders)] = (input_tensor, c)

            encoder_vector = h[len(self.conv_encoders)][0]
            outputs += [h[-1][0]]  # predictions

        # ipdb.set_trace()

        outputs = torch.stack(outputs, 1)
        # ipdb.set_trace()
        outputs = outputs.permute(0, 2, 1, 3, 4)
        outputs = self.decoder_CNN(outputs)
        outputs = torch.nn.Sigmoid()(outputs)

        return outputs, h

    def forward(self, x, hidden=None, future_step=None):

        # ipdb.set_trace()

        if future_step == None:
            future_step = self.params.out_seq_len

        x = x.unsqueeze(2)

        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (b, t, c, h, w)        #   batch, time, channel, height, width
        """
        # x = x.unsqueeze(2)
        # ipdb.set_trace()

        # find size of different input dimensions

        x = self.gaussian_noise(x)

        b, seq_len, _, h, w = x.size()

        # ipdb.set_trace()

        # initialize hidden states
        if hidden == None:
            hidden = []
            for i in range(len(self.conv_lstms)):
                state = self.conv_lstms[i].init_hidden(x)
                hidden += [state]

        # autoencoder forward
        outputs, h = self.autoencoder(x, seq_len, future_step, hidden)

        # ipdb.set_trace()

        outputs = outputs.permute(0, 2, 1, 3, 4).squeeze(2)
        # outputs = outputs.squeeze(2)

        # ipdb.set_trace()
        return outputs, h

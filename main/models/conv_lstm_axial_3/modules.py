import ipdb
import torch
import torch.nn as nn
import torch as t
import torch.nn.functional as f

from axial_attention import AxialAttention, AxialPositionalEmbedding

from main.torch_model_modules.components import GaussianNoise
from main.torch_model_modules.ConvLSTM import ConvLSTMCell
from main.torch_model_modules.ConvLSTMModule import ConvLSTMBlock
from main.torch_model_modules.axial import AxialDecoder



class EncoderDecoderConvLSTM(nn.Module):
    def __init__(self, params, nf=8):
        super(EncoderDecoderConvLSTM, self).__init__()

        self.params = params
        in_chan = 1

        # self.conv_lstm_out_chan = 32
        self.embedding_dim = 16
        """ ARCHITECTURE 

        # Encoder (ConvLSTM)
        # Encoder Vector (final hidden state of encoder)
        # Decoder (ConvLSTM) - takes Encoder Vector as input
        # Decoder (3D CNN) - produces regression predictions for our model


        """
        self.encoder_1_convlstm = ConvLSTMBlock(
            in_chan, 8, kernel_size=(3, 3), bias=True, dropout=0.2
        )
        self.encoder_2_convlstm = ConvLSTMBlock(
            8, self.embedding_dim, kernel_size=(3, 3), bias=True, dropout=0.3
        )
        # self.encoder_3_convlstm = ConvLSTMBlock(
        #     8, 16, kernel_size=(3, 3), bias=True, dropout=0.3
        # )
        # self.encoder_4_convlstm = ConvLSTMBlock(
        #     16, 32, kernel_size=(3, 3), bias=True,  dropout=0.1
        # )


        self.decoder_1_convlstm = ConvLSTMBlock(
            16, self.embedding_dim, kernel_size=(3, 3), bias=True, dropout=0.3
        )
        self.decoder_2_convlstm = ConvLSTMBlock(
            self.embedding_dim, self.embedding_dim , kernel_size=(3, 3), bias=True, act=nn.Tanhshrink(), dropout=False
        )
        # self.decoder_3_convlstm = ConvLSTMBlock(
        #     32, 16, kernel_size=(3, 3), bias=True
        # )
        # self.decoder_4_convlstm = ConvLSTMBlock(
        #     16, 1, kernel_size=(3, 3), bias=True
        # )

        self.conv_lstms = [ self.encoder_1_convlstm, self.encoder_2_convlstm, #self.encoder_3_convlstm, #self.encoder_4_convlstm,
         self.decoder_1_convlstm, self.decoder_2_convlstm]# self.decoder_2_convlstm, self.decoder_3_convlstm, self.decoder_4_convlstm ]

        self.conv_encoders = [self.encoder_1_convlstm, self.encoder_2_convlstm, ]# self.encoder_4_convlstm]
        self.conv_decoders = [self.decoder_1_convlstm, self.decoder_2_convlstm ]#self.decoder_2_convlstm, self.decoder_3_convlstm, self.decoder_4_convlstm]

        # self.decoder_CNN = nn.Sequential(
        #     nn.Conv3d(
        #         in_channels=32,
        #         out_channels=1,
        #         kernel_size=(1, 3, 3),
        #         padding=(0, 1, 1),
        #     )
        # )

        # self.decoder_axial = AxialDecoder(params, 16, 8,8)

        self.positional_embedding = AxialPositionalEmbedding(self.embedding_dim, (params.out_seq_len, 4, 5), -1)

        self.out_pos_embedding = AxialPositionalEmbedding(self.embedding_dim, (params.out_seq_len, 4, 5), 2)

        embedding_dim = self.embedding_dim
        self.axial_attention = nn.Sequential(
             AxialAttention(
                dim=embedding_dim,
                dim_index=-1,
                heads=8,
                dim_heads=4,
                num_dimensions=3,
            ),
            nn.GELU(),
             AxialAttention(
                dim=embedding_dim,
                dim_index=-1,
                heads=8,
                dim_heads=4,
                num_dimensions=3,
            ),
            nn.Tanhshrink(),
        )

        self.lstm_axial_attention = nn.Sequential(
             AxialAttention(
                dim=embedding_dim,
                dim_index=2,
                heads=8,
                dim_heads=4,
                num_dimensions=3,
            ),
            nn.GELU(),
             AxialAttention(
                dim=embedding_dim,
                dim_index=2,
                heads=8,
                dim_heads=4,
                num_dimensions=3,
            ),
            nn.Tanhshrink(),
        )

        

        self.decoder = nn.Sequential(
            nn.Linear(self.embedding_dim*2, 1),
            nn.Sigmoid()
            )

        self.encoder = nn.Sequential(
            nn.Linear( 1, self.embedding_dim),
            nn.Tanhshrink(),
        )

        self.noise_variance = 0.001
        self.gaussian_noise = GaussianNoise(self.noise_variance)



   

    def autoencoder(
        self, x, seq_len, future_step, h
    ):

        outputs = []


        # encoder
        for t in range(seq_len):

            input_tensor = x[:, t, :, :, :]
            # looping over encoders
            for i in range(len(self.conv_encoders)):
                input_tensor, c = self.conv_encoders[i](input_tensor=input_tensor, cur_state=h[i])
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
                input_tensor, c = self.conv_decoders[i](input_tensor=input_tensor, cur_state=h[i+len(self.conv_encoders)])
                h[i+len(self.conv_encoders)] = (input_tensor, c)

 
            encoder_vector = h[-1][0]
            outputs += [encoder_vector]  # predictions

        # ipdb.set_trace()

        outputs = torch.stack(outputs, 1)
        # ipdb.set_trace()
        # outputs = outputs.permute(0, 2, 1, 3, 4)
        # outputs = self.decoder_CNN(outputs)
        # outputs = torch.nn.Sigmoid()(outputs)
        # outputs = self.decoder_axial(outputs)

        return outputs, h

    def forward(self, x, hidden = None, future_step = None):

        # ipdb.set_trace()

        if future_step == None:
            future_step = self.params.out_seq_len

        x = x.unsqueeze(2)

        # random noise from channels
        # X shape is (batch_size, seq_len, channels, height, width)
        # x = x + torch.randn(x.shape) * self.noise_variance
        # noise has double the channels as the input
        # noise = torch.randn((x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3], x.shape[4]) , device=x.device) * self.noise_variance
        # # concat channels
        # x = torch.cat([x, noise], dim=2)

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
        outputs, h = self.autoencoder(
            x, seq_len, future_step, hidden
        )

        outputs = nn.Tanhshrink()(outputs)
        outputs = self.out_pos_embedding(outputs)
        outputs = self.lstm_axial_attention(outputs)

        # ipdb.set_trace()     
        x = x.permute(0,1,3,4,2)   
        embedded = self.encoder(x)
        embedded = self.positional_embedding(embedded)
        embedded = self.axial_attention(embedded)

        outputs = outputs.permute(0,1,3,4,2)

        # ipdb.set_trace()
        outputs = self.decoder(t.cat([embedded, outputs], dim=-1))
        # outputs = self.decoder(outputs + embedded)

        # outputs = outputs.permute(0, 2, 1, 3, 4).squeeze(2)
        # outputs = outputs.squeeze(2)
        outputs = outputs.squeeze(-1)

        # ipdb.set_trace()
        return outputs, h



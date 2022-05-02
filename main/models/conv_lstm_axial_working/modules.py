import math
from black import out
import ipdb
import torch
import torch.nn as nn
import torch as t
import torch.nn.functional as f

from main.torch_model_modules.ConvLSTMModule import ConvLSTMBlock
from main.torch_model_modules.components import GaussianNoise
from axial_attention import AxialAttention, AxialPositionalEmbedding


class EncoderDecoderConvLSTM(nn.Module):
    def __init__(self, params, nf=8):
        super(EncoderDecoderConvLSTM, self).__init__()

        self.params = params
        in_chan = 1

        self.conv_lstm_out_chan = 32

        self.z_dim = 16
        """ ARCHITECTURE 

        # Encoder (ConvLSTM)
        # Encoder Vector (final hidden state of encoder)
        # Decoder (ConvLSTM) - takes Encoder Vector as input
        # Decoder (3D CNN) - produces regression predictions for our model

        

        """
        self.act = t.sigmoid

        self.conv_encoders = [
            ConvLSTMBlock(
                in_chan,
                16,
                kernel_size=(3, 3),
                bias=True,
                dropout=False,
                batch_norm=False,
            ),
            ConvLSTMBlock(
                16,
                32,
                kernel_size=(3, 3),
                bias=True,
                dropout=0.1,
                batch_norm=False,
            ),

            # ConvLSTMBlock(
            #     32,
            #     32,
            #     kernel_size=(3, 3),
            #     bias=True,
            #     dropout=0.1,
            #     batch_norm=False,
            # ),
            # ConvLSTMBlock(
            # 16, 16, kernel_size=(3, 3), bias=True, dropout=False
            # ),
        ]

        self.conv_decoders = [
            ConvLSTMBlock(
                32,
                64,
                kernel_size=(3, 3),
                bias=True,
                dropout=0.3,
                batch_norm=False,
            ),
            ConvLSTMBlock(
                64,
                32,
                kernel_size=(3, 3),
                bias=True,
                dropout=False,
                batch_norm=False,
            
            ),
        ]

        self.conv_lstms = self.conv_encoders + self.conv_decoders

        for i in range(len(self.conv_lstms)):
            setattr(self, "conv_lstm_" + str(i), self.conv_lstms[i])

        self.decoder_CNN = nn.Sequential(
            nn.Conv3d(
                in_channels=self.z_dim,
                out_channels=1,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
            )
        )

        self.embedding_dim = 32
        embedding_dim = self.embedding_dim

        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear( 20 * self.embedding_dim * 2, 200),
            nn.LeakyReLU(0.2),
            nn.Linear(200, 100),
            nn.LeakyReLU(0.2),
            nn.Linear(100,1),
            nn.Sigmoid()

        )

        self.embedding_encoder = nn.Sequential(
            nn.Linear(1, embedding_dim, bias=True),
            # nn.Sigmoid()
        )


        self.positional_embedding = AxialPositionalEmbedding(
            embedding_dim, (params.in_seq_len, 4, 5), -1
        )


        self.attentions = nn.Sequential(
            AxialAttention(
                dim=self.embedding_dim,  # embedding dimension
                dim_index=-1,  # where is the embedding dimension
                heads=4,  # number of heads for multi-head attention
                dim_heads=2,
                num_dimensions=3,  # number of axial dimensions (images is 2, video is 3, or more)
            ),
            nn.ReLU(),
            nn.Dropout3d(0.1),     
            #    AxialAttention(
            #     dim=self.embedding_dim,  # embedding dimension
            #     dim_index=-1,  # where is the embedding dimension
            #     heads=4,  # number of heads for multi-head attention
            #     dim_heads=2,
            #     num_dimensions=3,  # number of axial dimensions (images is 2, video is 3, or more)
            # ),
            # nn.ReLU(),
            # nn.Dropout3d(0.1),         
        )

    

        self.noise = 0.0001

        self.gaussian_noise = GaussianNoise(self.noise)

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

            encoder_vector = h[-1][0]
            outputs += [h[-1][0]]  # predictions

     
        outputs= torch.stack(outputs, 1)
    
    
        return outputs, h

    def forward(self, x, hidden = None, future_step=1):

        # ipdb.set_trace()



        if self.training:
            x = self.gaussian_noise(x)


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

        # noise vector adding to channels

        # x = self.gaussian_noise(x)

        b, seq_len, _, h, w = x.size()

        # ipdb.set_trace()

        # initialize hidden states
        if hidden == None:
            hidden = []
            for i in range(len(self.conv_lstms)):
                state = self.conv_lstms[i].init_hidden(x)
                hidden += [state]

        # autoencoder forward
        outputs, hidden = self.autoencoder(x, seq_len, 1, hidden)


        x = x.permute(0,1,3,4,2)   
        embedded = self.embedding_encoder(x)
        embedded = self.positional_embedding(embedded)
        embedded = self.attentions(embedded)
        embedded = embedded.max(1)[0]

        # ipdb.set_trace()
        outputs = outputs.max(1)[0]
        outputs = outputs.squeeze(1)
        outputs = outputs.permute(0,2,3,1)

        # final = outputs + embedded
        # ipdb.set_trace()
        # outputs = outputs.flatten(1)
        # embedded = embedded.flatten(1)
        final = t.cat([outputs, embedded], -1)

        outputs = self.decoder(final)

        # ipdb.set_trace()

        # outputs = outputs.permute(0, 2, 1, 3, 4).squeeze(2)
        # outputs = outputs.squeeze(2)

        # ipdb.set_trace()
        return outputs, hidden



    def get_n_future_steps(self, x, future_step):

        seq = future_step
        future_steps = future_step
        future_steps = math.ceil(future_steps / x.shape[1])
        outs = []

        for i in range(future_steps):
            out = self(x)
            x = out
            outs += [out]

        # ipdb.set_trace()
        out = t.stack(outs, dim=1)
        out = out.view(
            x.shape[0], int(future_steps * x.shape[1]), x.shape[2], x.shape[3]
        )
        out = out[:, -seq:, :, :]

        return out    
import math
from turtle import forward
import torch as t
import torch.nn as nn
from axial_attention import AxialAttention, AxialPositionalEmbedding
import ipdb

from main.torch_model_modules.components import GaussianNoise


# Encoder decoder that transforms numbers between 0 and 1 to embeddings
class EncoderDecoderEmbeddings(nn.Module):
    def __init__(self, params, embedding_dim=32):
        super().__init__()
        self.params = params

        self.embedding_dim = embedding_dim
        # embedding encoder
        self.embedding_encoder = nn.Sequential(
            nn.Linear(1, embedding_dim, bias=True),
            # nn.Sigmoid()
        )

        # decoding the embedding to the output
        self.embedding_decoder = nn.Sequential(
            nn.Linear(embedding_dim, 1, bias=True), nn.Sigmoid()
        )

        # self.positional_embedding = AxialPositionalEmbedding(
        #     embedding_dim, (params.in_seq_len, 4, 5), -1
        # )

        # self.transpose_pos_embedding = AxialPositionalEmbedding(
        #     embedding_dim, (params.in_seq_len, 5, 4 ), -1)
        # self.embeding = nn.Embedding(5000, 256)
        self.tanh = nn.Tanhshrink()

        self.attentions = nn.Sequential(
            AxialAttention(
                dim=self.embedding_dim,  # embedding dimension
                dim_index=-1,  # where is the embedding dimension
                heads=4,  # number of heads for multi-head attention
                dim_heads=2,
                num_dimensions=3,  # number of axial dimensions (images is 2, video is 3, or more)
            ),
            nn.ReLU(),
            # nn.Dropout3d(0.15),
            # nn.Dropout3d(0.7),
            # AxialAttention(
            #     dim=self.embedding_dim,  # embedding dimension
            #     dim_index=-1,  # where is the embedding dimension
            #     heads=4,  # number of heads for multi-head attention,
            #     dim_heads=2,
            #     num_dimensions=3,  # number of axial dimensions (images is 2, video is 3, or more)
            # ),
            # nn.ReLU(),

            # nn.Dropout3d(0.3),
        )



        # self.decoder_CNN = nn.Sequential(
        #     nn.Conv3d(
        #         in_channels=self.params.in_seq_len,
        #         out_channels=1,
        #         kernel_size=(1, 3, 3),
        #         padding=(0, 1, 1),
        #     ),
        #     nn.LeakyReLU(0.2),
        # )

        self.decoder = nn.Sequential(
            nn.Linear( self.params.in_seq_len * 20 * self.embedding_dim, 100),
            nn.LeakyReLU(0.2),
            nn.Linear(100, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()

        )

        self.noise_layer = GaussianNoise(0.001)

    
    def forward(self, x: t.Tensor, future_step=None):
        x = self.net(x)
        x = self.decoder(x.view(x.shape[0], -1))
        # ipdb.set_trace()
        return x

    # def forward(self, x: t.Tensor, future_step=1) -> t.Tensor:
    #     context = x
    #     for _ in range(future_step):
    #         input = context[:, -self.params.in_seq_len :]
    #         next_step = self.net(input).view(x.shape[0], 1, 4, 5)
    #         context = t.cat((context, next_step), dim=1)
    #     return context[:, -future_step:]


    def net(self, x):


        x = x.unsqueeze(-1)
        


  
  
        if self.training:
  
            x = self.noise_layer(x)
        else:
            x = self.noise_layer(x,0.0001)

  
  
  
        x = self.embedding_encoder(x)
  
        x = self.attentions(x)

        # ipdb.set_trace()
        # x = self.embedding_decoder(x) 

        # x = self.decoder_CNN(x)
        # x = x.squeeze(1)
        # x = x.squeeze(-1)
     
        return x

    def get_n_future_steps(self, x, future_steps):
        
        seq = future_steps
        future_steps =math.ceil(future_steps / x.shape[1])
        outs = []

        for i in range(future_steps):
            out = self(x)
            x = out
            outs.append(out)

        out = t.stack(outs, dim=1)
        out = out.view(x.shape[0], int(future_steps * x.shape[1]), x.shape[2], x.shape[3])  
        out = out[:, -seq :, :, :]

        return out

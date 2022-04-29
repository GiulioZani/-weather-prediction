from ipaddress import ip_address
import torch as t
import torch.nn as nn
from axial_attention import AxialAttention, AxialPositionalEmbedding
import ipdb

from main.models.conv.model import GaussianNoise

class AxialDecoder(nn.Module):
    def __init__(self, params, embedding_dim, heads=16, dim_heads=16) -> None:
        super().__init__()

        self.embedding_decoder = nn.Linear(embedding_dim, 1, bias=True)
        self.act = nn.Sigmoid()

        self.positional_embedding = AxialPositionalEmbedding(embedding_dim, (params.out_seq_len, 4, 5), 2)

        self.attentions = nn.Sequential(
            AxialAttention(
                dim=embedding_dim,
                dim_index=2,
                heads=16,
                dim_heads=16,
                num_dimensions=3,
            ),
                # nn.BatchNorm2d(embedding_dim),
                # nn.Dropout3d(0.1),
            # nn.LeakyReLU(0.2, inplace=True),
            AxialAttention(
                dim=embedding_dim,
                dim_index=2,
                heads=16,
                dim_heads=16,
                num_dimensions=3,
            ),
                # nn.LeakyReLU(0.2, inplace=True),

            )

    
    def forward(self, x):

        # shape of x is (batch_size, seq_len, embedding_dim, w,h)

        x = self.positional_embedding(x)
        x = self.attentions(x)
        x = x.permute(0, 1, 3, 4, 2)
        x = self.embedding_decoder(x)
        x = self.act(x)
        return x


# Encoder decoder that transforms numbers between 0 and 1 to embeddings
class EncoderDecoderEmbeddings(nn.Module):
    def __init__(self, params, embedding_dim=2, embedding_vocab_size=1024):
        super().__init__()
        
        
        # embedding encoder
        self.embedding_encoder = nn.Sequential(
            nn.Linear(2, 256, bias=False),
            # nn.LeakyReLU(0.2, inplace=True)

        )

        # decoding the embedding to the output
        self.embedding_decoder = nn.Sequential(
            nn.Linear(256, 1, bias=True),
            nn.Sigmoid()
        )

        self.positional_embedding = AxialPositionalEmbedding(256, (8, 4, 5), -1)
        # self.embeding = nn.Embedding(5000, 256)
        self.tanh = nn.Tanh()


        self.attentions = nn.Sequential(
            AxialAttention(
                dim = 256 ,           # embedding dimension
                dim_index = -1,       # where is the embedding dimension
                heads = 16,           # number of heads for multi-head attention
                dim_heads=16,
                num_dimensions = 3,  # number of axial dimensions (images is 2, video is 3, or more)
            ),
            nn.LeakyReLU(0.2, inplace=True),
           
                AxialAttention(
                dim = 256 ,           # embedding dimension
                dim_index = -1,       # where is the embedding dimension
                heads = 16,           # number of heads for multi-head attention,
                dim_heads=16,
                num_dimensions = 3,  # number of axial dimensions (images is 2, video is 3, or more)
            )
            ,
            nn.LeakyReLU(0.2, inplace=True),
 
        )
 

        self.noise_layer = GaussianNoise(0.0001)


    
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.noise_layer(x)

        # random noise vector
        # ipdb.set_trace()
        rand = t.rand_like(x)
        x =t.cat([x, rand], dim=-1)

        # ipdb.set_trace()
        x = self.embedding_encoder(x)
        x = self.tanh(x)

        x = self.positional_embedding(x)
        # x = x.squeeze(-2)

        # x = self.act(x)
        x = self.attentions(x)
        x = self.embedding_decoder(x)

        # ipdb.set_trace()
        return x.squeeze(-1)
        
        
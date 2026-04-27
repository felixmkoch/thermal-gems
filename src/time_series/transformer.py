from torch import nn
from torch.nn import TransformerEncoder
from torch.nn import TransformerEncoderLayer
import torch
import math


class Transformer(nn.Module):

    def __init__(self,
                 d_model: int = 256,
                 n_head: int = 8,
                 num_layers: int = 8,
                 hidden_size: int = 1024,
                 num_features:int = 8,
                 forecast_horizon: int = 1,
                 dropout: float = 0.1,
                 activation_function: nn.Module = "relu",
                 layer_norm_epsilon: float = 1e-05,
                 seq_len: int = 96,
                 device: str = "cpu",
                 dtype: str = None,
                 num_targets: int = 1,
                 **kwargs
                 ):
        
        super().__init__()

        self.d_model = d_model
        self.n_head = n_head
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_features = num_features
        self.dropout = dropout
        self.activation_function = activation_function
        self.layer_norm_epsilon = layer_norm_epsilon
        self.device = device
        self.dtype = dtype
        self.forecast_horizon = forecast_horizon
        self.num_targets = num_targets

        self.encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=hidden_size,
            dropout=dropout,
            activation=activation_function,
            layer_norm_eps=layer_norm_epsilon,
            batch_first=True,
            norm_first=False,
            bias=True,
            device=device,
            dtype=dtype
        )

        self.transformer_encoder = TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=num_layers
        )

        self.positional_encoding = PositionalEncoding(d_model=d_model, seq_len=seq_len)

        self.encoder = nn.Linear(num_features, d_model)

        self.decoder = nn.Linear(d_model, forecast_horizon * num_targets)


    def forward(self, x, src_mask=None):

        x = self.encoder(x)

        x = self.positional_encoding(x)

        x = self.transformer_encoder(x, src_mask)

        output = self.decoder(x[:, -1, :])
        output = output.view(-1, self.forecast_horizon, self.num_targets)  # Reshape to (batch_size, forecast_horizon, num_targets)
        return output


class PositionalEncoding(nn.Module):
  def __init__(self, d_model: int, seq_len: int = 96):

    super().__init__()     
    
    pe = torch.zeros(seq_len, d_model)    

    k = torch.arange(0, seq_len).unsqueeze(1)  

    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))  # Encoding Factor.

    # Part from attention is all you need. Sin on even and cosine on odds.
    pe[:, 0::2] = torch.sin(k * div_term)    
    pe[:, 1::2] = torch.cos(k * div_term)  
  
    pe = pe.unsqueeze(0)                # Add batch dimension in front.          

    self.register_buffer("pe", pe)      # Causes the optimizer not to optimize that part.        

  def forward(self, x: torch.Tensor):

    x = x + self.pe[:, : x.size(1)]  # Add positional encoding to the tensor of size [batch_size, seq_len, num_features]

    return x
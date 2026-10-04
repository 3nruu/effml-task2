import math

import torch
from torch import nn, Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        pos = torch.arange(max_len,dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float()* (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)

        pe[:, 0::2] = torch.sin(pos * div_term)

        pe[:, 1::2] = torch.cos(pos * div_term[:pe[:, 1::2].shape[1]])

        pe = pe.unsqueeze(1)

        self.pe = nn.Parameter(pe, requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size[0]]

        return self.dropout(x)

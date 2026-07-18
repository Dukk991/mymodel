import torch
import math
from torch import nn

class SelfAttention(nn.Module):
    
    def __init__(self, dim, max_length, bias=True):
        super().__init__()

        self.dim = dim

        self.kqv = nn.Linear(dim, 3*dim, bias=bias)

        self.register_buffer("mask", torch.tril(torch.ones(max_length, max_length)))

    def forward(self, x):
        _, T, _ = x.size() #batch size, sequence length, embed dim

        q, k, v = self.kqv(x).split(self.dim, dim=2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dim)
        
        att = att.masked_fill(self.mask[:T, :T] == 0, float("-inf"))

        att = torch.softmax(att, dim=-1)

        y = att @ v
        return y
import torch.nn as nn

class PositionalEmbedding(nn.Module):
    def __init__(self, max_lenght, dim):
        super().__init__()

        self.embedding = nn.Embedding(max_lenght, dim)
    
    def forward(self, positions):
        return self.embedding(positions)
# sinusoidal positional embedding
import torch
import math

def positional_embedding(max_len, dim):
    """Calculate postional embedding vector.
    Args:
        max_len: maximum length of sentence
        dim: dimension of each embedding vector
    """
    dimensions = torch.arange(0, dim, 2).float()

    positions = torch.arange(max_len).unsqueeze(1).float() 

    div_term = torch.exp(dimensions* -(math.log(10000.0) / dim))

    PE = torch.zeros(max_len, dim, requires_grad=False)

    PE[:, 0::2] = torch.sin(positions * div_term)

    PE[:, 1::2] = torch.cos(positions * div_term)

    return PE


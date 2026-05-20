import torch.nn as nn
import torch.nn.functional as F
import torch

class CausalSelfAttention(nn.Module):
    def __init__(self, dim_model, num_heads):
        super().__init__()
        self.dim_head = dim_model // num_heads
        self.W_q = nn.Parameter(torch.empty((num_heads, dim_model, self.dim_head)))
        self.W_k = nn.Parameter(torch.empty((num_heads, dim_model, self.dim_head)))
        self.W_v = nn.Parameter(torch.empty((num_heads, dim_model, self.dim_head)))
        self.W_o = nn.Parameter(torch.empty((num_heads, self.dim_head, dim_model)))
        self.b_q = nn.Parameter(torch.zeros((num_heads, self.dim_head)))
        self.b_k = nn.Parameter(torch.zeros((num_heads, self.dim_head)))
        self.b_v = nn.Parameter(torch.zeros((num_heads, self.dim_head)))
        self.b_o = nn.Parameter(torch.zeros((dim_model,)))

    def forward(self, x):
        q = torch.einsum(
            "n_head d_model d_head, batch seq d_model -> batch seq n_head d_head",
            self.W_q, x
        ) + self.b_q[None, None, :, :]

        k = torch.einsum(
            "n_head d_model d_head, batch seq d_model -> batch seq n_head d_head",
            self.W_k, x
        ) + self.b_k[None, None, :, :]

        v = torch.einsum(
            "n_head d_model d_head, batch seq d_model -> batch seq n_head d_head",
            self.W_v, x
        ) + self.b_v[None, None, :, :]

        mask = torch.ones(
            (q.size(1), k.size(1))
        ).triu(diagonal=1).bool()

        z = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=True
        )

        return torch.einsum(
            "n_head d_head d_model, batch seq n_head d_head -> batch seq d_model",
            self.W_o, z
        ) + self.b_o[None, None, :]

class MLP(nn.Module):
    def __init__(self, dim_model, dim_mlp):
        super().__init__()
        self.fc = nn.Linear(dim_model, dim_mlp)
        self.proj = nn.Linear(dim_mlp, dim_model)

    def forward(self, x):
        x = F.relu(self.fc(x))
        return self.proj(x.square())

class TransformerBlock(nn.Module):
    def __init__(self, dim_model, dim_mlp, num_heads):
        super().__init__()
        self.attn = CausalSelfAttention(dim_model, num_heads)
        self.ln1 = nn.LayerNorm(dim_model)
        self.mlp = MLP(dim_model, dim_mlp)
        self.ln2 = nn.LayerNorm(dim_model)

    def forward(self, x):
        h = self.attn(self.ln1(x))
        x = h + x
        h = self.mlp(self.ln2(x))
        return h + x

class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim_model: int,
        dim_mlp: int,
        num_heads: int,
        num_layers: int,
        tie_embeddings: bool,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.tie_embeddings = tie_embeddings

        self.token_embedding = nn.Embedding(vocab_size, dim_model)

    def forward(self, x):
        pass

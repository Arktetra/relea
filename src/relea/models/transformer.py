import torch.nn as nn
import torch.nn.functional as F
import torch

class Rotary(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freqs = 1. / (base ** torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        self.register_buffer("inv_freqs", inv_freqs, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: torch.Tensor | None = None
        self._sin_cached: torch.Tensor | None = None
    
    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if (
            self._seq_len_cached != seq_len
            or self._cos_cached is None
            or self._sin_cached is None
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freqs.dtype)
            freqs = torch.outer(t, self.inv_freqs)
            self._cos_cached = freqs[None, None, :, :].cos()
            self._sin_cached = freqs[None, None, :, :].sin()
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)
    
def apply_rotary_emb(x, cos, sin):
    half = x.size(-1) // 2
    x1, x2 = x[:, :half], x[:, half:]
    return torch.cat((x1 * cos + x2 * sin), (x1 * (-sin) + x2 * cos), dim=-1)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim_model: int, num_heads: int, rotary_base: int):
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
        
        self.rotary = Rotary(self.dim_head, rotary_base)

    def forward(self, x):
        q = torch.einsum(
            "n_head d_model d_head, batch seq d_model -> batch seq n_head d_head",
            self.W_q, x
        ) + self.b_q[None, None, :, :]

        k = torch.einsum(
            "n_head d_model d_head, batch seq d_model -> batch seq n_head d_head",
            self.W_k, x
        ) + self.b_k[None, None, :, :]

        cos, sin = self.rotary(q.size(1), q.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(q, cos, sin)

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
        self.blocks = nn.ModuleList([
            TransformerBlock(dim_model, dim_mlp, num_heads)
            for _ in range(num_layers)
        ])

        if not tie_embeddings:
            self.unembed = nn.Linear(dim_model, vocab_size)

    def forward(self, x):
        h = self.token_embedding[x]

        for block in self.blocks:
            h = block(h)
        h = F.layer_norm(h, h.size())

        if self.tie_embeddings:
            h = torch.einsum(
                "batch seq dim_model, vocab_size dim_model -> batch seq vocab_size",
                h, self.token_embedding
            )
        else:
            h = self.unembed(h)

        return F.softmax(h, dim=-1)


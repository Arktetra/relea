import torch
import torch.nn as nn
import torch.nn.functional as F

from relea.models.transformer import Rotary, apply_rotary_emb

class Qwen3RMSNorm(nn.Module):
    def __init__(self, dim_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones((dim_model,)))
        self.var_eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.to(torch.float32)
        var = x.square().mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.var_eps)
        return self.weight * x.to(dtype)

class Qwen3MLP(nn.Module):
    def __init__(
        self,
        dim_model: int,
        dim_mlp: int,
    ):
        self.up_proj = nn.Linear(dim_model, dim_mlp)
        self.gate_proj = nn.Linear(dim_model, dim_mlp)
        self.down_proj = nn.Linear(dim_mlp, dim_model)
        
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class Qwen3Attention(nn.Module):
    def __init__(
        self, 
        dim_model: int, 
        num_heads: int, 
        num_kv_heads: int,
        dropout: float,
        rope_base: int,
        rms_norm_eps: 1e-6,
    ):
        super().__init__()
        self.dim_model = dim_model
        self.dim_head = dim_model // num_heads
        self.num_heads = num_heads
        self.rope_base = rope_base
        self.num_kv_groups = num_heads // num_kv_heads
        self.dropout = dropout

        self.W_q = nn.Parameter(torch.empty((num_heads, dim_model, self.dim_head)))
        self.W_k = nn.Parameter(torch.empty((num_kv_heads, dim_model, self.dim_head)))
        self.W_v = nn.Parameter(torch.empty((num_kv_heads, dim_model, self.dim_head)))
        self.W_o = nn.Parameter(torch.empty((num_heads, self.dim_head, dim_model)))
        self.b_q = nn.Parameter(torch.zeros((num_heads, self.dim_head)))
        self.b_k = nn.Parameter(torch.zeros((num_heads, self.dim_head)))
        self.b_v = nn.Parameter(torch.zeros((num_heads, self.dim_head)))
        self.b_o = nn.Parameter(torch.zeros((dim_model,)))

        self.q_norm = Qwen3RMSNorm(self.dim_head, eps=rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.dim_head, eps=rms_norm_eps)

        self.rotary = Rotary(self.dim_head, base=rope_base)
    
    def forward(self, x):
        q = torch.einsum(
            "n_head d_model d_head, batch seq d_model -> batch seq n_head d_head",
            self.W_q, x
        ) + self.b_q[None, None, :, :]

        k = torch.einsum(
            "n_kv_head d_model d_head, batch seq d_model -> batch seq n_kv_head d_head",
            self.W_k, x
        ) + self.b_k[None, None, :, :]

        v = torch.einsum(
            "n_kv_head d_model d_head, batch seq d_model -> batch seq n_kv_head d_head",
            self.W_v, x
        ) + self.b_v[None, None, :, :]

        cos, sin = self.rotary(q.size(1), q.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(q, cos, sin)

        k = self._repeat_kv(k, self.num_kv_groups)
        v = self._repeat_kv(v, self.num_kv_groups)

        mask = torch.ones((q.size(1), k.size(1))).triu(diagonal=1).bool()

        z = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=True
        )

        return torch.einsum(
            "n_head d_head d_model, batch seq n_head d_head -> batch seq d_model",
            self.W_o, z
        ) + self.b_o[None, None, :]

    def _repeat_kv(x: torch.Tensor, n_rep: int):
        """
        This is used for repeating the key and value vectors. It repeats these
        vectors to convert their shape from (batch num_kv_heads, seq, dim_head)
        to (batch num_heads, seq, dim_head).
        """
        batch, num_kv_heads, seq, dim_head = x.shape
        if n_rep == 1:
            return x
        
        x = x[:, :, None, :, :].expand([batch, num_kv_heads, n_rep, seq, dim_head])
        return x.reshape((batch, num_kv_heads * n_rep, seq, dim_head))

class Qwen3TransformerBlock(nn.Module):
    def __init__(
        self,
        dim_model: int,
        dim_mlp: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: int,
        dropout: float,
        rms_norm_eps: 1e-6
    ):
        super().__init__()
        self.norm1 = Qwen3RMSNorm(dim_model)
        self.attn = Qwen3Attention(
            dim_model, 
            num_heads, 
            num_kv_heads, 
            dropout, 
            rope_base, 
            rms_norm_eps
        )
        self.norm2 = Qwen3RMSNorm(dim_model)
        self.mlp = Qwen3MLP(dim_model, dim_mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.attn(h)
        x = x + h
        h = self.norm2(x)
        return self.mlp(h)

class Qwen3Model(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim_model: int,
        dim_mlp: int,
        num_heads: int,
        num_kv_heads: int,
        num_layers: int,
        rope_base: int,
        dropout: float,
        rms_norm_eps: float = 1e-6,
        tie_embeddings: bool = True
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim_model)
        self.blocks = nn.ModuleList([
            Qwen3TransformerBlock(
                dim_model,
                dim_mlp,
                num_heads,
                num_kv_heads,
                rope_base,
                dropout,
                rms_norm_eps
            ) for _ in range(num_layers)
        ])
        self.norm = Qwen3RMSNorm(dim_model)

        self.lm_head = nn.Linear(dim_model, vocab_size) if tie_embeddings else None

    def forward(self, x):
        h = self.token_embedding[x]
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)

        if self.lm_head is not None:
            h = h @ self.token_embedding.T
        else:
            h = self.lm_head(h)
        
        return h
        
from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from relea.configs.dinov3 import DINOV3ViTConfig

@torch.inference_mode()
def get_patch_coords(num_patches_w, num_patches_h, dtype, device):
    # Create coordinates
    coords_x = torch.arange(0.5, num_patches_w, dtype=dtype, device=device)
    coords_y = torch.arange(0.5, num_patches_h, dtype=dtype, device=device)

    # Normalize coordinates to [0, 1]
    coords_x = coords_x / num_patches_w
    coords_y = coords_y / num_patches_h

    coords_X, coords_Y = torch.meshgrid((coords_x, coords_y), indexing="ij")
    coords = torch.stack((coords_X, coords_Y), dim=-1)

    # Normalize Coordinates to [-1, 1]
    coords = 2 * coords - 1

    # [num_patches_w, num_patches_h, 2] -> [num_patches, 2]
    coords = coords.flatten(0, 1)
    return coords

@torch.inference_mode()
def augment_patch_coords(
    coords: torch.Tensor, 
    shift: Optional[float] = None, 
    jitter: Optional[float] = None, 
    scale: Optional[float] = None
):
    dtype, device = coords.dtype, coords.device

    if shift is not None:
        shift_hw = torch.empty(1, 2, dtype=dtype, device=device)
        coords = coords + shift_hw.uniform_(-shift, shift)

    if jitter is not None:
        jitter = torch.log(torch.tensor(jitter))
        jitter_hw = torch.empty(1, 2, dtype=dtype, device=device)
        coords = coords * jitter_hw.uniform_(-jitter, jitter).exp()

    if scale is not None:
        scale = torch.log(torch.tensor(scale))
        scale_hw = torch.empty(1, dtype=dtype, device=device)
        coords = coords * scale_hw.uniform_(-scale, scale).exp()

    return coords

class DINOV3ViTEmbeddings(nn.Module):
    def __init__(
        self,
        num_channels: int,
        hidden_size: int,
        patch_size: int,
        num_register_tokens: int
    ):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.mask_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.register_tokens = nn.Parameter(torch.randn(1, num_register_tokens, hidden_size))
        self.patch_embeddings = nn.Conv2d(
            in_channels=num_channels,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            # padding=1
        )

    @staticmethod
    def from_config(config: DINOV3ViTConfig):
        return DINOV3ViTEmbeddings(
            num_channels=config.num_channels,
            patch_size=config.patch_size,
            hidden_size=config.hidden_size,
            num_register_tokens=config.num_register_tokens,
        )

    def forward(self, x: torch.Tensor, bool_mask_pos: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = x.shape[0]
        target_dtype = self.patch_embeddings.weight.dtype

        # [batch, channel, height, width] -> [batch, hidden_size, height // patch_size, width // patch_size]
        patch_embeddings = self.patch_embeddings(x.to(dtype=target_dtype))
        # [batch, hidden_size, height // patch_size, width // patch_size] -> [batch, num_patches, hidden_size]
        patch_embeddings = patch_embeddings.flatten(start_dim=2).transpose(1, 2)

        if bool_mask_pos is not None:
            self.mask_token = self.mask_token.to(dtype=target_dtype)
            patch_embeddings = torch.where(bool_mask_pos, self.mask_token, patch_embeddings)

        cls_token = self.cls_token.expand(batch_size, -1, -1)
        register_tokens = self.register_tokens.expand(batch_size, -1, -1)

        return torch.cat([cls_token, register_tokens, patch_embeddings], dim=1)

class DINOV3ViTRopeEmbeddings(nn.Module):
    def __init__(
        self, 
        base: float,
        num_attention_heads: int,
        hidden_size: int,
        patch_size: int,
        pos_embed_shift: float,
        pos_embed_jitter: float,
        pos_embed_scale: float,
    ):
        super().__init__()
        base = base
        dim_head = hidden_size // num_attention_heads
        inv_freqs = 1. / (base ** (torch.arange(0, 1, 4 / dim_head)))
        self.register_buffer("inv_freqs", inv_freqs, persistent=False)
        self.patch_size = patch_size
        self.shift = pos_embed_shift
        self.jitter = pos_embed_jitter
        self.scale = pos_embed_scale

    @staticmethod
    def from_config(config: DINOV3ViTConfig):
        DINOV3ViTRopeEmbeddings(
            base=config.rope_theta,
            num_attention_heads=config.num_attention_heads,
            hidden_size=config.hidden_size,
            patch_size=config.patch_size,
            pos_embed_shift=config.pos_embed_shift,
            jitter=config.pos_embed_jitter,
            scale=config.pos_embed_rescale
        )

    def forward(self, x):
        _, _, H, W = x.shape
        num_patches_w = W // self.patch_size
        num_patches_h = H // self.patch_size
        dtype = x.dtype
        device = x.device

        patch_coords = get_patch_coords(num_patches_w, num_patches_h, dtype, device)

        if self.training:
            patch_coords = augment_patch_coords(
                patch_coords,
                shift=self.shift,
                jitter=self.jitter,
                scale=self.scale
            )

        # [num_patches, 2, 1] * [1, 1, d_head / 4] -> [num_patches, 2, d_head / 4]
        angles = 2 * math.pi * patch_coords[:, :, None] * self.inv_freqs[None, None, :]
        # [num_patches, 2, d_head / 4] -> [num_patches, d_head / 2]
        angles = angles.flatten(1)
        # [num_patches, d_head / 2] -> [num_patches, d_head]
        angles = angles.tile(2)

        return angles.cos(), angles.sin()

def rotate_half(x: torch.Tensor):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

def apply_pos_embeds(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
):
    num_tokens = q.shape[-2]
    num_patches = cos.shape[-2]
    num_prefix_tokens = num_tokens - num_patches

    q_patches = q[:, :, num_prefix_tokens:, :]
    k_patches = k[:, :, num_prefix_tokens:, :]

    q_patches = q_patches * cos + rotate_half(q_patches) * sin
    k_patches = k_patches * cos + rotate_half(k_patches) * sin

    q[:, :, num_prefix_tokens:, :] = q_patches
    k[:, :, num_prefix_tokens:, :] = k_patches

    return q, k
    

class DINOV3ViTAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        query_bias: bool,
        key_bias: bool,
        value_bias: bool,
        proj_bias: bool,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_head = num_attention_heads
        self.d_head = hidden_size // num_attention_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=query_bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=key_bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=value_bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=proj_bias)
        self.dropout = dropout
        self.scale = self.d_head ** -0.5

    @staticmethod
    def from_config(config: DINOV3ViTConfig):
        return DINOV3ViTAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            query_bias=config.query_bias,
            key_bias=config.key_bias,
            value_bias=config.value_bias,
            proj_bias=config.proj_bias
        )

    def forward(
        self, 
        x: torch.Tensor,
        pos_embeds: Tuple[torch.Tensor, torch.Tensor],
        attn_mask: Optional[torch.Tensor] = None,
    ):
        B, N, _ = x.shape   # [batch, num_tokens, hidden_size]

        q = self.q_proj(x)  # [batch, num_tokens, hidden_size]     
        k = self.k_proj(x)  # [batch, num_tokens, hidden_size]
        v = self.v_proj(x)  # [batch, num_tokens, hidden_size]

        q = q.view(B, N, self.n_head, self.d_head).transpose(1, 2)  # [batch, num_heads, num_tokens, dim_head]
        k = k.view(B, N, self.n_head, self.d_head).transpose(1, 2)  # [batch, num_heads, num_tokens, dim_head]
        v = v.view(B, N, self.n_head, self.d_head).transpose(1, 2)  # [batch, num_heads, num_tokens, dim_head]

        cos, sin = pos_embeds   # [num_patches, dim_head], [num_patches, dim_head]
        q, k = apply_pos_embeds(q, k, cos, sin)

        z =  F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=0.0 if not self.training else self.dropout,
            scale=self.scale
        )   # [batch, num_heads, num_tokens, dim_head]
        
        z = z.transpose(1, 2)  #[batch, num_heads, num_tokens, dim_head] -> [batch, num_tokens, num_heads, dim_head]

        return self.o_proj(z.reshape(B, N, -1).contiguous())

class DINOV3ViTMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        mlp_bias: bool,
    ):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=mlp_bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=mlp_bias)

    def forward(self, x):
        return self.down_proj(F.gelu(self.up_proj(x)))

class DINOV3GatedMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        mlp_bias: bool,
    ):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=mlp_bias)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=mlp_bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=mlp_bias)

    def forward(self, x):
        return self.down_proj(F.gelu(self.gate_proj(x)) * self.up_proj(x))

class DINOV3DropPath(nn.Module):
    def __init__(
        self,
        drop_prob: float,
    ):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor):
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0], ) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x.div(keep_prob) * random_tensor


class DINOV3LayerScale(nn.Module):
    def __init__(self, hidden_size, layer_scale_value):
        super().__init__()
        self.lambda1 = nn.Parameter(layer_scale_value * torch.ones(hidden_size))

    def forward(self, x: torch.Tensor):
        return self.lambda1 * x

class DINOV3ViTLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        query_bias: bool,
        key_bias: bool,
        value_bias: bool,
        proj_bias: bool,
        mlp_bias: bool,
        layer_norm_eps: float,
        layer_scale_value: float,
        drop_path_rate: float,
        use_gated_mlp: bool,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attention = DINOV3ViTAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            query_bias=query_bias,
            key_bias=key_bias,
            value_bias=value_bias,
            proj_bias=proj_bias,
            dropout=dropout,
        )
        self.layer_scale1 = DINOV3LayerScale(
            hidden_size, layer_scale_value
        )
        self.drop_path = DINOV3DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.layer_scale2 = DINOV3LayerScale(
            hidden_size, layer_scale_value
        )
        if use_gated_mlp:
            self.mlp = DINOV3GatedMLP(hidden_size, intermediate_size, mlp_bias)
        else:
            self.mlp = DINOV3ViTMLP(hidden_size, intermediate_size, mlp_bias)

    @staticmethod
    def from_config(config: DINOV3ViTConfig):
        return DINOV3ViTLayer(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            query_bias=config.query_bias,
            key_bias=config.key_bias,
            value_bias=config.value_bias,
            proj_bias=config.proj_bias,
            mlp_bias=config.mlp_bias,
            layer_norm_eps=config.layer_norm_eps,
            layer_scale_value=config.layerscale_value,
            drop_path_rate=config.drop_path_rate,
            use_gated_mlp=config.use_gated_mlp,
            dropout=config.attention_dropout,
        )

    def forward(
        self, 
        x: torch.Tensor,
        pos_embeds: Tuple[torch.Tensor, torch.Tensor],
        attn_mask: Optional[torch.Tensor] = None,
    ):
        h = self.norm1(x)
        h = self.attention(h, pos_embeds, attn_mask)
        h = self.layer_scale1(h)
        h = self.drop_path(h)

        x =  x + h
        
        h = self.norm2(x)
        h = self.mlp(h)
        h = self.layer_scale2(h)
        h = self.drop_path(h)
        
        return x + h

class DINOV3Encoder(nn.Module):
    def __init__(
        self, 
        hidden_size: int,
        num_hidden_layers: int,
        intermediate_size: int,
        num_attention_heads: int,
        query_bias: bool,
        key_bias: bool,
        value_bias: bool,
        proj_bias: bool,
        mlp_bias: bool,
        layer_norm_eps: float,
        layer_scale_value: float,
        drop_path_rate: float,
        use_gated_mlp: bool,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layer = nn.ModuleList(
            [
                DINOV3ViTLayer(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_attention_heads=num_attention_heads,
                    query_bias=query_bias,
                    key_bias=key_bias,
                    value_bias=value_bias,
                    proj_bias=proj_bias,
                    mlp_bias=mlp_bias,
                    layer_norm_eps=layer_norm_eps,
                    layer_scale_value=layer_scale_value,
                    drop_path_rate=drop_path_rate,
                    use_gated_mlp=use_gated_mlp,
                    dropout=dropout,
                )
                for _ in range(num_hidden_layers)
            ]
        )

    @staticmethod
    def from_config(config: DINOV3ViTConfig):
        return DINOV3Encoder(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            query_bias=config.query_bias,
            key_bias=config.key_bias,
            value_bias=config.value_bias,
            proj_bias=config.proj_bias,
            mlp_bias=config.mlp_bias,
            layer_norm_eps=config.layer_norm_eps,
            layer_scale_value=config.layer_scale_value,
            drop_path_rate=config.drop_path_rate,
            use_gated_mlp=config.use_gated_mlp,
            dropout=config.dropout,
            scale=config.scale
        )

    def forward(
            self,
            x: torch.Tensor,
            pos_embeds: torch.Tensor
        ) -> torch.Tensor:
            for module in self.layer:
                x = module(x, pos_embeds)
            return x

class DINOV3ViTModel(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_hidden_layers: int,
        patch_size: int,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        query_bias: bool,
        key_bias: bool,
        value_bias: bool,
        proj_bias: bool,
        mlp_bias: bool,
        layer_norm_eps: float,
        layer_scale_value: float,
        use_gated_mlp: bool,
        rope_theta: float,
        pos_embed_shift: float,
        pos_embed_jitter: float,
        pos_embed_scale: float,
        num_register_tokens: int,
        drop_path_rate: float,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.embeddings = DINOV3ViTEmbeddings(
            num_channels=num_channels,
            hidden_size=hidden_size,
            patch_size=patch_size,
            num_register_tokens=num_register_tokens
        )
        self.rope_embeddings = DINOV3ViTRopeEmbeddings(
            base=rope_theta,
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
            patch_size=patch_size,
            pos_embed_shift=pos_embed_shift,
            pos_embed_jitter=pos_embed_jitter,
            pos_embed_scale=pos_embed_scale
        )
        self.model = DINOV3Encoder(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            query_bias=query_bias,
            key_bias=key_bias,
            value_bias=value_bias,
            proj_bias=proj_bias,
            mlp_bias=mlp_bias,
            layer_norm_eps=layer_norm_eps,
            layer_scale_value=layer_scale_value,
            drop_path_rate=drop_path_rate,
            use_gated_mlp=use_gated_mlp,
            dropout=attention_dropout,
        )
        self.norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(
        self, 
        x: torch.Tensor, 
        bool_mask_pos: Optional[torch.Tensor] = None,
    ):
        embeds = self.embeddings(x, bool_mask_pos)
        pos_embeds = self.rope_embeddings(x)

        h = self.model(embeds, pos_embeds)
        h = self.norm(h)

        return h

    def from_config(config: DINOV3ViTConfig):
        return DINOV3ViTModel(
            num_channels=config.num_channels,
            num_hidden_layers=config.num_hidden_layers,
            patch_size=config.patch_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            query_bias=config.query_bias,
            key_bias=config.key_bias,
            value_bias=config.value_bias,
            proj_bias=config.proj_bias,
            mlp_bias=config.mlp_bias,
            layer_norm_eps=config.layer_norm_eps,
            layer_scale_value=config.layer_scale_value,
            use_gated_mlp=config.use_gated_mlp,
            rope_theta=config.rope_theta,
            pos_embed_shift=config.pos_embed_shift,
            pos_embed_jitter=config.pos_embed_jitter,
            pos_embed_scale=config.pos_embed_rescale,
            num_register_tokens=config.num_register_tokens,
            drop_path_rate=config.drop_path_rate,
            attention_dropout=config.attention_dropout,
        )
from dataclasses import dataclass

@dataclass
class QwenConfig:
    vocab_size: int
    dim_model: int
    dim_mlp: int
    num_heads: int
    num_kv_heads: int
    num_layers: int
    rope_base: int
    dropout: float
    rms_norm_eps: float
    tie_embeddings: bool

@dataclass
class Qwen3_0_6B_Config(QwenConfig):
    vocab_size: int = 151936
    dim_model: int = 1024
    dim_mlp: int = 3072
    dim_head: int = 128
    num_heads: int = 16
    num_kv_heads: int = 8
    num_layers: int = 28
    rope_base: int = 1000000
    dropout: float = 0.0
    rms_norm_eps: float = 1e-6
    tie_embeddings: bool = True
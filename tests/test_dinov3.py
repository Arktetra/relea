import torch
import torch.nn.functional as F

from transformers import AutoModel
from transformers.models.dinov2.modeling_dinov2 import eager_attention_forward
from transformers.models.dinov3_vit.modeling_dinov3_vit import apply_rotary_pos_emb
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from relea.configs.dinov3 import DINOV3ViTb16Config
from relea.models.dinov3 import (
    DINOV3ViTModel,
    apply_pos_embeds
)
from relea.utils.test_utils import shape_test, equality_test

class TestDINO:
    config = DINOV3ViTb16Config()
    reference_model = AutoModel.from_pretrained(
        "facebook/dinov3-vitb16-pretrain-lvd1689m"
    )
    model: DINOV3ViTModel = DINOV3ViTModel.from_config(config)
    model.load_state_dict(reference_model.state_dict())
    model = model.eval()
    reference_model = reference_model.eval()
    pixel_values = torch.sigmoid(torch.randn(1, 3, 336, 336))

    @torch.inference_mode()
    def test_patch_embeddings(self):
        patch_embeds = self.model.embeddings(self.pixel_values)
        ref_patch_embeds = self.reference_model.embeddings(self.pixel_values)

        shape_test(patch_embeds, ref_patch_embeds)
        equality_test(patch_embeds, ref_patch_embeds)
        
    @torch.inference_mode()
    def test_rope_embeddings(self):
        cos, sin = self.model.rope_embeddings(self.pixel_values)
        ref_cos, ref_sin = self.reference_model.rope_embeddings(self.pixel_values)
        
        shape_test(cos, ref_cos)
        equality_test(cos, ref_cos)
        shape_test(sin, ref_sin)
        equality_test(sin, ref_sin)
        
    @torch.inference_mode()
    def test_qkv_proj(self):
        patch_embeds = self.model.embeddings(self.pixel_values)
        # pos_embeds = self.model.rope_embeddings(self.pixel_values)
        
        q = self.model.model.layer[0].attention.q_proj(patch_embeds)
        k = self.model.model.layer[0].attention.k_proj(patch_embeds)
        v = self.model.model.layer[0].attention.v_proj(patch_embeds)
        
        ref_q = self.reference_model.model.layer[0].attention.q_proj(patch_embeds)
        ref_k = self.reference_model.model.layer[0].attention.k_proj(patch_embeds)
        ref_v = self.reference_model.model.layer[0].attention.v_proj(patch_embeds)
        
        shape_test(q, ref_q)
        shape_test(k, ref_k)
        shape_test(v, ref_v)
        
        equality_test(q, ref_q)
        equality_test(k, ref_k)
        equality_test(v, ref_v)
        
    @torch.inference_mode()
    def test_apply_pos_embed(self):
        patch_embeds = self.model.embeddings(self.pixel_values)
        cos, sin = self.model.rope_embeddings(self.pixel_values)
        
        B, N, _ = patch_embeds.shape
        d_head = self.config.hidden_size // self.config.num_attention_heads
        
        q = self.model.model.layer[0].attention.q_proj(patch_embeds)
        k = self.model.model.layer[0].attention.k_proj(patch_embeds)
        
        q = q.view(B, N, self.config.num_attention_heads, d_head).transpose(1, 2)  # [batch, num_heads, num_tokens, dim_head]
        k = k.view(B, N, self.config.num_attention_heads, d_head).transpose(1, 2)  # [batch, num_heads, num_tokens, dim_head]
        
        ref_q, ref_k = apply_rotary_pos_emb(q, k, cos, sin)
        q, k = apply_pos_embeds(q, k, cos, sin)
        
        shape_test(q, ref_q)
        shape_test(k, ref_k)
        equality_test(q, ref_q)
        equality_test(k, ref_k)
        
    @torch.inference_mode()
    def test_attention_detailed(self):
        patch_embeds = self.model.embeddings(self.pixel_values)
        cos, sin = self.model.rope_embeddings(self.pixel_values)
        
        B, N, _ = patch_embeds.shape
        d_head = self.config.hidden_size // self.config.num_attention_heads
        
        q = self.model.model.layer[0].attention.q_proj(patch_embeds)
        k = self.model.model.layer[0].attention.k_proj(patch_embeds)
        v = self.model.model.layer[0].attention.v_proj(patch_embeds)
        
        q = q.view(B, N, self.config.num_attention_heads, d_head).transpose(1, 2)  # [batch, num_heads, num_tokens, dim_head]
        k = k.view(B, N, self.config.num_attention_heads, d_head).transpose(1, 2)  # [batch, num_heads, num_tokens, dim_head]
        v = v.view(B, N, self.config.num_attention_heads, d_head).transpose(1, 2)  # [batch, num_heads, num_tokens, dim_head]
        
        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.reference_model.config._attn_implementation, eager_attention_forward
        )
        
        ref_z, _ = attention_interface(
            self.reference_model.model.layer[0].attention,
            q, k, v,
            None,
            dropout=0.0,
            scaling=d_head ** -0.5
        )
        z = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0,
            scale=d_head ** -0.5
        ).transpose(1, 2)
        
        shape_test(z, ref_z)
        equality_test(z, ref_z)
        
        ref_o = self.reference_model.model.layer[0].attention.o_proj(ref_z.reshape(B, N, -1).contiguous())
        o = self.model.model.layer[0].attention.o_proj(z.reshape(B, N, -1).contiguous())
        
        shape_test(o, ref_o)
        equality_test(o, ref_o)
        
    @torch.inference_mode()
    def test_attention(self):
        patch_embeds = self.model.embeddings(self.pixel_values)
        pos_embeds = self.model.rope_embeddings(self.pixel_values)
        
        o = self.model.model.layer[0].attention(patch_embeds, pos_embeds)
        ref_o, _ = self.reference_model.model.layer[0].attention(patch_embeds, None, pos_embeds)
        
        shape_test(o, ref_o)
        equality_test(o, ref_o)
        
    @torch.inference_mode()
    def test_first_attention(self):
        patch_embeds = self.model.embeddings(self.pixel_values)
        pos_embeds = self.model.rope_embeddings(self.pixel_values)
        output = self.model.model.layer[0](patch_embeds, pos_embeds)
        ref_output = self.reference_model.model.layer[0](patch_embeds, None, pos_embeds)
        
        shape_test(output, ref_output)
        equality_test(output, ref_output)
        
    @torch.inference_mode()
    def test_layer_first_half(self):
        patch_embeds = self.model.embeddings(self.pixel_values)
        pos_embeds = self.model.rope_embeddings(self.pixel_values)
        
        x = patch_embeds
        
        h = self.model.model.layer[0].norm1(x)
        h = self.model.model.layer[0].attention(h, pos_embeds)
        h = self.model.model.layer[0].layer_scale1(h)
        h = self.model.model.layer[0].drop_path(h)
        
        x = x + h 
        
        ref_x = patch_embeds
                
        ref_h = self.model.model.layer[0].norm1(ref_x)
        ref_h = self.model.model.layer[0].attention(ref_h, pos_embeds)
        ref_h = self.model.model.layer[0].layer_scale1(ref_h)
        ref_h = self.model.model.layer[0].drop_path(ref_h)
        
        ref_x = ref_x + ref_h 
        
        shape_test(x, ref_x)
        equality_test(x, ref_x)
        
    @torch.inference_mode()
    def test_layer_second_half(self):
        patch_embeds = self.model.embeddings(self.pixel_values)
        pos_embeds = self.model.rope_embeddings(self.pixel_values)
        
        x_pre = patch_embeds
                
        h = self.model.model.layer[0].norm1(x_pre)
        h = self.model.model.layer[0].attention(h, pos_embeds)
        h = self.model.model.layer[0].layer_scale1(h)
        h = self.model.model.layer[0].drop_path(h)
        
        x_pre = x_pre + h
        
        x = x_pre
        
        h = self.model.model.layer[0].norm2(x)
        h = self.model.model.layer[0].mlp(h)
        h = self.model.model.layer[0].layer_scale2(h)
        h = self.model.model.layer[0].drop_path(h)
        
        x = x + h 
        
        ref_x = x_pre
                
        ref_h = self.model.model.layer[0].norm2(ref_x)
        ref_h = self.model.model.layer[0].mlp(ref_h)
        ref_h = self.model.model.layer[0].layer_scale2(ref_h)
        ref_h = self.model.model.layer[0].drop_path(ref_h)
        
        ref_x = ref_x + ref_h 
        
        shape_test(x, ref_x)
        equality_test(x, ref_x)
        
    @torch.inference_mode()
    def test_final_outputs(self):
        reference_output = self.reference_model(self.pixel_values)
        output = self.model(self.pixel_values)

        shape_test(output, reference_output.last_hidden_state)
        equality_test(output, reference_output.last_hidden_state)
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from timm.layers import GluMlp
from functools import partial
import copy
try:
    import flash_attn.flash_attn_interface
    from models.diffatten import MultiheadFlashDiff
except ImportError:
    print("No flash module")
from einops import rearrange
from torch.nn.functional import silu
from models.network_karras import GroupNorm, Linear, Conv2d
# from models.ml_sigmoid_attention.flash_sigmoid import (
#     flash_attn_func,
#     flash_attn_kvpacked_func,
#     flash_attn_qkvpacked_func,
# )

def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )

def modulate(x, shift, scale):
    return x * PixelNorm()(1 + scale.unsqueeze(1)) + PixelNorm()(shift.unsqueeze(1))

# def modulate(x, shift, scale):
#     return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=-1, keepdim=True) + 1e-8)

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.in_channels = num_classes + use_cfg_embedding
        self.embedding_table = nn.Embedding(self.in_channels, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings
    
    def get_in_channels(self):
        return self.in_channels

   
#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, norm, wo_norm, linear_act, depth=12, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.wo_norm = wo_norm
        self.norm1 = norm(hidden_size, elementwise_affine=False, eps=1e-6) if not wo_norm else nn.Identity() 
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, norm_layer=norm, qk_norm=True, **block_kwargs) \
            if not wo_norm else Attention(hidden_size, num_heads=num_heads, qkv_bias=True, norm_layer=norm, **block_kwargs)
        self.norm2 = norm(hidden_size, elementwise_affine=False, eps=1e-6) if not wo_norm else nn.Identity()
        mlp_hidden_dim = int(hidden_size * mlp_ratio) ##### note for GluMLP they use half of mlp hidden dim, should consider double them
        if linear_act == "mish":
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.Mish, drop=0)
        elif linear_act == "relu":
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.ReLU, drop=0)   
        elif linear_act == "silu":
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.SiLU, drop=0)   
        elif linear_act == "glu_sigmoid":
            self.mlp = GluMlp(in_features=hidden_size, hidden_features=mlp_hidden_dim*2, act_layer=nn.Sigmoid, gate_last=False)
        elif linear_act == "glu_mish":
            self.mlp = GluMlp(in_features=hidden_size, hidden_features=mlp_hidden_dim*2, act_layer=nn.Mish, gate_last=False)
        elif linear_act == "glu_silu":
            self.mlp = GluMlp(in_features=hidden_size, hidden_features=mlp_hidden_dim*2, act_layer=nn.SiLU, gate_last=False)
        elif linear_act == "glu_gelu":
            self.mlp = GluMlp(in_features=hidden_size, hidden_features=mlp_hidden_dim*2, act_layer=partial(nn.GELU, "tanh"), gate_last=False)
        elif linear_act == "gelu":
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=partial(nn.GELU, "tanh"), drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        
    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
    
class DiTBlockFlashAttn(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, norm, wo_norm, linear_act, depth=12, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.wo_norm = wo_norm
        self.norm1 = norm(hidden_size, elementwise_affine=False, eps=1e-6) if not wo_norm else nn.Identity() 
        self.num_heads = num_heads
        self.attn_qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.attn_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.q_norm = norm(hidden_size) if wo_norm else nn.Identity()
        self.k_norm = norm(hidden_size) if wo_norm else nn.Identity()
        # self.prev_norm = norm(hidden_size) if wo_norm else nn.Identity()
        
        self.norm2 = norm(hidden_size, elementwise_affine=False, eps=1e-6) #if not wo_norm else nn.Identity()
        mlp_hidden_dim = int(hidden_size * mlp_ratio) ##### note for GluMLP they use half of mlp hidden dim, should consider double them
        if linear_act == "mish":
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.Mish, drop=0)
        elif linear_act == "relu":
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.ReLU, drop=0)   
        elif linear_act == "silu":
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.SiLU, drop=0)   
        elif linear_act == "glu_sigmoid":
            self.mlp = GluMlp(in_features=hidden_size, hidden_features=mlp_hidden_dim*2, act_layer=nn.Sigmoid, gate_last=False)
        elif linear_act == "glu_mish":
            self.mlp = GluMlp(in_features=hidden_size, hidden_features=mlp_hidden_dim*2, act_layer=nn.Mish, gate_last=False)
        elif linear_act == "glu_silu":
            self.mlp = GluMlp(in_features=hidden_size, hidden_features=mlp_hidden_dim*2, act_layer=nn.SiLU, gate_last=False)
        elif linear_act == "glu_gelu":
            self.mlp = GluMlp(in_features=hidden_size, hidden_features=mlp_hidden_dim*2, act_layer=partial(nn.GELU, "tanh"), gate_last=False)
        elif linear_act == "gelu":
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=partial(nn.GELU, "tanh"), drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def _attn(self, x):
        # x = self.prev_norm(x)
        qkv = self.attn_qkv(x)
        q,k,v = qkv.chunk(3, dim=2)
        q, k = self.q_norm(q), self.k_norm(k)
        qkv = torch.cat([q, k, v], dim=2).to(dtype=qkv.dtype)
        qkv = rearrange(qkv,
                        'b s (three h d) -> (b s) three h d',
                        three=3,
                        h=self.num_heads)
        batch_size, seq_len = x.size(0), x.size(1)
        cu_seqlens = torch.arange(
            0, (batch_size + 1) * seq_len, step=seq_len,
            dtype=torch.int32, device=qkv.device)
        x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, seq_len, 0., causal=False)
        x = rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)
        return self.attn_out(x)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        attn = self._attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_msa.unsqueeze(1) * attn
        mlp = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = x + gate_mlp.unsqueeze(1) * mlp
        return x
    
# class DiTBlockFlashSigmoidAttn(nn.Module):
#     """
#     A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
#     """
#     def __init__(self, hidden_size, num_heads, norm, wo_norm, linear_act, depth=12, mlp_ratio=4.0, **block_kwargs):
#         super().__init__()
#         self.wo_norm = wo_norm
#         self.norm1 = norm(hidden_size, elementwise_affine=False, eps=1e-6) if not wo_norm else nn.Identity() 
#         self.num_heads = num_heads
#         self.attn_qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
#         self.attn_out = nn.Linear(hidden_size, hidden_size, bias=False)
#         self.q_norm = norm(hidden_size) if wo_norm else nn.Identity()
#         self.k_norm = norm(hidden_size) if wo_norm else nn.Identity()
#         # self.prev_norm = norm(hidden_size) if wo_norm else nn.Identity()
        
#         self.norm2 = norm(hidden_size, elementwise_affine=False, eps=1e-6) #if not wo_norm else nn.Identity()
#         mlp_hidden_dim = int(hidden_size * mlp_ratio) ##### note for GluMLP they use half of mlp hidden dim, should consider double them
#         if linear_act == "mish":
#             self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.Mish, drop=0)
#         elif linear_act == "relu":
#             self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.ReLU, drop=0)   
#         elif linear_act == "silu":
#             self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.SiLU, drop=0)   
#         elif linear_act == "glu_sigmoid":
#             self.mlp = GluMlp(in_features=hidden_size, hidden_features=mlp_hidden_dim*2, act_layer=nn.Sigmoid, gate_last=False)
#         elif linear_act == "glu_mish":
#             self.mlp = GluMlp(in_features=hidden_size, hidden_features=mlp_hidden_dim*2, act_layer=nn.Mish, gate_last=False)
#         elif linear_act == "glu_silu":
#             self.mlp = GluMlp(in_features=hidden_size, hidden_features=mlp_hidden_dim*2, act_layer=nn.SiLU, gate_last=False)
#         elif linear_act == "glu_gelu":
#             self.mlp = GluMlp(in_features=hidden_size, hidden_features=mlp_hidden_dim*2, act_layer=partial(nn.GELU, "tanh"), gate_last=False)
#         elif linear_act == "gelu":
#             self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=partial(nn.GELU, "tanh"), drop=0)
#         self.adaLN_modulation = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(hidden_size, 6 * hidden_size, bias=True),
#         )

#     def _attn(self, x):
#         # x = self.prev_norm(x)
#         qkv = self.attn_qkv(x)
#         q,k,v = qkv.chunk(3, dim=2)
#         q, k = self.q_norm(q), self.k_norm(k)
#         qkv = torch.cat([q, k, v], dim=2).to(dtype=qkv.dtype)
#         qkv = rearrange(qkv,
#                         'b s (three h d) -> (b s) three h d',
#                         three=3,
#                         h=self.num_heads)
#         batch_size, seq_len = x.size(0), x.size(1)
#         cu_seqlens = torch.arange(
#             0, (batch_size + 1) * seq_len, step=seq_len,
#             dtype=torch.int32, device=qkv.device)
#         x = flash_attn_qkvpacked_func(qkv, 0., causal=False)
#         x = rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)
#         return self.attn_out(x)

#     def forward(self, x, c):
#         shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
#         attn = self._attn(modulate(self.norm1(x), shift_msa, scale_msa))
#         x = x + gate_msa.unsqueeze(1) * attn
#         mlp = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
#         x = x + gate_mlp.unsqueeze(1) * mlp
#         return x

class DiTBlockFlashDiffattn(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, norm, wo_norm, linear_act, depth=12, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.wo_norm = wo_norm
        self.norm1 = norm(hidden_size, elementwise_affine=False, eps=1e-6) if not wo_norm else nn.Identity() 
        self.num_heads = num_heads
        self.attn = MultiheadFlashDiff(embed_dim=hidden_size, depth=depth, num_heads=num_heads, norm_layer=norm)
        self.norm2 = norm(hidden_size, elementwise_affine=False, eps=1e-6) #if not wo_norm else nn.Identity()
        mlp_hidden_dim = int(hidden_size * mlp_ratio) ##### note for GluMLP they use half of mlp hidden dim, should consider double them
        if linear_act == "mish":
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.Mish, drop=0)
        elif linear_act == "relu":
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.ReLU, drop=0)   
        elif linear_act == "silu":
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.SiLU, drop=0)   
        elif linear_act == "glu_sigmoid":
            self.mlp = GluMlp(in_features=hidden_size, hidden_features=mlp_hidden_dim*2, act_layer=nn.Sigmoid, gate_last=False)
        elif linear_act == "glu_mish":
            self.mlp = GluMlp(in_features=hidden_size, hidden_features=mlp_hidden_dim*2, act_layer=nn.Mish, gate_last=False)
        elif linear_act == "glu_silu":
            self.mlp = GluMlp(in_features=hidden_size, hidden_features=mlp_hidden_dim*2, act_layer=nn.SiLU, gate_last=False)
        elif linear_act == "glu_gelu":
            self.mlp = GluMlp(in_features=hidden_size, hidden_features=mlp_hidden_dim*2, act_layer=partial(nn.GELU, "tanh"), gate_last=False)
        elif linear_act == "gelu":
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=partial(nn.GELU, "tanh"), drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, norm, wo_norm):
        super().__init__()
        self.wo_norm = wo_norm
        self.norm_final = norm(hidden_size, elementwise_affine=False, eps=1e-6) if not wo_norm else nn.Identity() 
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

#----------------------------------------------------------------------------
# NonScalingLayer normalization.

class NonScalingLayerNorm(nn.LayerNorm):
    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            torch.nn.init.ones_(self.weight)
            self.weight.requires_grad = False
            if self.bias is not None:
                torch.nn.init.zeros_(self.bias)
                
#----------------------------------------------------------------------------
# Resblock patch embedding.
class FinalConvLayer(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, emb_channels, up=False, down=False, dropout=0, skip_scale=1, eps=1e-5,
        resample_filter=[1,1], resample_proj=False, adaptive_scale=True, init=dict(), init_zero=dict(init_weight=0),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=3, up=up, down=down, resample_filter=resample_filter, **init)
        self.affine = Linear(in_features=emb_channels, out_features=out_channels*(2 if adaptive_scale else 1), **init)
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero)

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels!= in_channels else 0
            self.skip = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=kernel, up=up, down=down, resample_filter=resample_filter, **init)

    def forward(self, x, emb):
        orig = x
        x = self.conv0(silu(self.norm0(x)))
        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = silu(self.norm1(x.add_(params)))
        x = self.conv1(torch.nn.functional.dropout(x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale
        return x 

#----------------------------------------------------------------------------
# Resblock final layer and unpatchify


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        no_scale = False,
        linear_act=None,
        wo_norm=False,
        attn_type=False,
        num_register=0,
        final_conv=False,
        use_repa=False,
        projector_dim=None,
        z_dims=None,
        encoder_depth=None,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.depth = depth
        self.attn_type = attn_type
        self.num_register = num_register
        self.final_conv = final_conv
        # init for conv
        init = dict(init_mode='kaiming_uniform', init_weight=np.sqrt(1/3), init_bias=np.sqrt(1/3))
        init_zero = dict(init_mode='kaiming_uniform', init_weight=0, init_bias=0)

        self.norm = NonScalingLayerNorm if no_scale else nn.LayerNorm
        if attn_type == "normal":
            self.attn_blk = DiTBlock 
        elif attn_type == "flash":
            self.attn_blk = DiTBlockFlashAttn
        elif attn_type == "diffattn":
            self.attn_blk = DiTBlockFlashDiffattn
        elif attn_type == "sigmoidattn":
            self.attn_blk = None # DiTBlockFlashSigmoidAttn
        else:
            raise("No implementation for this")
        
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_register, hidden_size), requires_grad=False)
        if self.num_register > 0:
            self.register_tokens = nn.Parameter(torch.rand((1, self.num_register, hidden_size)))

        self.blocks = nn.ModuleList([
            self.attn_blk(hidden_size, 
                     num_heads, 
                     mlp_ratio=mlp_ratio, 
                     norm=self.norm, 
                     linear_act=linear_act, 
                     wo_norm=wo_norm, 
                     depth=self.depth) \
                for _ in range(depth)
        ])
        if not self.final_conv:
            self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, norm=self.norm, wo_norm=wo_norm)
        else:
            self.final_layer = FinalConvLayer(hidden_size, hidden_size//2, hidden_size, up=True, init=init, init_zero=init_zero)
            self.output = nn.Sequential(
                GroupNorm(num_channels=hidden_size//2),
                # self.out_norm = nn.LayerNorm(normalized_shape=[hidden_size//2, input_size, input_size])
                Conv2d(in_channels=hidden_size//2, out_channels=self.out_channels, kernel=3, **init_zero))
        
        ############## REPA ##############
        self.use_repa = use_repa
        self.encoder_depth = encoder_depth
        if self.use_repa:   
            self.projectors = nn.ModuleList([
                build_mlp(hidden_size, projector_dim, z_dim) for z_dim in z_dims
            ])
        ############## REPA ##############
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5), cls_token=(self.num_register>0), extra_tokens=self.num_register)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        if not self.final_conv:
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.final_layer.linear.weight, 0)
            nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y=None, is_train=False):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        if y is None:
            y = torch.ones(x.size(0), dtype=torch.long, device=x.device) * (self.y_embedder.get_in_channels() - 1)
        x = self.x_embedder(x)
        if self.num_register > 0:
            x = torch.cat([x, self.register_tokens.expand(x.shape[0], -1, -1)], dim=1)
        x = x + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        N, T, D = x.shape
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)
        if self.attn_type in ["flash", "diffattn"]:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                for idx, block in enumerate(self.blocks):
                    x = block(x, c)
        else:
            for idx, block in enumerate(self.blocks):
                x = block(x, c)
                if is_train and self.use_repa and (idx + 1) == self.encoder_depth:
                    zs = [projector(x.reshape(-1, D)).reshape(N, T, -1) for projector in self.projectors]
        if self.num_register > 0:
            x = x[:, :self.x_embedder.num_patches, :]
        if not self.final_conv:
            x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
            x = self.unpatchify(x)                   # (N, out_channels, H, W)
        else:
            h = w = int(x.shape[1] ** 0.5)
            x = x.reshape(shape=(x.shape[0], h, w, -1)).permute(0, 3, 1, 2)
            x = self.final_layer(x, c)
            x = self.output(x)
        
        if is_train and self.use_repa:
            return x, zs
        return x

    def forward_with_cfg(self, x, t, cfg_scale, y=None):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # print(model_out.shape)
        # exit()
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(model_out, len(model_out) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return eps #torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([pos_embed, np.zeros([extra_tokens, embed_dim])], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}
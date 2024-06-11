import math
from typing import Optional
from functools import partial

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed
from timm.models.layers import DropPath, PatchEmbed
# from timm.models.layers.patch_embed import resample_patch_embed
from mamba_ssm.modules.mamba_simple import Mamba
from rope import *
from pe.my_rotary import get_2d_sincos_rotary_embed, apply_rotary
from pe.cpe import PosCNN, AdaInPosCNN
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
from scanning.jpeg_compression import dct2, idct2
import torch_dct as dct



def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


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
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
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
        use_cfg_embedding = dropout_prob > 0 # 1
        self.in_channels = num_classes + use_cfg_embedding # 1001
        self.embedding_table = nn.Embedding(self.in_channels, hidden_size)
        self.num_classes = num_classes # 1000
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels) # 1000 or labels
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

    def get_in_channels(self):
        return self.in_channels # 1001


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
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

class GatedMlpBlock(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=True)
        self.fc2 = linear_layer(in_features, hidden_features, bias=True)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc3 = linear_layer(hidden_features, out_features, bias=True)

    def forward(self, x):
        x = self.act(self.fc1(x)) + self.fc2(x)
        x = self.norm(x)
        x = self.fc3(x)
        return x
    
class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class DiMBlock(nn.Module):
    def __init__(self, 
                 dim, 
                 mixer_cls, 
                 norm_cls=nn.LayerNorm, 
                 fused_add_norm=False, 
                 residual_in_fp32=False, 
                 drop_path=0.1,
                 skip=False):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.norm_2 = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        mlp_hidden_dim = int(dim * 4)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, 
                hidden_states: Tensor, 
                c: Optional[Tensor] = None, 
                inference_params=None, 
                skip=None):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
            c: (N, D)
        """
        if self.skip_linear is not None:
            hidden_states = self.skip_linear(torch.cat([hidden_states, skip], dim=-1))
        shift_ssm, scale_ssm, gate_ssm, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        hidden_states = hidden_states + gate_ssm.unsqueeze(1) * self.drop_path(self.mixer(modulate(self.norm(hidden_states), shift_ssm, scale_ssm), inference_params=inference_params))
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm_2(hidden_states), shift_mlp, scale_mlp))
        
        return hidden_states

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

class DiMMambaAttn(nn.Module):
    def __init__(self, 
                 dim,
                 patch_size,
                 mixer_cls, 
                 norm_cls=nn.LayerNorm, 
                 fused_add_norm=False, 
                 residual_in_fp32=False, 
                 drop_path=0.1,
                 skip=False):
        """
        This idea is divided image into patches. We apply Mamba inside patches and apply attn patch-wise. This mitigate the role of seq order of mamba and attn capture the global attn
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.norm_2 = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        mlp_hidden_dim = int(dim * 4)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, 
                hidden_states: Tensor, 
                c: Optional[Tensor] = None, 
                inference_params=None, 
                skip=None):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
            c: (N, D)
        """
        if self.skip_linear is not None:
            hidden_states = self.skip_linear(torch.cat([hidden_states, skip], dim=-1))
        shift_ssm, scale_ssm, gate_ssm, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        hidden_states = hidden_states + gate_ssm.unsqueeze(1) * self.drop_path(self.mixer(modulate(self.norm(hidden_states), shift_ssm, scale_ssm), inference_params=inference_params))
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm_2(hidden_states), shift_mlp, scale_mlp))
        
        return hidden_states

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
    
class DiMBlockRaw(nn.Module):
    def __init__(self, dim,
                 mixer_cls,
                 norm_cls=nn.LayerNorm, 
                 fused_add_norm=False, 
                 residual_in_fp32=False, 
                 drop_path=0.,
                 skip=False):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 3 * dim, bias=True))
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None

    def forward(
        self, hidden_states: Tensor, c: Optional[Tensor] = None, inference_params=None, skip=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if self.skip_linear is not None:
            hidden_states = self.skip_linear(torch.cat([hidden_states, skip], dim=-1))
        shift_ssm, scale_ssm, gate_ssm = self.adaLN_modulation(c).chunk(3, dim=1)
        hidden_states = hidden_states + gate_ssm.unsqueeze(1) * self.mixer(modulate(self.norm(hidden_states), shift_ssm, scale_ssm), inference_params=inference_params)
        return hidden_states

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)



class UDiM(nn.Module):
    def __init__(
        self,
        img_resolution=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        label_dropout=0.1,
        num_classes=1000,
        learn_sigma=True,
        ssm_cfg=None,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        bimamba_type="none",
        initializer_cfg=None,
        pe_type = "ape",
        block_type = "linear",
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.depth = depth = int(depth//2)
        self.initializer_cfg = initializer_cfg
        # using rotary embedding
        self.pe_type = pe_type
        # block type
        self.block_type = block_type

        self.x_embedder = PatchEmbed(img_resolution, patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, label_dropout)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        
        if self.pe_type == "rope":
            # I'm not sure what pt_seq_len for
            self.emb_sin, self.emb_cos = get_2d_sincos_rotary_embed(hidden_size, int(num_patches**0.5))
            self.emb_sin = torch.from_numpy(self.emb_sin).to(dtype=torch.float32)
            self.emb_cos = torch.from_numpy(self.emb_cos).to(dtype=torch.float32)
        elif self.pe_type == "cpe":
            self.pos_cnn = AdaInPosCNN(hidden_size, hidden_size)
            # self.pos_cnn = PosCNN(hidden_size, hidden_size)


        self.down_blocks = nn.ModuleList(
            [
                create_block(
                    hidden_size,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=1e-5,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    bimamba_type=bimamba_type,
                    drop_path=0.,
                    block_type=block_type
                )
                for i in range(depth)
            ]
        )
        self.mid_block = create_block(
                    hidden_size,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=1e-5,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=depth,
                    bimamba_type=bimamba_type,
                    drop_path=0.,
                    block_type=block_type
                )
        self.up_blocks = nn.ModuleList(
            [
                create_block(
                    hidden_size,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=1e-5,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=depth+i+1,
                    bimamba_type=bimamba_type,
                    drop_path=0.,
                    skip=True,
                    block_type=block_type
                )
                for i in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
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
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5))
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
        for block in self.down_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        for block in self.up_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.mid_block.adaLN_modulation[-1].bias, 0)
        
        
        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=self.depth,
                **(self.initializer_cfg if self.initializer_cfg is not None else {}),
            )
        )

        # Zero-out output layers:
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

    def forward(self, x, t = None, y=None, **kwargs):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        if t is None:
            # for compute Gflops
            t = torch.randint(0, 1000, (x.shape[0],), device=x.device)
        if y is None:
            y = torch.ones(x.size(0), dtype=torch.long, device=x.device) * (self.y_embedder.get_in_channels() - 1) # 1000
        
        t = self.t_embedder(t)  # (N, D)
        y = self.y_embedder(y, self.training)  # (N, D)
        c = t + y  # (N, D)
        
        # add rope !
        if self.pe_type == "ape":
            x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        elif self.pe_type == "rope":
            self.emb_cos = self.emb_cos.to(x.device)
            self.emb_sin = self.emb_sin.to(x.device)
            x = apply_rotary(self.x_embedder(x), self.emb_sin, self.emb_cos)
        elif self.pe_type == "cpe":
            x = self.x_embedder(x)
            h = w = int(self.x_embedder.num_patches**0.5)
            # x = self.pos_cnn(x, c, H = h, W = w)
            # x = self.pos_cnn(x, H = h, W = w)
        else:
            raise("Unsupport PE")

        skips = []
        # Down block
        for idx, block in enumerate(self.down_blocks):
            if self.pe_type == "ape":
                # PE + feature (Pefeat)
                # if idx <= 5:
                #     x = block(x, c, inference_params=None) + self.pos_embed  # (N, T, D)
                # else:
                #     x = block(x, c, inference_params=None)
                # ViM raw
                x = block(x, c, inference_params=None)
            elif self.pe_type == "rope":
                # use RoPE
                x = block(apply_rotary(x, self.emb_sin, self.emb_cos), c, inference_params=None)
                # x = block(x, c, inference_params=None)
            elif self.pe_type == "cpe":
                if idx == 1:
                    h = w = int(self.x_embedder.num_patches**0.5)
                    x = self.pos_cnn(x, c, H = h, W = w)
                x = block(x, c, inference_params=None)
            skips.append(x)
        # Middle block
        x = self.mid_block(x, c, inference_params=None)
        # Up block
        for idx, block in enumerate(self.up_blocks):
            if self.pe_type == "ape":
                # PE + feature (Pefeat)
                # if idx <= 5:
                #     x = block(x, c, inference_params=None) + self.pos_embed  # (N, T, D)
                # else:
                #     x = block(x, c, inference_params=None)
                # ViM raw
                x = block(x, c, inference_params=None, skip=skips.pop())
            elif self.pe_type == "rope":
                # use RoPE
                # x = block(apply_rotary(x, self.emb_sin, self.emb_cos), c, inference_params=None)
                x = block(x, c, inference_params=None, skip=skips.pop())
            elif self.pe_type == "cpe":
                x = block(x, c, inference_params=None, skip=skips.pop())
                
        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y=None, cfg_scale=1.0, **kwargs):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(t, combined, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: blk.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, blk in enumerate(self.blocks)
        }

class DiM(nn.Module):
    def __init__(
        self,
        img_resolution=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        label_dropout=0.1,
        num_classes=1000,
        learn_sigma=True,
        ssm_cfg=None,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        bimamba_type="none",
        initializer_cfg=None,
        pe_type = "ape",
        cpe_dilated = False,
        block_type = "linear",
        using_dct = False,
        use_wavelet = False,
        is_jamba = False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.depth = depth
        self.initializer_cfg = initializer_cfg
        self.using_dct = using_dct
        # using rotary embedding
        self.pe_type = pe_type
        # block type
        self.block_type = block_type
        self.x_embedder = PatchEmbed(img_resolution, patch_size, in_channels, hidden_size)
        # self.freq_embedder = PatchEmbed(img_resolution, patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, label_dropout)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        
        if self.pe_type == "rope":
            # I'm not sure what pt_seq_len for
            self.emb_sin, self.emb_cos = get_2d_sincos_rotary_embed(hidden_size, int(num_patches**0.5))
            self.emb_sin = torch.from_numpy(self.emb_sin).to(dtype=torch.float32)
            self.emb_cos = torch.from_numpy(self.emb_cos).to(dtype=torch.float32)
        elif self.pe_type == "cpe":
            # self.pos_cnn = AdaInPosCNN(hidden_size, hidden_size)
            print("using cpe")
            if not cpe_dilated:
                self.pos_cnn = AdaInPosCNN(hidden_size, hidden_size)
                # self.pos_cnn = PosCNN(hidden_size, hidden_size)
            else:
                self.pos_cnn = AdaInPosCNN(hidden_size, hidden_size, use_dilated=cpe_dilated)
                # self.pos_cnn = PosCNN(hidden_size, hidden_size, use_dilated=cpe_dilated)

        if is_jamba:
            self.blocks = nn.ModuleList()
            for i in range(depth):
                self.blocks.append(create_block(
                        hidden_size,
                        ssm_cfg=ssm_cfg,
                        norm_epsilon=1e-5,
                        rms_norm=rms_norm,
                        residual_in_fp32=residual_in_fp32,
                        fused_add_norm=fused_add_norm,
                        layer_idx=i,
                        bimamba_type=bimamba_type,
                        drop_path=0.,
                        block_type=block_type
                    ) if i % 6 != 5 else
                    create_block(
                        hidden_size,
                        ssm_cfg=ssm_cfg,
                        norm_epsilon=1e-5,
                        rms_norm=rms_norm,
                        residual_in_fp32=residual_in_fp32,
                        fused_add_norm=fused_add_norm,
                        layer_idx=i,
                        bimamba_type=bimamba_type,
                        drop_path=0.,
                        block_type="trans"))
        else:
            self.blocks = nn.ModuleList(
                [
                    create_block(
                        hidden_size,
                        ssm_cfg=ssm_cfg,
                        norm_epsilon=1e-5,
                        rms_norm=rms_norm,
                        residual_in_fp32=residual_in_fp32,
                        fused_add_norm=fused_add_norm,
                        layer_idx=i,
                        bimamba_type=bimamba_type,
                        drop_path=0.,
                        block_type=block_type
                    )
                    for i in range(depth)
                ]
            )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
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
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5))
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
        
        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=self.depth,
                **(self.initializer_cfg if self.initializer_cfg is not None else {}),
            )
        )

        # Zero-out output layers:
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

    def forward(self, x, t = None, y=None, **kwargs):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        raw_image = x
        if t is None:
            # for compute Gflops
            t = torch.randint(0, 1000, (x.shape[0],), device=x.device)
        if y is None:
            y = torch.ones(x.size(0), dtype=torch.long, device=x.device) * (self.y_embedder.get_in_channels() - 1) # 1000
        
        t = self.t_embedder(t)  # (N, D)
        y = self.y_embedder(y, self.training)  # (N, D)
        c = t + y  # (N, D)
        
        # add rope !
        if self.pe_type == "ape":
            if self.using_dct:
                freq = dct.dct_2d(raw_image)
                x = torch.cat([x, freq], dim=1)
            x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
            
        elif self.pe_type == "rope":
            self.emb_cos = self.emb_cos.to(x.device)
            self.emb_sin = self.emb_sin.to(x.device)
            x = apply_rotary(self.x_embedder(x), self.emb_sin, self.emb_cos)
        elif self.pe_type == "cpe":
            x = self.x_embedder(x)
            h = w = int(self.x_embedder.num_patches**0.5)
            # x = self.pos_cnn(x, c, H = h, W = w)
            # x = self.pos_cnn(x, H = h, W = w)
        else:
            raise("Unsupport PE")

        # please comment in/out if want to use ViM Pefeat
        for idx, block in enumerate(self.blocks):
            if self.pe_type == "ape":
                # ViM raw
                x = block(x, c)
            elif self.pe_type == "rope":
                # use RoPE
                # x = block(apply_rotary(x, self.emb_sin, self.emb_cos), c)
                x = block(x, c)
            elif self.pe_type == "cpe":
                if idx == 1:
                    h = w = int(self.x_embedder.num_patches**0.5)
                    x = self.pos_cnn(x, c, H = h, W = w)
                    # x = self.pos_cnn(x, H = h, W = w)
                x = block(x, c)
        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y=None, cfg_scale=1.0, **kwargs):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(t, combined, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: blk.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, blk in enumerate(self.blocks)
        }



# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)

def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    bimamba_type="none",
    block_type="linear",
    skip=False,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, bimamba_type=bimamba_type, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    if block_type == "linear":
        block = DiMBlock(
            d_model,
            mixer_cls,
            norm_cls=norm_cls,
            drop_path=drop_path,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            skip=skip,
        )
        block.layer_idx = layer_idx
    elif block_type == "raw":
        block = DiMBlockRaw(
            d_model,
            mixer_cls,
            norm_cls=norm_cls,
            drop_path=drop_path,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            skip=skip,
        )
        block.layer_idx = layer_idx
    elif block_type == "trans":
        block = DiTBlock(hidden_size=d_model, num_heads=16, mlp_ratio=4.0)
    
    return block


def DiM_XL_2(**kwargs):
    return DiM(depth=28, 
        hidden_size=1152, 
        patch_size=2, 
        bimamba_type="v2", 
        initializer_cfg=None,
        fused_add_norm=False, 
        rms_norm=False, 
        ssm_cfg=None, 
        residual_in_fp32=True,
        **kwargs)

def DiM_L_2(**kwargs):
    return DiM(depth=24, # make it double for raw
        hidden_size=1024, 
        patch_size=2, 
        bimamba_type="v2", 
        initializer_cfg=None,
        fused_add_norm=False, 
        rms_norm=False, 
        ssm_cfg=None, 
        residual_in_fp32=True,
        **kwargs)
    
def DiM_L_2_Jamba(**kwargs):
    return DiM(depth=24, # make it double for raw
        hidden_size=1024, 
        patch_size=2, 
        bimamba_type="v2", 
        initializer_cfg=None,
        fused_add_norm=False, 
        rms_norm=False, 
        ssm_cfg=None, 
        residual_in_fp32=True,
        is_jamba=True,
        **kwargs)

def DiM_L_2_Wavelet(**kwargs):
    return DiM(depth=24, # make it double for raw
        hidden_size=1024, 
        patch_size=1, 
        bimamba_type="v2", 
        initializer_cfg=None,
        fused_add_norm=False, 
        rms_norm=False, 
        ssm_cfg=None, 
        residual_in_fp32=True,
        **kwargs)

def DiM_LS_2(**kwargs):
    return DiM(depth=18, 
        hidden_size=1024, 
        patch_size=2, 
        bimamba_type="v2", 
        initializer_cfg=None,
        fused_add_norm=False, 
        rms_norm=False, 
        ssm_cfg=None, 
        residual_in_fp32=True,
        **kwargs)
    
def DiM_B_2(**kwargs):
    return DiM(depth=12, 
        hidden_size=768, 
        patch_size=2, 
        bimamba_type="v2", 
        initializer_cfg=None,
        fused_add_norm=False, 
        rms_norm=False, 
        ssm_cfg=None, 
        residual_in_fp32=True,
        **kwargs)

def UDiM_XL_2(**kwargs):
    return UDiM(depth=28, 
        hidden_size=1152, 
        patch_size=2, 
        bimamba_type="v2", 
        initializer_cfg=None,
        fused_add_norm=False, 
        rms_norm=False, 
        ssm_cfg=None, 
        residual_in_fp32=True,
        **kwargs)

def UDiM_L_2(**kwargs):
    return UDiM(depth=24, 
        hidden_size=1024, 
        patch_size=2, 
        bimamba_type="v2", 
        initializer_cfg=None,
        fused_add_norm=False, 
        rms_norm=False, 
        ssm_cfg=None, 
        residual_in_fp32=True,
        **kwargs)
    
def UDiM_B_2(**kwargs):
    return UDiM(depth=12, 
        hidden_size=768, 
        patch_size=2, 
        bimamba_type="v2", 
        initializer_cfg=None,
        fused_add_norm=False, 
        rms_norm=False, 
        ssm_cfg=None, 
        residual_in_fp32=True,
        **kwargs)

DiM_models = {
    "DiM-XL/2": DiM_XL_2,
    "DiM-L/2": DiM_L_2,
    "DiM-L/2_Wave": DiM_L_2_Wavelet,
    "DiM-L/2_Jamba": DiM_L_2_Jamba,
    "DiM-LS/2": DiM_LS_2,
    "DiM-B/2": DiM_B_2,
    "UDiM-XL/2": UDiM_XL_2,
    "UDiM-L/2": UDiM_L_2,
    "UDiM-B/2": UDiM_B_2,
}

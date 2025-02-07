import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SigmoidAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def custom_sigmoid(self, x):
        return 0.5 * (1 + torch.tanh(0.5 * x))

    def forward(self, x):
        B, N, C = x.shape
        # Calculate bias_scalar based on the number of tokens (N)
        self.bias_scalar = -math.log(N)

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        # Compute the attention logits with the added bias scalar
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # Apply the custom sigmoid
        attn = self.custom_sigmoid(attn + self.bias_scalar)  
        attn = self.attn_drop(attn)

        # Compute the output
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LinearAttention(Attention_):
    r"""Lightweight linear attention"""

    PAD_VAL = 1

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: Optional[int] = None,
        heads_ratio: float = 1.0,
        dim=32,
        eps=1e-15,
        use_bias=False,
        qk_norm=False,
        norm_eps=1e-5,
    ):
        heads = heads or int(out_dim // dim * heads_ratio)
        super().__init__(in_dim, num_heads=heads, qkv_bias=use_bias)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.dim = out_dim // heads  # TODO: need some change
        self.eps = eps

        self.kernel_func = nn.ReLU(inplace=False)
        if qk_norm:
            self.q_norm = RMSNorm(in_dim, scale_factor=1.0, eps=norm_eps)
            self.k_norm = RMSNorm(in_dim, scale_factor=1.0, eps=norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    @torch.amp.autocast("cuda", enabled=os.environ.get("AUTOCAST_LINEAR_ATTN", False) == "true")
    def attn_matmul(self, q, k, v: torch.Tensor) -> torch.Tensor:
        # lightweight linear attention
        q = self.kernel_func(q)  # B, h, h_d, N
        k = self.kernel_func(k)

        use_fp32_attention = getattr(self, "fp32_attention", False)  # necessary for NAN loss
        if use_fp32_attention:
            q, k, v = q.float(), k.float(), v.float()

        v = F.pad(v, (0, 0, 0, 1), mode="constant", value=LiteLA.PAD_VAL)
        vk = torch.matmul(v, k)
        out = torch.matmul(vk, q)

        if out.dtype in [torch.float16, torch.bfloat16]:
            out = out.float()
        out = out[:, :, :-1] / (out[:, :, -1:] + self.eps)

        return out

    def forward(self, x: torch.Tensor, mask=None, HW=None, block_id=None) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, C)
        q, k, v = qkv.unbind(2)  # B, N, 3, C --> B, N, C
        dtype = q.dtype

        q = self.q_norm(q).transpose(-1, -2)  # (B, N, C) -> (B, C, N)
        k = self.k_norm(k).transpose(-1, -2)  # (B, N, C) -> (B, C, N)
        v = v.transpose(-1, -2)

        q = q.reshape(B, C // self.dim, self.dim, N)  # (B, h, h_d, N)
        k = k.reshape(B, C // self.dim, self.dim, N).transpose(-1, -2)  # (B, h, N, h_d)
        v = v.reshape(B, C // self.dim, self.dim, N)  # (B, h, h_d, N)

        out = self.attn_matmul(q, k, v).to(dtype)

        out = out.view(B, C, N).permute(0, 2, 1)  # B, N, C
        out = self.proj(out)

        if torch.get_autocast_gpu_dtype() == torch.float16:
            out = out.clip(-65504, 65504)

        return out

    @property
    def module_str(self) -> str:
        _str = type(self).__name__ + "("
        eps = f"{self.eps:.1E}"
        _str += f"i={self.in_dim},o={self.out_dim},h={self.heads},d={self.dim},eps={eps}"
        return _str

    def __repr__(self):
        return f"EPS{self.eps}-" + super().__repr__()
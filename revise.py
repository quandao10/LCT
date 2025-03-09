import torch
import torch.nn as nn
import math

def scale_dot_product(q, k, v, mask=None):
    d = q.size(-1)
    scores = torch.bmm(q, k.transpose(1,2))/math.sqrt(d)
    if mask is not None:
        scores = scores.masked_fill(mask==0, float('-inf'))
    attention_weight = torch.softmax(scores, dim=-1)
    output = torch.bmm(attention_weight, v)
    return output

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.d_k = embed_dim
        self.d_v = embed_dim
        
        # Linear layers projection
        self.query_proj = nn.Linear(embed_dim, self.d_k)
        self.key_proj = nn.Linear(embed_dim, self.d_k)
        self.value_proj = nn.Linear(embed_dim, self.d_v)
        
        self.out_proj = nn.Linear(self.d_v, embed_dim)
        
    def forward(self, x, mask=None):
        
        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)
        
        attention = scale_dot_product(Q, K, V, mask)
        
        output = self.out_proj(attention)
        return output, attention        
    
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        assert embed_dim % heads == 0, "the embed dim must be divisible by heads"
        
        self.head_dim = embed_dim // heads
        
        self.linear_proj = nn.Linear(embed_dim, embed_dim*3)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        
    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.linear_proj(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask==0, float('-inf'))
        attn = nn.functional.softmax(attn, dim=-1)
        x = attn @ v
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.out_proj(x)
        return x        


class Convolution(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=(3, 3)):
        super(Convolution, self).__init__()
        self.h, self.w = kernel_size
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.weight = nn.Parameter(torch.randn((channel_out, channel_in, self.h, self.w)))
        self.bias = nn.Parameter(torch.randn((channel_out, )))
        
    def forward(self, x):
        self.x = x # cache for backwark
        C, H, W = x.shape
        C_out, H_out, W_out = self.channel_out, H-self.h+1, W-self.w+1
        out = torch.zeros(size=(C_out, H_out, W_out))
        for c in range(0, C_out):
            for i in range(0, H_out):
                for j in range(0, W_out):
                    val = 0
                    for c_in in range(self.channel_in):
                        for u in range(self.h):
                            for v in range(self.w):
                                val += self.weight[c, c_in, u, v] * x[c_in, u+i, v+j]
                    out[c, i, j] = val + self.bias[c]
        return out
    
    def forward_(self, x):
        self.x = x # cache for backwark
        C, H, W = x.shape
        C_out, H_out, W_out = self.channel_out, H-self.h+1, W-self.w+1
        out = torch.zeros(size=(C_out, H_out, W_out))
        for c in range(0, C_out):
            for i in range(0, H_out):
                for j in range(0, W_out):
                    out[c, i, j] = torch.sum(self.weight[c, :, :, :].squeeze(0) \
                        * x[:, i:i+self.h, j:j+self.w]) + self.bias[c]
        return out
    
    def backward(self, dout):
        # dout: (C_out, H_out, W_out):
        C_out, H_out, W_out = dout.shape
        
        db = torch.zeros_like(self.bias)
        dw = torch.zeros_like(self.weight)
        dx = torch.zeros_like(self.x)
        
        # compute db
        for i in range(self.channel_out):
            db[i] = torch.sum(dout[i])
            
        # compute dw
        for c in range(0, C_out):
            for c_in in range(self.channel_in):
                for u in range(0, self.h):
                    for v in range(0, self.w):
                        dval = 0
                        for i in range(H_out):
                            for j in range(W_out):
                                dval += dout[c, i, j] * self.x[c_in, u+i, v+j]
                        dw[c, c_in, u, v] = dval
        
        # compute dx
        for c in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    for c_in in range(self.channel_in):
                        for u in range(self.h):
                            for v in range(self.w):
                                dx[c_in, u+i, v+j] += dout[c, i, j] * self.weight[c, c_in, u, v]
        return dx, dw, db
    
    def backward_(self, dout):
        C_out, H_out, W_out = dout.shape
        
        dw = torch.zeros_like(self.weight)
        dx = torch.zeros_like(self.x)
        
        db = torch.sum(dout, dim=(-1, -2))
        for c in range(0, C_out):
            for c_in in range(self.channel_in):
                for u in range(0, self.h):
                    for v in range(0, self.w):
                        dw[c, c_in, u, v] = torch.sum(dout[c_in]*self.weight[c])
        
        for c in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    for c_in in range(self.channel_in):
                        for u in range(self.h):
                            for v in range(self.w):
                                dx[c_in, u+i, v+j] += dout[c, i, j] * self.weight[c, c_in, u, v]
                            
        return dx, dw, db
    

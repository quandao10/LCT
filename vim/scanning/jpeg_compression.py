import torch_dct as dct
import torch


def dct2(x):
    """2D Discrete Cosine Transform"""
    return torch.fft.fftn(x, dim=(-2, -1))

def idct2(x):
    """2D Inverse Discrete Cosine Transform"""
    return torch.fft.ifftn(x, dim=(-2, -1)).real
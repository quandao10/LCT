import numpy as np
import matplotlib.pyplot as plt
import torch as th
import math
import torch

def test_scale():
    total_steps = 210000
    start_scales = 10
    end_scales = 1280

    def improve_scale_fn(step):
        temp = np.floor(total_steps/(np.log2(np.floor(end_scales/start_scales))+1))
        scales = min(start_scales*2**np.floor(step/temp), end_scales) + 1
        return None, int(scales)

    scales = []
    for i in range(total_steps):
        _, scale = improve_scale_fn(i)
        scales.append(scale)
    plt.plot(scales)
    plt.savefig("scales.jpg")
    
test_scale()
    

def test_erf_dist():
    sigma_max = 80.
    sigma_min = 0.002
    rho = 7
    p_mean = -1.1
    p_std = 2.0
    def icm_dist(num_scales):
        indices = th.Tensor(range(num_scales))
        sigmas = sigma_max ** (1 / rho) + indices / (num_scales - 1) * (
                sigma_min ** (1 / rho) - sigma_max ** (1 / rho)
            )
        erf_sigmas = th.erf((th.log(sigmas)-p_mean)/(math.sqrt(2)*p_std))
        unnorm_prob = erf_sigmas[:-1]-erf_sigmas[1:]
        dist = th.distributions.categorical.Categorical(probs=unnorm_prob)
        return dist, erf_sigmas, unnorm_prob


def norm_dim(x):
    C, H, W = x.shape
    return torch.sqrt(torch.sum(x**2)/(C*H*W))


def test_scale_reweight():
    x = torch.randn((4, 32, 32))
    distances = []
    for _ in range(1000):
        n = torch.randn_like(x)
        distances.append(1/norm_dim(x-n))
    distances = torch.Tensor(distances)
    distances = (distances-distances.min())/(distances.max()-distances.min()) + distances.min()
    print(distances.min(), distances.max())
    print(norm_dim(x)**2)

def construct_constant_c(scales=[11, 21, 41, 81, 161, 321, 641], intial_c=0.0345):
    scale_dict = {}
    for scale in scales:
        scale_dict[scale] = torch.tensor(math.exp(-1.15 * math.log(float(scale - 1)) - 0.85))
    scale_dict[11] = intial_c
    return scale_dict

# print(construct_constant_c())
# from argparse import Namespace
# from diffusers.models import AutoencoderKL
# import torchvision
# args = Namespace(**{"dataset": "subset_imagenet_256", "repa_enc_info": "4:dinov2-vit-b", "datadir": "/common/users/qd66/repa/latent_imagenet256"})
# from datasets_prep import get_repa_dataset
# data = get_repa_dataset(args)
# vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to("cuda")
# image, _, label = data[4000]
# image = vae.decode(image.unsqueeze(0).to("cuda")).sample
# torchvision.utils.save_image(image, "./debug/test.jpg", normalize=True)
# print(label)
# print(data.label_to_classidx)
import torch.nn as nn


class GroupRMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, group=1):
        super().__init__()
        assert dim % group == 0, "dimension should be divisable the number of groups"
        self.eps = eps
        self.group = group
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        size = x.size()
        x = x.reshape(*size[:-1], self.group, self.dim//self.group)
        norm_term = torch.rsqrt(x.pow(2).mean(-1, keepdim=True)+self.eps)
        x  = x * norm_term
        return x.reshape(*size)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    
# x = torch.randn((2, 2, 256))
# x = GroupRMSNorm(256, group=2)(x)
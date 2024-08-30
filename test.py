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

# test_erf_dist()

# test_scale_reweight()

def test_ckpt():
    pass
import numpy as np
import matplotlib.pyplot as plt
import torch as th
import math

def test_scale():
    total_steps = 459000
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

test_erf_dist()
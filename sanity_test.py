import numpy as np
import torch.nn as nn
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
    
# test_scale()
    

def test_erf_dist():
    sigma_max = 80.
    sigma_min = 0.002
    rho = 9
    p_mean = -0.8
    p_std = 1.5
    def icm_dist(num_scales):
        indices = th.Tensor(range(num_scales))
        sigmas = (sigma_max ** (1 / rho) + indices / (num_scales - 1) * (
                sigma_min ** (1 / rho) - sigma_max ** (1 / rho)
            ))**rho
        erf_sigmas = th.erf((th.log(sigmas)-p_mean)/(math.sqrt(2)*p_std))
        unnorm_prob = erf_sigmas[:-1]-erf_sigmas[1:]
        dist = th.distributions.categorical.Categorical(probs=unnorm_prob)
        return dist, erf_sigmas, unnorm_prob, sigmas
    # plot icm_dist 
    dist, erf_sigmas, unnorm_prob, sigmas = icm_dist(513)
    samples = dist.sample(sample_shape=(10000,))
    sigmas = sigmas[samples]
    plt.figure()
    plt.hist(sigmas, bins=100)
    plt.savefig("samples.jpg")

# test_erf_dist()

def test_log_norm():
    p_mean = -0.8
    p_std = 1.5
    rnd_normal = torch.randn([10000,])
    t = (rnd_normal * p_std + p_mean).exp()
    plt.figure()
    plt.hist(t, bins=100)
    plt.savefig("t.jpg")
    
# test_log_norm()

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

def test_plot_weight():
    sigma_max = 80.
    sigma_min = 0.002
    rho = 7
    num_scales = 513
    indices = th.Tensor(range(num_scales))
    sigmas = (sigma_max ** (1 / rho) + indices / (num_scales - 1) * (
            sigma_min ** (1 / rho) - sigma_max ** (1 / rho)
        ))**rho
    weights = 1/(sigmas[:-1]-sigmas[1:])
    xs = sigmas[:-1]
    log_xs = torch.log(xs)
    # save to txt
    with open("weights.txt", "w") as f:
        for weight in weights:
            f.write(f"{weight}\n")
            
    with open("log_xs.txt", "w") as f:
        for x in log_xs:
            f.write(f"{x}\n")
    # weights = torch.min(weights, torch.tensor(20.))
    # plot line
    snr = 1/sigmas[:-1]**2
    # softmin_snr = 1/(sigmas**2 + 20**-1) + 1
    plt.plot(log_xs, weights, label="weights")
    # plt.plot(log_xs, snr, label="snr")
    # plt.figure()
    # plt.plot(x, label="x")
    plt.legend()
    plt.savefig("weights.jpg")
    plt.figure()
   
    
# test_plot_weight()

def weight_schedule(a = 0.9299, b = 0.9246):
    sigma_max = 80.
    sigma_min = 0.002
    rho = 7
    num_scales = 513
    indices = th.Tensor(range(num_scales))
    sigmas = (sigma_max ** (1 / rho) + indices / (num_scales - 1) * (
            sigma_min ** (1 / rho) - sigma_max ** (1 / rho)
        ))**rho
    indices = int(num_scales*0.70)
    print(sigmas[indices])
    # xs = sigmas[:-1]
    # log_xs = th.log(xs)
    # # print(log_xs)
    # print(math.log((sigma_max ** (1 / rho) + (num_scales-2) / (num_scales - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho)))**rho))
    # min_log = th.min(log_xs)
    # print(min_log)
    # shifted_log_x = log_xs - min_log
    # base = np.exp(-a * shifted_log_x) ** b
    # base = 19*base+1
    # plt.plot(torch.log(xs), base)
    # plt.savefig("exp_decay.jpg")
    # return base

# weights = weight_schedule()
def gaussian_fixed_sigma(x, k=6, h=1, sigma=2):
    return k * np.exp(- (x - h)**2 / (2 * sigma**2))

def monotonic_gaussian(x, k=6, h=1, sigma=2):
    bell = k * np.exp(- (x - h)**2 / (2 * sigma**2))
    bell[x < h] = k
    return bell

# Calculate Gaussian y values
x_new = np.linspace(-4.5, 6.5, 1000)
y_gaussian_fixed = gaussian_fixed_sigma(x_new)
print(y_gaussian_fixed.max(), y_gaussian_fixed.min())
y_monotonic_gaussian = monotonic_gaussian(x_new)
print(y_monotonic_gaussian.max(), y_monotonic_gaussian.min())
# Plot
plt.figure(figsize=(8, 6))
plt.plot(x_new, y_gaussian_fixed)
plt.plot(x_new, y_monotonic_gaussian)
plt.title('Gaussian Bell Shape: Peak 5, Sigma = 2')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("gaussian.jpg")
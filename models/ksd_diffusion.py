"""
Based on: https://github.com/crowsonkb/k-diffusion
"""
import numpy as np
import torch as th
import torch.nn.functional as F
from . import dist_util
import math
from .nn import mean_flat, append_dims, append_zero

def get_weightings(weight_schedule, snrs, sigma_data, next_t=-1e-4, t=0):
    if weight_schedule == "snr":
        weightings = snrs
    elif weight_schedule == "snr+1":
        weightings = snrs + 1
    elif weight_schedule == "karras":
        weightings = snrs + 1.0 / sigma_data**2
    elif weight_schedule == "truncated-snr":
        weightings = th.clamp(snrs, min=1.0)
    elif weight_schedule == "uniform":
        weightings = th.ones_like(snrs)
    ### ICT weighting
    elif weight_schedule == "ict":
        weightings = 1/(t-next_t)
    elif weight_schedule == "ict_trunc":
        weightings = 1/(t-next_t) * th.clamp(snrs, max=5.0, min=1.0)
    else:
        raise NotImplementedError()
    return weightings

def pairwise_distance(X, Y):
    """
    Compute pairwise Euclidean distances between rows of X and Y.

    Args:
        X: Tensor of shape (n, d)
        Y: Tensor of shape (m, d)

    Returns:
        distances: Tensor of shape (n, m) where distances[i, j] = ||X[i] - Y[j]||.
    """
    # Compute differences using broadcasting
    diff = X.unsqueeze(1) - Y.unsqueeze(0)  # shape: (n, m, d)
    distances = th.sqrt(th.sum(diff ** 2, dim=2) + 1e-8)  # (n, m)
    return distances

def energy_distance(X, Y):
    """
    Compute the energy distance between two distributions given samples X and Y.

    Energy distance is defined as:
        D(P, Q) = 2 E||X - Y|| - E||X - X'|| - E||Y - Y'||,
    where expectations are estimated over pairs of samples.

    Args:
        X: Tensor of shape (n, d) representing samples from distribution P.
        Y: Tensor of shape (m, d) representing samples from distribution Q.

    Returns:
        energy_dist: Scalar tensor representing the energy distance.
    """
    # Compute the average distance between samples from different distributions
    dXY = pairwise_distance(X, Y).mean()
    # Compute the average distance between samples within X
    dXX = pairwise_distance(X, X).mean()
    # Compute the average distance between samples within Y
    dYY = pairwise_distance(Y, Y).mean()
    
    # Combine them using the energy distance formula
    energy_dist = 2 * dXY - dXX - dYY
    return energy_dist

def compute_pairwise_differences(X):
    """
    Compute pairwise differences and squared Euclidean distances.
    
    Args:
        X: Tensor of shape (n, c, h, w)
        
    Returns:
        diff: Tensor of shape (n, n, d) where diff[i, j] = X[i] - X[j]
        sq_dists: Tensor of shape (n, n) with squared Euclidean distances.
    """
    # Expand dimensions to compute differences
    X1 = X.unsqueeze(1)  # (n, 1, d)
    X2 = X.unsqueeze(0)  # (1, n, d)
    diff = X1 - X2       # (n, n, d)
    sq_dists = th.sum(diff ** 2, dim=2)  # (n, n)
    return diff, sq_dists

def kernel_stein_group(samples, scores, kernel_type='imq'):
    N, K, _ = samples.shape
    loss = th.tensor(0.0, device=samples.device)
    for i in range(N):
        loss += kernel_stein_discrepancy(samples[i], scores[i], kernel_type=kernel_type)
    return loss/N


def kernel_stein_discrepancy(samples, scores, kernel_type='rbf', **kwargs):
    """
    Compute the Kernel Stein Discrepancy (KSD) for a set of samples using PyTorch.
    
    The Stein kernel u_P(x,y) is defined differently for each kernel:
    
    RBF kernel:
      k(x,y) = exp(-||x-y||^2/(2*h^2))
      
      u_P(x,y) = k(x,y) * [ s(x)^T s(y)
                           - (1/h^2)*( s(y)^T (x-y) - s(x)^T (x-y) )
                           + d/h^2 - ||x-y||^2/h^4 ]
    
    IMQ kernel:
      k(x,y) = (c + ||x-y||^2)^(-beta)
      
      u_P(x,y) = k(x,y) * [ s(x)^T s(y)
                           - (2*beta/(c+||x-y||^2))*( s(y)^T (x-y) - s(x)^T (x-y) )
                           + (2*beta*d)/(c+||x-y||^2)
                           - (4*beta*(beta+1)||x-y||^2)/(c+||x-y||^2)^2 ]
    
    Args:
        samples: Tensor of shape (n, d) containing the samples.
        score_fn: A function that takes a tensor of shape (n, d) and returns the score s_P(x) (tensor of shape (n, d)).
        kernel_type: Either 'rbf' or 'imq' to choose the kernel.
        kwargs: Additional keyword arguments for the kernel parameters.
            For 'rbf': h (bandwidth, default=1.0).
            For 'imq': c (default=1.0) and beta (default=0.5).
            
    Returns:
        ksd: Scalar tensor representing the KSD.
    """
    n, d = samples.shape    
    diff, sq_dists = compute_pairwise_differences(samples)  # diff: (n, n, d), sq_dists: (n, n)
    # print(samples)
    # print(scores)
    # print(sq_dists)
    
    # Compute score dot products: s(x)^T s(y) for each pair
    score_dot = th.matmul(scores, scores.t())  # (n, n)
    # print(score_dot)
    
    # Compute s(x)^T (x-y) and s(y)^T (x-y)
    # score_diff: for each pair (i,j): s(x_i)^T (x_i - x_j)
    score_diff = (scores.unsqueeze(1) * diff).sum(dim=2)  # (n, n)
    # score_diff_y: for each pair (i,j): s(x_j)^T (x_i - x_j)
    score_diff_y = (scores.unsqueeze(0) * diff).sum(dim=2)  # (n, n)
    
    # print(score_diff)
    # print(score_diff_y)
    
    if kernel_type == 'rbf':
        h = kwargs.get('h', 1.0)
        # RBF kernel: k(x,y) = exp(-||x-y||^2 / (2*h^2))
        K = th.exp(-sq_dists / (2 * h**2))
        # print(K)
        U = K * ( score_dot
                 - (1/h**2) * (score_diff_y - score_diff)
                 + (d / h**2)
                 - (sq_dists / h**4) )
    elif kernel_type == 'imq':
        c = kwargs.get('c', 1.0)
        beta = kwargs.get('beta', 0.5)
        # IMQ kernel: k(x,y) = (c + ||x-y||^2)^(-beta)
        denom = c + sq_dists
        K = denom.pow(-beta)
        U = K * ( score_dot - (2 * beta / denom) * (score_diff_y - score_diff) + (2 * beta * d) / denom - (4 * beta * (beta + 1) * sq_dists) / (denom**2) )
    else:
        raise ValueError("Unsupported kernel_type. Choose 'rbf' or 'imq'.")
    
    # Average over all pairs (including diagonal)
    ksd_sq = U.mean()
    if th.isnan(ksd_sq) or th.isinf(ksd_sq):
        print(K)
        print(samples)
        print(sq_dists)
        print(score_dot)
        print(score_diff_y, score_diff)
        exit()
    # Return the square root if ksd_sq is non-negative (numerically it should be)
    ksd = th.sqrt(ksd_sq) if ksd_sq >= 0 else th.tensor(float('nan'))
    return ksd


class KSD_Denoiser:
    def __init__(
        self,
        args,
        sigma_data: float = 0.5,
        rho=7.0,
    ):
        self.args = args
        self.sigma_data = sigma_data
        self.sigma_max = args.sigma_max
        self.sigma_min = args.sigma_min
        self.weight_schedule = args.weight_schedule
        self.rho = rho
        self.p_mean = -1.1
        self.p_std = 2.0
        self.use_repa = args.use_repa
        self.repa_timesteps = args.repa_timesteps
        
    def get_snr(self, sigmas):
        return sigmas**-2

    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma - self.sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in
    
    def icm_dist(self, num_scales):
        indices = th.Tensor(range(num_scales))
        sigmas = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
                self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
            )
        sigmas = sigmas**self.rho
        erf_sigmas = th.erf((th.log(sigmas)-self.p_mean)/(math.sqrt(2)*self.p_std))
        unnorm_prob = erf_sigmas[:-1]-erf_sigmas[1:]
        dist = th.distributions.categorical.Categorical(probs=unnorm_prob)
        return dist

    def ksd_losses(
        self,
        model,
        x_start,
        group_size,
        num_scales,
        model_kwargs=None,
        target_model=None,
        noise=None,
    ):
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)

        dims = x_start.ndim
        bs = x_start.shape[0]

        def denoise_fn(x, t):
            model_output, denoised, ssl_feat_pred = self.denoise(
                model, x, t, **model_kwargs
            )
            return denoised, ssl_feat_pred

        @th.no_grad()
        def target_denoise_fn(x, t):
            return self.denoise(target_model, x, t, **model_kwargs)[1]

        if self.args.noise_sampler == "ict":
            indices = self.icm_dist(num_scales).sample(sample_shape=(x_start.shape[0]//group_size,)).to(x_start.device)
        else:
            indices = th.randint(0, num_scales - 2, (x_start.shape[0]//group_size,), device=x_start.device)
        indices = indices.reshape(indices.size(0), 1).repeat(1, group_size).flatten()
        indices_Tm2 = th.randint(int(0.75*num_scales), num_scales - 1, (x_start.shape[0],), device=x_start.device)
        ### need check here since yang song using the difference scheduler compared to karras: dunno why ? Yang Song code differently from paper check carefully
        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        tp1 = self.sigma_max ** (1 / self.rho) + (indices + 1) / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho
        tp1 = tp1**self.rho
        Tm2 = indices_Tm2**self.rho
        # next_t < t, indices > indices + 1
        
        # perturb and feed through model
        x_t = x_start + noise * append_dims(t, dims) # x_t (t > 2)
        x_Tm2 = x_start + noise * append_dims(Tm2, dims) # x_2
        x_merge, _ = denoise_fn(th.cat([x_t, x_Tm2]), th.cat([t, Tm2]))
        x_0t, x_0Tm2 = x_merge.chunk(2, dim=0)
        
        # ksd loss
        x_tp1 = x_0t + (x_t-x_0t)/append_dims(t, dims) * append_dims(tp1, dims)
        x_0tp1 = target_denoise_fn(x_tp1.detach(), tp1).detach()
        score_x_tp1 = (x_tp1 - x_0tp1)/append_dims(tp1, dims)        
        ksd_loss = kernel_stein_group(x_tp1.reshape(bs//group_size, group_size,-1), score_x_tp1.reshape(bs//group_size, group_size, -1))
        # diff loss
        diff_loss = mean_flat((x_0Tm2 - x_start)**2)
        terms = {}
        terms["ksd_loss"] = ksd_loss * 0.1
        terms["diff_loss"] = diff_loss
        terms["t"] = t
        return terms


    def denoise(self, model, x_t, sigmas, **model_kwargs):
        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim)
            for x in self.get_scalings_for_boundary_condition(sigmas)
        ]
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        model_output, ssl_feat_pred = model(c_in * x_t, rescaled_t, **model_kwargs)
        denoised = c_out * model_output + c_skip * x_t
        return model_output, denoised, ssl_feat_pred


def karras_sample(
    diffusion,
    generator,
    model,
    shape,
    steps,
    progress=False,
    callback=None,
    model_kwargs=None,
    device=None,
    sigma_min=0.002,
    sigma_max=80,  # higher for highres?
    rho=7.0,
    sampler="heun",
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
    noise=None,
    ts=None,):
    if noise is None:
        x_T = generator.randn(*shape, device=device) * sigma_max
    else:
        x_T = noise

    sigmas = get_sigmas_karras(steps, sigma_min, sigma_max, rho, device=device)

    sample_fn = {
        "heun": sample_heun,
        "onestep": sample_onestep,
        "euler": sample_euler,
        "multistep": stochastic_iterative_sampler,
    }[sampler]

    if sampler in ["heun", "dpm"]:
        sampler_args = dict(
            s_churn=s_churn, s_tmin=s_tmin, s_tmax=s_tmax, s_noise=s_noise, generator=generator,
        )
    elif sampler == "multistep":
        sampler_args = dict(
            ts=ts, t_min=sigma_min, t_max=sigma_max, rho=diffusion.rho, steps=steps, generator=generator,
        )
    else:
        sampler_args = {}

    def denoiser(x_t, sigma):
        _, denoised, _ = diffusion.denoise(model, x_t, sigma, **model_kwargs)
        return denoised

    if sampler not in ["heun", "dpm", "multistep"] :
        x_0 = sample_fn(
            denoiser,
            x_T,
            sigmas,
            None,
            progress=progress,
            callback=callback,
            **sampler_args,
        )
    else:
        x_0 = sample_fn(
            denoiser,
            x_T,
            sigmas,
            progress=progress,
            callback=callback,
            **sampler_args,
        )
    return x_0


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = th.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)


def get_ancestral_step(sigma_from, sigma_to):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    sigma_up = (
        sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2
    ) ** 0.5
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up


@th.no_grad()
def sample_euler_ancestral(model, x, sigmas, generator, progress=False, callback=None):
    """Ancestral sampling with Euler method steps."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        denoised = model(x, sigmas[i] * s_in)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1])
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                    "denoised": denoised,
                }
            )
        d = to_d(x, sigmas[i], denoised)
        # Euler method
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        x = x + generator.randn_like(x) * sigma_up
    return x


@th.no_grad()
def sample_midpoint_ancestral(model, x, ts, generator, progress=False, callback=None):
    """Ancestral sampling with midpoint method steps."""
    s_in = x.new_ones([x.shape[0]])
    step_size = 1 / len(ts)
    if progress:
        from tqdm.auto import tqdm

        ts = tqdm(ts)

    for tn in ts:
        dn = model(x, tn * s_in)
        dn_2 = model(x + (step_size / 2) * dn, (tn + step_size / 2) * s_in)
        x = x + step_size * dn_2
        if callback is not None:
            callback({"x": x, "tn": tn, "dn": dn, "dn_2": dn_2})
    return x


@th.no_grad()
def sample_heun(
    denoiser,
    x,
    sigmas,
    generator,
    progress=False,
    callback=None,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )
        eps = generator.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = denoiser(x, sigma_hat * s_in)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigma_hat,
                    "denoised": denoised,
                }
            )
        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == 0:
            # Euler method
            x = x + d * dt
        else:
            # Heun's method
            x_2 = x + d * dt
            denoised_2 = denoiser(x_2, sigmas[i + 1] * s_in)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
    return x


@th.no_grad()
def sample_euler(
    denoiser,
    x,
    sigmas,
    generator,
    progress=True,
    callback=None,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm
        indices = tqdm(indices)

    for i in indices:
        sigma = sigmas[i]
        denoised = denoiser(x, sigma * s_in)
        d = to_d(x, sigma, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "denoised": denoised,
                }
            )
        dt = sigmas[i + 1] - sigma
        x = x + d * dt
    return x


@th.no_grad()
def sample_dpm(
    denoiser,
    x,
    sigmas,
    generator,
    progress=False,
    callback=None,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
):
    """A sampler inspired by DPM-Solver-2 and Algorithm 2 from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )
        eps = generator.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = denoiser(x, sigma_hat * s_in)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigma_hat,
                    "denoised": denoised,
                }
            )
        # Midpoint method, where the midpoint is chosen according to a rho=3 Karras schedule
        sigma_mid = ((sigma_hat ** (1 / 3) + sigmas[i + 1] ** (1 / 3)) / 2) ** 3
        dt_1 = sigma_mid - sigma_hat
        dt_2 = sigmas[i + 1] - sigma_hat
        x_2 = x + d * dt_1
        denoised_2 = denoiser(x_2, sigma_mid * s_in)
        d_2 = to_d(x_2, sigma_mid, denoised_2)
        x = x + d_2 * dt_2
    return x


@th.no_grad()
def sample_onestep(
    distiller,
    x,
    sigmas,
    generator,
    progress=False,
    callback=None,
):
    """Single-step generation from a distilled model."""
    s_in = x.new_ones([x.shape[0]])
    return distiller(x, sigmas[0] * s_in)


@th.no_grad()
def stochastic_iterative_sampler(
    distiller,
    x,
    sigmas,
    generator,
    ts,
    progress=False,
    callback=None,
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
):
    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])

    for i in range(len(ts) - 1):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        x0 = distiller(x, t * s_in)
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)
        x = x0 + generator.randn_like(x) * np.sqrt(next_t**2 - t_min**2)
    return x


@th.no_grad()
def iterative_colorization(
    distiller,
    images,
    x,
    ts,
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
    generator=None,
):
    def obtain_orthogonal_matrix():
        vector = np.asarray([0.2989, 0.5870, 0.1140])
        vector = vector / np.linalg.norm(vector)
        matrix = np.eye(3)
        matrix[:, 0] = vector
        matrix = np.linalg.qr(matrix)[0]
        if np.sum(matrix[:, 0]) < 0:
            matrix = -matrix
        return matrix

    Q = th.from_numpy(obtain_orthogonal_matrix()).to(dist_util.dev()).to(th.float32)
    mask = th.zeros(*x.shape[1:], device=dist_util.dev())
    mask[0, ...] = 1.0

    def replacement(x0, x1):
        x0 = th.einsum("bchw,cd->bdhw", x0, Q)
        x1 = th.einsum("bchw,cd->bdhw", x1, Q)

        x_mix = x0 * mask + x1 * (1.0 - mask)
        x_mix = th.einsum("bdhw,cd->bchw", x_mix, Q)
        return x_mix

    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])
    images = replacement(images, th.zeros_like(images))

    for i in range(len(ts) - 1):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        x0 = distiller(x, t * s_in)
        # x0 = th.clamp(x0, -1.0, 1.0)
        x0 = replacement(images, x0)
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)
        x = x0 + generator.randn_like(x) * np.sqrt(next_t**2 - t_min**2)

    return x, images


@th.no_grad()
def iterative_inpainting(
    distiller,
    images,
    x,
    ts,
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
    generator=None,
):
    from PIL import Image, ImageDraw, ImageFont

    image_size = x.shape[-1]

    # create a blank image with a white background
    img = Image.new("RGB", (image_size, image_size), color="white")

    # get a drawing context for the image
    draw = ImageDraw.Draw(img)

    # load a font
    font = ImageFont.truetype("arial.ttf", 250)

    # draw the letter "C" in black
    draw.text((50, 0), "S", font=font, fill=(0, 0, 0))

    # convert the image to a numpy array
    img_np = np.array(img)
    img_np = img_np.transpose(2, 0, 1)
    img_th = th.from_numpy(img_np).to(dist_util.dev())

    mask = th.zeros(*x.shape, device=dist_util.dev())
    mask = mask.reshape(-1, 7, 3, image_size, image_size)

    mask[::2, :, img_th > 0.5] = 1.0
    mask[1::2, :, img_th < 0.5] = 1.0
    mask = mask.reshape(-1, 3, image_size, image_size)

    def replacement(x0, x1):
        x_mix = x0 * mask + x1 * (1 - mask)
        return x_mix

    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])
    images = replacement(images, -th.ones_like(images))

    for i in range(len(ts) - 1):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        x0 = distiller(x, t * s_in)
        # x0 = th.clamp(x0, -1.0, 1.0)
        x0 = replacement(images, x0)
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)
        x = x0 + generator.randn_like(x) * np.sqrt(next_t**2 - t_min**2)

    return x, images


@th.no_grad()
def iterative_superres(
    distiller,
    images,
    x,
    ts,
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
    generator=None,
):
    patch_size = 8

    def obtain_orthogonal_matrix():
        vector = np.asarray([1] * patch_size**2)
        vector = vector / np.linalg.norm(vector)
        matrix = np.eye(patch_size**2)
        matrix[:, 0] = vector
        matrix = np.linalg.qr(matrix)[0]
        if np.sum(matrix[:, 0]) < 0:
            matrix = -matrix
        return matrix

    Q = th.from_numpy(obtain_orthogonal_matrix()).to(dist_util.dev()).to(th.float32)

    image_size = x.shape[-1]

    def replacement(x0, x1):
        x0_flatten = (
            x0.reshape(-1, 3, image_size, image_size)
            .reshape(
                -1,
                3,
                image_size // patch_size,
                patch_size,
                image_size // patch_size,
                patch_size,
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(-1, 3, image_size**2 // patch_size**2, patch_size**2)
        )
        x1_flatten = (
            x1.reshape(-1, 3, image_size, image_size)
            .reshape(
                -1,
                3,
                image_size // patch_size,
                patch_size,
                image_size // patch_size,
                patch_size,
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(-1, 3, image_size**2 // patch_size**2, patch_size**2)
        )
        x0 = th.einsum("bcnd,de->bcne", x0_flatten, Q)
        x1 = th.einsum("bcnd,de->bcne", x1_flatten, Q)
        x_mix = x0.new_zeros(x0.shape)
        x_mix[..., 0] = x0[..., 0]
        x_mix[..., 1:] = x1[..., 1:]
        x_mix = th.einsum("bcne,de->bcnd", x_mix, Q)
        x_mix = (
            x_mix.reshape(
                -1,
                3,
                image_size // patch_size,
                image_size // patch_size,
                patch_size,
                patch_size,
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(-1, 3, image_size, image_size)
        )
        return x_mix

    def average_image_patches(x):
        x_flatten = (
            x.reshape(-1, 3, image_size, image_size)
            .reshape(
                -1,
                3,
                image_size // patch_size,
                patch_size,
                image_size // patch_size,
                patch_size,
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(-1, 3, image_size**2 // patch_size**2, patch_size**2)
        )
        x_flatten[..., :] = x_flatten.mean(dim=-1, keepdim=True)
        return (
            x_flatten.reshape(
                -1,
                3,
                image_size // patch_size,
                image_size // patch_size,
                patch_size,
                patch_size,
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(-1, 3, image_size, image_size)
        )

    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])
    images = average_image_patches(images)

    for i in range(len(ts) - 1):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        x0 = distiller(x, t * s_in)
        # x0 = th.clamp(x0, -1.0, 1.0)
        x0 = replacement(images, x0)
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)
        x = x0 + generator.randn_like(x) * np.sqrt(next_t**2 - t_min**2)

    return x, images

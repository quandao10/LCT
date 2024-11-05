import torch
import torchvision
from models.script_util import create_model_and_diffusion
from argparse import Namespace
from models.nn import mean_flat, append_dims, append_zero

device=3
args = dict(
    image_size=32,
    num_in_channels=4,
    num_classes=0,
    learn_sigma=False,
    model_type="DiT-B/2",
    no_scale=False,
    wo_norm=True,
    flash=False,
    use_scale_residual=False,
    linear_act="relu",
    sigma_max=80.0,
    sigma_min=0.002,
    weight_schedule="ict",
    loss_norm="cauchy",
    rho=7,
)
args = Namespace(**args)
model, diffusion = create_model_and_diffusion(args)
checkpoint = torch.load("/research/cbim/medical/qd66/lct_v2/latent_celeb256/celeb_dit_best_setting_700ep_B_relu_eps1e-5_unnormalized/checkpoints/nan_ckpt.pth", 
                        map_location=torch.device(f'cuda:{device}'))
device = torch.device(f'cuda:{device}')
model.load_state_dict(checkpoint["model"])
model = model.to(device)
x = checkpoint["x"]
n = checkpoint["n"]
num_scales=641
indices = torch.randint(0, num_scales - 1, (x.shape[0],), device=x.device)
t = args.sigma_max ** (1 / args.rho) + indices / (num_scales - 1) * (
    args.sigma_min ** (1 / args.rho) - args.sigma_max ** (1 / args.rho)
)
t = t**args.rho
x_t = x + n * append_dims(t, x.ndim)
# out = model(x_t, t)
print(model.x_embedder.proj.weight)
print(model.t_embedder.mlp[0].weight)
# print(out)
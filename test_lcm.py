from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, LCMScheduler
import torch
from PIL import Image

unet = UNet2DConditionModel.from_pretrained(
    "latent-consistency/lcm-sdxl",
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", unet=unet, torch_dtype=torch.float16, variant="fp16",
).to("cuda")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
num_inference_steps = 8
prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"
generator = torch.manual_seed(0)
image = pipe(
    prompt=prompt, num_inference_steps=num_inference_steps, generator=generator, guidance_scale=8.0
).images[0]
# print(image.shape)
image.save(f"test_{num_inference_steps}.png")
from diffusers import AutoencoderKL
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

# Encode and decode image

device = "cuda:0"
vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)

image_path = "/lustre/scratch/client/movian/research/users/khanhdn10/datasets/celeba_256_png/img00000170.png"
image = Image.open(image_path)
image = image.convert("RGB")
image = transforms.Resize((256, 256))(image)
image = transforms.ToTensor()(image).unsqueeze(0)
latent = vae.encode(image.to(dtype=vae.dtype)).latent_dist.sample()
image = vae.decode(latent).sample.float()
save_image(image, "results/test_encode_decode.jpg", nrow=8, padding=2, normalize=True)
from diffusers import DDPMPipeline
import torch

# Load pretrained DDPM
pipe = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32")

# Create your custom noise
custom_noise = torch.randn((1, 3, 32, 32), device="cpu")

# Manually run the reverse diffusion process
image = custom_noise
for t in reversed(range(0,1000)):
    with torch.no_grad():
        noise_pred = pipe.unet(image, t).sample
    image = pipe.scheduler.step(noise_pred, t, image).prev_sample
    if t % 50 == 0:
        save_image = image.clamp(-1, 1)  # optional, ensures valid range

# Convert to [0, 255] uint8
        save_image = (save_image / 2 + 0.5).clamp(0, 1)  # normalize to [0, 1]
        save_image = save_image.mul(255).byte()

# Reorder dimensions: (B, C, H, W) → (B, H, W, C)
        save_image = save_image.permute(0, 2, 3, 1)

# Convert to numpy and PIL
        save_image = save_image.cpu().numpy()
        pil_images = pipe.numpy_to_pil(save_image)
        pil_images[0].save("custom_noise_result.png")

# Convert to PIL image
image = image.clamp(-1, 1)  # optional, ensures valid range

# Convert to [0, 255] uint8
image = (image / 2 + 0.5).clamp(0, 1)  # normalize to [0, 1]
image = image.mul(255).byte()

# Reorder dimensions: (B, C, H, W) → (B, H, W, C)
image = image.permute(0, 2, 3, 1)

# Convert to numpy and PIL
image = image.cpu().numpy()
pil_images = pipe.numpy_to_pil(image)
pil_images[0].save("custom_noise_result.png")

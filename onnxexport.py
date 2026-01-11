import torch
import numpy as np


from fastgan_models import FastGenerator



def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)

latent_dim = 256
image_size = 256
model_path = "./models/FastGAN/all_45000.pth"

device = torch.device("cpu")

generator = FastGenerator(ngf=64,nz=256,nc=3, im_size=image_size)
state_dict = torch.load(model_path, map_location=device)
#state_dict["g_ema"] = {k.replace('module.', ''): v for k, v in state_dict["g_ema"].items()}
#generator.load_state_dict(state_dict["g_ema"])

load_params(generator, state_dict["g_ema"])

noise = torch.randn(1,latent_dim,1,1)

with torch.no_grad():
    for _ in range(50):
        _ = generator(torch.randn(1, 256, 1, 1).to(device))

generator = generator.eval()



torch.onnx.export(
    generator,
    noise,
    "onnx/fastgan.onnx",
    input_names=["z"],
    output_names=["image"],
)

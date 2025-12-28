import torch
import numpy as np


from training.models import Generator




latent_dim = 256
image_size = 512
model = Generator(4, latent_dim, image_size, 64,3,6)
state_dict = torch.load("models/512,512,3/amazing_logos.pth", map_location=torch.device("cpu"))

noise = torch.randn(1,latent_dim,1,1)
alpha = torch.tensor(1.0)


model = model.eval()

for m in model.modules():
    if isinstance(m, torch.nn.BatchNorm2d):
        m.track_running_stats = False


torch.onnx.export(
    model,
    (noise,alpha),
    "logos.onnx",
    input_names=["z","alpha"],
    output_names=["image"],
    opset_version=19,
    do_constant_folding=True,
)

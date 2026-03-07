import torch
from tqdm import tqdm


from fastgan_models import FastGenerator
from training.models import Generator


model = "custom"

def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)

latent_dim = 256
image_size = 512
model_path = "./models/512,512,3/amazing_logos.pth"

device = torch.device("cpu")

if model == "fastgan":
    generator = FastGenerator(ngf=64,nz=256,nc=3, im_size=image_size)
    state_dict = torch.load(model_path, map_location=device)
    load_params(generator, state_dict["g_ema"])
    print(f"✅ Loaded generator from {model_path}")

layer = 6
if model == "custom":
    generator = Generator(4, latent_dim, image_size, 64,3,layer)
    state_dict = torch.load(model_path, map_location=device)
    generator.load_state_dict(state_dict)
    print(f"✅ Loaded generator from {model_path}")


noise = torch.randn(1,latent_dim,1,1)

with torch.no_grad():
    for _ in tqdm(range(50),desc="Generating to fill up Batch Norm"):
        _ = generator(torch.randn(1, 256, 1, 1).to(device))

generator = generator.eval()



torch.onnx.export(
    generator,
    noise,
    "onnx/amazing_logos.onnx",
    export_params=True,
    opset_version=19,       # latest ONNX opset recommended
    do_constant_folding=True, # optimize constants
    input_names=['z'],
    output_names=['image'],
    dynamo=True
)


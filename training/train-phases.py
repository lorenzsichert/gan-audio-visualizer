from torch import mean, optim, torch
import torch.nn as nn
from torch.utils.data import Dataset, dataset
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms
from torchvision.utils import save_image

from models import Generator
from models import Discriminator

# Use as many threads as possible
torch.set_num_threads(20)
torch.set_num_interop_threads(20)

n_epochs = 2000
b1 = 0.5
b2 = 0.99
latent_dim = 256
features = 64
init_size = 4
img_size = 512
layer = 1
channels = 1
batch_size = 32
dataset_size = -1
sample_interval = 64


alpha_end = 2.0
alpha_incease = 0.0001
alpha_dropdown = 1.0
counting_alpha = 0.0


# --- Dataset Loading ---
link = "iamkaikai/amazing_logos_v2"
split = "train"
image_tag = "image"


class DatasetTransform(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return self.dataset.num_rows

    def __getitem__(self, index):
        item = self.dataset[index][image_tag]
        if self.transform:
            item = self.transform(item)
        return item

def convert_to_rgb(x):
    return x.convert("RGB")


transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.Lambda(convert_to_rgb),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])

ds = load_dataset(link)
train = ds[split]
dataset = DatasetTransform(train, transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# --- Cuda Init ---
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Running on {device}")



# --- Image Blending ---

def seperate_image(image, layer, alpha):
    image_normal = nn.functional.interpolate(image, size=(pow(2, layer+2), pow(2,layer+2)), mode="bilinear")
    image_normal_high = nn.functional.interpolate(image_normal, size=(pow(2, layer+3), pow(2,layer+3)), mode="bilinear")
    image_high = nn.functional.interpolate(image, size=(pow(2, layer+3), pow(2,layer+3)), mode="bilinear")

    image_blend_high = (1-alpha) * image_normal_high + alpha * image_high
    image_blend_normal = nn.functional.interpolate(image_blend_high, size=(pow(2, layer+2), pow(2,layer+2)), mode="bilinear")
    return image_blend_normal, image_blend_high


generator = Generator(init_size, latent_dim, img_size, features, channels, layer)
discriminator = Discriminator(init_size, img_size, features, channels, layer)

try:
    generator.load_state_dict(torch.load(f"G-{layer}-a1.405.pth"))
    discriminator.load_state_dict(torch.load(f"D-{layer}-a1.405.pth"))
except:
    print("Models could not be loaded!")


generator.to(device)
discriminator.to(device)


optimizerG = optim.Adam(generator.parameters(), lr=0.001, betas=[0.0, 0.9])
optimizerD = optim.Adam(discriminator.parameters(), lr=0.001, betas=[0.0, 0.9])


alpha = 1.0


for ep in range(n_epochs):
    print(f"Epoch {ep}:")


    i = 0
    for batch in dataloader:
        counting_alpha += alpha_incease
        if (counting_alpha >= alpha_end and layer <= 5):
            counting_alpha = 0.0
            alpha_incease *= alpha_dropdown
            torch.save(generator.state_dict(), f"G-{layer}.pth")
            torch.save(discriminator.state_dict(), f"D-{layer}.pth")
            layer += 1
            generator.add_layer(layer)
            discriminator.add_layer(layer)

            torch.save(generator.state_dict(), f"G-{layer}.pth")
            torch.save(discriminator.state_dict(), f"D-{layer}.pth")

            generator.to(device)
            discriminator.to(device)

            optimizerG = optim.Adam(generator.parameters(), lr=0.001, betas=[0.0, 0.9])
            optimizerD = optim.Adam(discriminator.parameters(), lr=0.001, betas=[0.0, 0.9])
        alpha = min(max(0.0,counting_alpha),1.0)
        i += 1
        discriminator.zero_grad()

        # Train Discriminator on Real Images
        real = batch.to(device)

        real_normal, real_high = seperate_image(real, layer, alpha)

        output_real = discriminator(real_normal, real_high, alpha)
        loss_real = mean(nn.functional.relu(1 - output_real))
        loss_real.backward()

        
        # Train Discriminator on Fake Images
        noise = torch.randn(batch_size, latent_dim, 1, 1).to(device)
        fake = generator(noise, alpha)
        fake_normal, fake_high = seperate_image(fake, layer, alpha)
        output_fake = discriminator(fake_normal, fake_high, alpha) 
        loss_fake = mean(nn.functional.relu(1 + output_fake))
        loss_fake.backward()
        optimizerD.step()
 

        # Train Generator with Discriminator
        generator.zero_grad()
        noise = torch.randn(batch_size, latent_dim, 1, 1).to(device)
        output = generator(noise, alpha)
        output_normal, output_high = seperate_image(output, layer, alpha)
        output_fake = discriminator(output_normal, output_high, alpha) 
        loss_generated = -mean(output_fake)
        loss_generated.backward()
        optimizerG.step()



        if i % 16 == 0:
            print(f"Ep: {ep}, i: {i}/{len(dataloader)}, alpha: {counting_alpha:.3f}, D(r): {mean(output_real):.3f}, D(f): {mean(output_fake):.3f}, D Loss: {(loss_real + loss_fake)/2:.3f}, G Loss:  {loss_generated:.3f}")
        if i % sample_interval == 0:
            save_image(output, f"image-{ep}.png", normalize=True)
        if i % (sample_interval * 16) == 0:
            torch.save(generator.state_dict(), f"G-{layer}-a{counting_alpha:.3f}.pth")
            torch.save(discriminator.state_dict(), f"D-{layer}-a{counting_alpha:.3f}.pth")


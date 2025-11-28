from numpy import log2
from torch import mean, optim, torch
import torch.nn as nn
from torch.utils.data import Dataset, dataset
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms
from torchvision.utils import save_image


n_epochs = 2000
b1 = 0.5
b2 = 0.99
latent_dim = 100
features = 64
init_size = 4
img_size = 128
layer = 1
channels = 3
batch_size = 16
dataset_size = -1
sample_interval = 64


alpha_end = 1.5
alpha_incease = 0.0002
alpha_dropdown = 0.25


# --- Dataset Loading ---
link = "uoft-cs/cifar10"
split = "train"
image_tag = "img"


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

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])

ds = load_dataset(link)
train = ds[split]
dataset = DatasetTransform(train, transform)
dataloader = DataLoader(dataset, batch_size=batch_size)


# --- Cuda Init ---
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Running on {device}")

class Generator(nn.Module):
    def __init__(self, init_size, latent_dim, img_size, features, channels, layer):
        super(Generator, self).__init__()
        self.init_size = init_size
        self.features = features
        self.channels = channels
        self.layer = layer

        self.upscales = log2(img_size/self.init_size)
        
        print(self.upscales)


        self.network = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, self.init_size * features, kernel_size=4, stride=1, padding=0),
            nn.InstanceNorm2d(self.init_size * features // 1),
            nn.ReLU(inplace=True)
        )
        for i in range(1,layer):
            self.network.extend(nn.Sequential(
                nn.ConvTranspose2d(self.init_size * features // pow(2, i-1), self.init_size * features // pow(2, i), kernel_size=4, stride=1, padding=0),
                nn.InstanceNorm2d(self.init_size * features // pow(2, i)),
                nn.ReLU(inplace=True)
            ))


        self.normal_res_head = nn.Sequential(
            nn.ConvTranspose2d(self.init_size * features // pow(2, layer-1), 3, kernel_size=4, stride=2, padding=1)
        )

        self.high_res_block = nn.Sequential(
            nn.ConvTranspose2d(self.init_size * features // pow(2, layer-1), self.init_size * features // pow(2, layer), kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(self.init_size * features // pow(2, layer)),
            nn.ReLU(inplace=True)
        )

        self.high_res_head = nn.Sequential(
            nn.ConvTranspose2d(self.init_size * features // pow(2, layer), 3, kernel_size=4, stride=2, padding=1)
        )

        print(self.network)

    def forward(self, input, alpha):
        out = self.network(input)
        image_normal = self.normal_res_head(out)
        out_high = self.high_res_block(out)
        image_high = self.high_res_head(out_high)

        image_normal_high = nn.functional.interpolate(image_normal, size=image_high.size()[-2:],mode="bilinear", align_corners=False)
        image = (1-alpha) * image_normal_high + alpha * image_high

        return image

    def add_layer(self, layer):
        self.layer = layer
        self.network.extend(self.high_res_block)
        self.normal_res_head = self.high_res_head
        self.high_res_block = nn.Sequential(
            nn.ConvTranspose2d(self.init_size * features // pow(2, self.layer-1), self.init_size * features // pow(2, self.layer), kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(self.init_size * features // pow(2, self.layer)),
            nn.ReLU(inplace=True)
        )
        self.high_res_head = nn.Sequential(
            nn.ConvTranspose2d(self.init_size * features // pow(2, self.layer), 3, kernel_size=4, stride=2, padding=1)
        )
        print(self.network)

class Discriminator(nn.Module):
    def __init__(self, init_size, img_size, features, channels, layer):
        super(Discriminator, self).__init__()
        self.init_size = init_size
        self.features = features
        self.channels = channels
        self.layer = layer

        self.downscales = log2(img_size/self.init_size)

        print(self.downscales)


        self.network = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(self.init_size * features // pow(2, self.layer-1), 1, kernel_size=4, stride=1, padding=0)),
            nn.LeakyReLU(0.2, True),
        )

        for i in range(1,layer):
            self.network.extend(nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(self.init_size * features // pow(2, i), self.init_size * features // pow(2, i-1), kernel_size=4, stride=1, padding=0)),
                nn.LeakyReLU(0.2, True),
            ))

        self.head_normal = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(3, self.init_size * features // pow(2, self.layer-1), kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, True),
        )

        self.head_high = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(3, self.init_size * features // pow(2, self.layer), kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, True)
        )
        self.body_high = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(self.init_size * features // pow(2, self.layer), self.init_size * features // pow(2, self.layer-1), kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, True),
        )

        print(self.network)
    
    def forward(self, input_normal, input_high, alpha):
        pass_high = self.body_high(self.head_high(input_high)) # Downscale once
        pass_normal = self.head_normal(input_normal)
        input = (1 - alpha) * pass_normal + alpha * pass_high

        return self.network(input)

    def add_layer(self, layer):
        self.layer = layer
        self.network.insert(0, self.body_high)
        self.head_normal = self.head_high

        self.head_high = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(3, self.init_size * features // pow(2, self.layer), kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, True)
        )
        self.body_high = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(self.init_size * features // pow(2, self.layer), self.init_size * features // pow(2, self.layer-1), kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, True),
        )
        print(self.network)


# --- Image Blending ---

def seperate_image(image, layer, alpha):
    image_normal = nn.functional.interpolate(image, size=(pow(2, layer+2), pow(2,layer+2)), mode="bilinear")
    image_normal_high = nn.functional.interpolate(image_normal, size=(pow(2, layer+3), pow(2,layer+3)), mode="bilinear")
    image_high = nn.functional.interpolate(image, size=(pow(2, layer+3), pow(2,layer+3)), mode="bilinear")

    image_blend_high = (1-alpha) * image_normal_high + alpha * image_high
    image_blend_normal = nn.functional.interpolate(image_blend_high, size=(pow(2, layer+2), pow(2,layer+2)), mode="bilinear")
    return image_blend_normal, image_blend_high

layer = 1

generator = Generator(init_size, latent_dim, img_size, features, channels, layer)
discriminator = Discriminator(init_size, img_size, features, channels, layer)

generator.to(device)
discriminator.to(device)


optimizerG = optim.Adam(generator.parameters(), lr=0.001, betas=[0.0, 0.9])
optimizerD = optim.Adam(discriminator.parameters(), lr=0.001, betas=[0.0, 0.9])


alpha = 1.0
counting_alpha = 0.0



for ep in range(n_epochs):
    print(f"Epoch {ep}:")


    i = 0
    for batch in dataloader:
        counting_alpha += alpha_incease
        if (counting_alpha >= alpha_end and layer <= 4):
            counting_alpha = 0.0
            alpha_incease *= alpha_dropdown
            layer += 1
            generator.add_layer(layer)
            discriminator.add_layer(layer)

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
        noise = torch.randn(batch_size, 100, 1, 1).to(device)
        fake = generator(noise, alpha)
        fake_normal, fake_high = seperate_image(fake, layer, alpha)
        output_fake = discriminator(fake_normal, fake_high, alpha) 
        loss_fake = mean(nn.functional.relu(1 + output_fake))
        loss_fake.backward()
        optimizerD.step()
 

        # Train Generator with Discriminator
        generator.zero_grad()
        noise = torch.randn(batch_size, 100, 1, 1).to(device)
        output = generator(noise, alpha)
        output_normal, output_high = seperate_image(output, layer, alpha)
        output_fake = discriminator(output_normal, output_high, alpha) 
        loss_generated = -mean(output_fake)
        loss_generated.backward()
        optimizerG.step()



        if i % 16 == 0:
            print(f"Ep: {ep}, i: {i}/{len(dataloader)}, alpha: {alpha}, D(r): {mean(output_real):.3f}, D(f): {mean(output_fake):.3f}, D Loss: {(loss_real + loss_fake)/2:.3f}, G Loss:  {loss_generated:.3f}")
        if i % sample_interval == 0:
            save_image(output, f"image-{ep}.png", normalize=True)


import os
import numpy as np

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torchvision import datasets

import torch.nn as nn
import torch
from datasets import load_dataset
from PIL import Image




opt = {
    "n_epochs": 20000,
    "batch_size": 1,
    "lr": 0.0002,
    "b1": 0.5,
    "b2": 0.999,
    "n_cpu": 8,
    "latent_dim": 100,
    "img_size": 32,
    "channels": 3,
    "sample_interval": 100,
    "dataset_size": 100,
}


cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda') if cuda else torch.device('cpu')


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self, img_size, latent_dim, feature_g=64, channels=3):
        super(Generator, self).__init__()

        self.init_size = 8
        self.num_upsamples = int( np.log2(img_size // self.init_size ))

        in_ch = feature_g * 2 ** self.num_upsamples

        layers = []

        self.fc = [
            nn.ConvTranspose2d(latent_dim, in_ch, 4, 1, 0, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(True)
        ]

        def convolutional_block(in_sample, out_sample, stride, padding):
            return [
                nn.ConvTranspose2d(in_sample, out_sample, kernel_size=4, stride=stride, padding=padding),
                nn.BatchNorm2d(out_sample),
                nn.ReLU(True),
            ]

        layers += convolutional_block(latent_dim, in_ch, stride=1, padding=0)

        for _ in range(self.num_upsamples):
            out_ch = in_ch // 2 if in_ch > feature_g else feature_g
            layers += convolutional_block(in_ch, out_ch, stride=2, padding=1)
            in_ch = out_ch


        layers += [
            nn.ConvTranspose2d(in_ch, channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        ]

        self.net = nn.Sequential(*layers)

        print(self.net)

    def forward(self, z):
        img = self.net(z)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_size, feature_d=64, channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, 2, 1, bias=False)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []

        in_ch = channels
        out_ch = feature_d

        num_upsamples = int(np.log2(img_size // 8)) + 1

        for i in range(num_upsamples):
            layers += discriminator_block(in_ch, out_ch, normalize=(i != 0))
            in_ch = out_ch
            out_ch = min(out_ch * 2, feature_d * 16)

        layers.append(nn.Conv2d(in_ch, 1, 4, 1, 0))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)
        print(self.net)

    def forward(self, img):
        return self.net(img)


# Loss function
criterion = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator(opt["img_size"], opt["latent_dim"], 64, channels=opt["channels"])
discriminator = Discriminator(opt["img_size"], 64, channels=opt["channels"])

if cuda:
    generator.cuda()
    discriminator.cuda()
    criterion.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Optimizers
optimizerG = torch.optim.Adam(generator.parameters(), lr=opt["lr"], betas=(opt["b1"], opt["b2"]))
optimizerD = torch.optim.Adam(discriminator.parameters(), lr=opt["lr"], betas=(opt["b1"], opt["b2"]))

fake_label = 0
real_label = 1

fixed_noise = torch.randn(64, opt["latent_dim"], 1, 1, device=device)

# --- Load dataset ---
dataset = load_dataset("uoft-cs/cifar10")["train"]

save_dir = "data"  # ImageFolder expects at least one subfolder (class name)
class_name = "cifar_10"
class_dir = os.path.join(save_dir, class_name)
os.makedirs(class_dir, exist_ok=True)

# Save images to disk
for idx, example in enumerate(dataset):
    break
    img = example["img"]
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    img.save(os.path.join(class_dir, f"{idx}.png"))
print("img saved")

# Set up transforms
if opt["channels"] == 1:
  transform = transforms.Compose([
      transforms.Resize((opt["img_size"],opt["img_size"])),
      transforms.Grayscale(),
      transforms.ToTensor(),
      transforms.Normalize([0.5], [0.5])
  ])
else:
  transform = transforms.Compose([
      transforms.Resize((opt["img_size"],opt["img_size"])),
      transforms.ToTensor(),
      transforms.Normalize([0.5], [0.5])
  ])

# Create ImageFolder DataLoader
dataloader = torch.utils.data.DataLoader(
    datasets.ImageFolder(
        root=save_dir,
        transform=transform,
    ),
    batch_size=opt["batch_size"],
    shuffle=True,
)



# ----------
#  Training
# ----------
iters = 0

print("Starting Training Loop...")
print(f"Running on Device: {device}")
for epoch in range(opt["n_epochs"]):
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################

        ## Train with all-real batch
        discriminator.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = discriminator(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, opt["latent_dim"], 1, 1, device=device)
        # Generate fake image batch with G
        fake = generator(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = discriminator(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        generator.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = discriminator(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, opt["n_epochs"], i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            torch.save(generator.state_dict(), f'generator-{epoch}.pth')

        # Save Losses for plotting later

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == opt["n_epochs"]-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = generator(fixed_noise).detach().cpu()
                save_image(fake, f"image-{epoch}.png", normalize=True)

        iters += 1

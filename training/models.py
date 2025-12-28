import torch.nn as nn
from numpy import log2

class Generator(nn.Module):
    def __init__(self, init_size, latent_dim, img_size, features, channels, layer):
        super(Generator, self).__init__()
        self.init_size = init_size
        self.features = features
        self.channels = channels
        self.layer = layer

        self.upscales = log2(img_size/self.init_size)
        


        self.network = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, self.init_size * features, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(self.init_size * features // 1),
            nn.ReLU(inplace=True)
        )


        self.normal_res_head = nn.Sequential(
            nn.ConvTranspose2d(self.init_size * features // pow(2, 0), self.channels, kernel_size=4, stride=2, padding=1)
        )

        self.high_res_block = nn.Sequential(
            nn.ConvTranspose2d(self.init_size * features // pow(2, 0), self.init_size * features // pow(2, 1), kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.init_size * features // pow(2, 1)),
            nn.ReLU(inplace=True)
        )

        self.high_res_head = nn.Sequential(
            nn.ConvTranspose2d(self.init_size * features // pow(2, 1), self.channels, kernel_size=4, stride=2, padding=1)
        )

        for i in range(2,layer+1):
            self.add_layer(i)


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
            nn.ConvTranspose2d(self.init_size * self.features // pow(2, self.layer-1), self.init_size * self.features // pow(2, self.layer), kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.init_size * self.features // pow(2, self.layer)),
            nn.ReLU(inplace=True)
        )
        self.high_res_head = nn.Sequential(
            nn.ConvTranspose2d(self.init_size * self.features // pow(2, self.layer), self.channels, kernel_size=4, stride=2, padding=1)
        )

    def export_generative(self):
        final = self.network
        final.extend(self.high_res_block)
        final.extend(self.high_res_head)

        return final

class Discriminator(nn.Module):
    def __init__(self, init_size, img_size, features, channels, layer):
        super(Discriminator, self).__init__()
        self.init_size = init_size
        self.features = features
        self.channels = channels
        self.layer = layer

        self.downscales = log2(img_size/self.init_size)



        self.network = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(self.init_size * self.features // pow(2, 0), 1, kernel_size=4, stride=1, padding=0)),
            nn.LeakyReLU(0.2, True),
        )

        self.head_normal = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(self.channels, self.init_size * self.features // pow(2, 0), kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, True),
        )

        self.head_high = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(self.channels, self.init_size * self.features // pow(2, 1), kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, True)
        )
        self.body_high = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(self.init_size * self.features // pow(2, 1), self.init_size * self.features // pow(2, 0), kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, True),
        )
        for i in range(2,layer+1):
            self.add_layer(i)

    
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
            nn.utils.spectral_norm(nn.Conv2d(self.channels, self.init_size * self.features // pow(2, self.layer), kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, True)
        )
        self.body_high = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(self.init_size * self.features // pow(2, self.layer), self.init_size * self.features // pow(2, self.layer-1), kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, True),
        )

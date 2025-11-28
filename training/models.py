import torch.nn as nn
import numpy as np

class Generator(nn.Module):
    def __init__(self, img_size, latent_dim, feature_g=64, channels=3):
        super(Generator, self).__init__()

        self.init_size = 8
        self.num_upsamples = int( np.log2(img_size // self.init_size ))

        in_ch = feature_g * 2 ** self.num_upsamples

        layers = []

        def convolutional_block(in_sample, out_sample, stride, padding):
            return [
                nn.ConvTranspose2d(in_sample, out_sample, kernel_size=4, stride=stride, padding=padding),
                nn.InstanceNorm2d(out_sample),
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

        def discriminator_block(in_filters, out_filters):
            layers = [nn.Conv2d(in_filters, out_filters, 4, 2, 1, bias=False)]
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []

        in_ch = channels
        out_ch = feature_d

        num_upsamples = int(np.log2(img_size // 8)) + 1

        for _ in range(num_upsamples):
            layers += discriminator_block(in_ch, out_ch)
            in_ch = out_ch
            out_ch = min(out_ch * 2, feature_d * 16)

        layers.append(nn.Conv2d(in_ch, 1, 4, 1, 0))

        self.net = nn.Sequential(*layers)
        print(self.net)

    def forward(self, img):
        return self.net(img)

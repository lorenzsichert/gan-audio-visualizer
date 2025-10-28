import torch.nn as nn
import numpy as np

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

        print("Loaded Model:")
        print(self.net)

    def forward(self, z):
        img = self.net(z)
        return img

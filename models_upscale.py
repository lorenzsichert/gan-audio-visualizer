import torch.nn as nn
import torch
from torch.nn.functional import interpolate
from torch.nn.utils import spectral_norm



class InitLayer(nn.Module):
    def __init__(self, nz, channel):
        super().__init__()

        self.init = nn.Sequential(
                        spectral_norm(nn.ConvTranspose2d(nz, channel*2, 4, 1, 0, bias=False)),
                        nn.BatchNorm2d(channel*2), GLU() )

    def forward(self, noise):
        noise = noise.view(noise.shape[0], -1, 1, 1)
        return self.init(noise)

class NoiseInjection(nn.Module):
    def __init__(self, size):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.noise = None
        self.size = size

    def set_noise(self, noise):
        self.noise = noise

    def forward(self, feat):
        if self.noise is None:
            batch, _, height, width = feat.shape
            self.noise = torch.randn(batch, 1, height, width).to(feat.device)

        return feat + self.weight * self.noise 

class GLU(nn.Module):
    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])

def UpBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.utils.spectral_norm(nn.Conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False)),
        NoiseInjection(),
        nn.BatchNorm2d(out_planes*2), GLU(),
    )
    return block

def UpBlockComp(in_planes, out_planes, size):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        spectral_norm(nn.Conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False)),
        #convTranspose2d(in_planes, out_planes*2, 4, 2, 1, bias=False),
        NoiseInjection(size),
        nn.BatchNorm2d(out_planes*2), GLU(),
        spectral_norm(nn.Conv2d(out_planes, out_planes*2, 3, 1, 1, bias=False)),
        NoiseInjection(size),
        nn.BatchNorm2d(out_planes*2), GLU()
        )
    return block

class UGenerator(nn.Module):
    def __init__(self, nz=256, ngf=8, nc=3, img_size=256, layer=256):
        super().__init__()

        self.img_size = img_size
        self.layer = layer
        self.nc = nc

        nfc_multi = {4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        self.nfc = {}
        for k, v in nfc_multi.items():
            self.nfc[k] = int(v*ngf)

        self.init = InitLayer(nz, self.nfc[4])
        self.features = nn.ModuleList()
        for i in self.nfc:
            if i < layer:
                self.features.append(UpBlockComp(self.nfc[i], self.nfc[i*2],i*2))
            else:
                break

        self.to_big_low = nn.Sequential(
            spectral_norm(nn.Conv2d(self.nfc[layer // 2], self.nc, 3, 1, 1, bias=False))
        )
        self.to_big_high = nn.Sequential(
            spectral_norm(nn.Conv2d(self.nfc[layer], self.nc, 3, 1, 1, bias=False))
        )

        print(len(self.features))

    def add_layer(self):
        self.features.append(UpBlockComp(self.nfc[self.layer], self.nfc[self.layer*2], self.layer*2))
        self.to_big_low = self.to_big_high
        self.layer *= 2
        self.to_big_high = nn.Sequential(
            spectral_norm(nn.Conv2d(self.nfc[self.layer], self.nc, 3, 1, 1, bias=False))
        )



    def forward(self, input, alpha=1.0):
        feature = self.init(input)
        for i in range(len(self.features)-1):
            feature = self.features[i](feature)

        big_low = interpolate(self.to_big_low(feature), (self.layer,self.layer))
        big_high = self.to_big_high(self.features[len(self.features)-1](feature))

        return (1-alpha) * big_low + alpha * big_high






def downBlockHead(in_planes, out_planes):
    return nn.Sequential(
        spectral_norm(nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=False)),
        nn.LeakyReLU(0.2, inplace=True)
    )

def downBlock(in_planes, out_planes):
    return nn.Sequential(
        spectral_norm(nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=False)),
        nn.BatchNorm2d(out_planes), nn.LeakyReLU(0.2, inplace=True),
        spectral_norm(nn.Conv2d(out_planes, out_planes, 3, 1, 1, bias=False)),
        nn.BatchNorm2d(out_planes), nn.LeakyReLU(0.2, inplace=True)
    )

class DownBlockComp(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DownBlockComp, self).__init__()

        self.main = nn.Sequential(
            spectral_norm(nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(out_planes), nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(out_planes, out_planes, 3, 1, 1, bias=False)),
            nn.BatchNorm2d(out_planes), nn.LeakyReLU(0.2)
            )

        self.direct = nn.Sequential(
            nn.AvgPool2d(2, 2),
            spectral_norm(nn.Conv2d(in_planes, out_planes, 1, 1, 0, bias=False)),
            nn.BatchNorm2d(out_planes), nn.LeakyReLU(0.2))

    def forward(self, feat):
        return (self.main(feat) + self.direct(feat)) / 2


class Discriminator(nn.Module):
    def __init__(self, ndf, nc, img_size, layer):
        super().__init__()
        
        self.img_size = img_size
        self.layer = layer
        self.nc = nc

        nfc_multi = {4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        self.nfc = {}
        for k, v in nfc_multi.items():
            self.nfc[k] = int(v*ndf)

        self.rf = nn.Sequential(spectral_norm(nn.Conv2d(self.nfc[4], 1, 2, 1, 0)))
        self.features = nn.ModuleList()

        for i in self.nfc:
            if i < layer:
                self.features.append(downBlock(self.nfc[i*2], self.nfc[i]))
            else:
                break


        self.down_from_big_high = downBlockHead(3, self.nfc[self.layer])

        self.down_from_big_low = downBlockHead(3, self.nfc[self.layer // 2])


    def add_layer(self):
        self.features.append(downBlock(self.nfc[self.layer * 2], self.nfc[self.layer]))


        self.down_from_big_low = self.down_from_big_high
        self.down_from_big_high = downBlockHead(3, self.nfc[self.layer * 2])
        self.layer *= 2



    def forward(self, input, alpha):
        feature_high = self.down_from_big_high(input)
        feature_high_low = self.features[len(self.features)-1](feature_high)

        input_low = interpolate(input, (self.layer // 2, self.layer // 2))
        feature_low = self.down_from_big_low(input_low)


        feature = (1-alpha) * feature_low + alpha * feature_high_low

        for i in reversed(range(len(self.features)-1)):
            feature = self.features[i](feature)

        return self.rf(feature)


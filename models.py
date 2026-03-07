import torch.nn as nn
import torch
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

def UpBlock(in_planes, out_planes, size):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.utils.spectral_norm(nn.Conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False)),
        NoiseInjection(size),
        nn.BatchNorm2d(out_planes*2), GLU(),
    )
    return block

def UpBlockComp(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        spectral_norm(nn.Conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False)),
        #convTranspose2d(in_planes, out_planes*2, 4, 2, 1, bias=False),
        NoiseInjection(),
        nn.BatchNorm2d(out_planes*2), GLU(),
        spectral_norm(nn.Conv2d(out_planes, out_planes*2, 3, 1, 1, bias=False)),
        NoiseInjection(),
        nn.BatchNorm2d(out_planes*2), GLU()
        )
    return block

class SLE(nn.Module):
    """
    Skip-Layer Excitation:
    Uses low-resolution feature map to modulate high-resolution feature map.
    """
    def __init__(self, low_ch, high_ch):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(4)
        self.conv = nn.Sequential(
            nn.Conv2d(low_ch, high_ch, 4, 1, 0),
            nn.LeakyReLU(0.1),
            nn.Conv2d(high_ch, high_ch, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, low_feat, high_feat):
        excitation = self.avgpool(low_feat)
        excitation = self.conv(excitation)
        return high_feat * excitation

class Generator(nn.Module):
    def __init__(self, nz, ngf, nc, img_size, layer):
        super().__init__()

        self.img_size = img_size
        self.layer = layer
        self.nc = nc

        nfc_multi = {4:32, 8:16, 16:8, 32:4, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        self.nfc = {}
        for k, v in nfc_multi.items():
            self.nfc[k] = int(v*ngf)

        self.init = InitLayer(nz, self.nfc[4])
        self.features = nn.ModuleList()
        for i in self.nfc:
            if i < layer:
                self.features.append(UpBlock(self.nfc[i], self.nfc[i*2], i*2))

        self.to_big = nn.Sequential(
            spectral_norm(nn.Conv2d(self.nfc[layer], self.nc, 3, 1, 1, bias=False))
        )
        self.to_128 = nn.Sequential(
            spectral_norm(nn.Conv2d(self.nfc[128], self.nc, 3, 1, 1, bias=False))
        )

        self.sle = nn.ModuleList()


        for i in self.nfc:
            if i >= 64 and i <= layer:
                self.sle.append(SLE(self.nfc[i // 16], self.nfc[i]))

        print(self.features)
        print(self.sle)



    def forward(self, input):
        feature = self.init(input)
        features = [feature]
        f = 0
        for i in self.nfc:
            if i < self.layer // 2:
                feature = self.features[f](feature)
                if i >= 32:
                    feature = self.sle[f-3](features[f-3], feature)
                features.append(feature)
                f += 1


        feature = self.features[len(self.features)-1](feature)
        if self.layer == 128:
            feature = self.sle[1](features[1],feature)
        if self.layer == 256:
            feature = self.sle[2](features[2],feature)
        if self.layer == 512:
            feature = self.sle[3](features[3],feature)
        if self.layer == 1024:
            feature = self.sle[4](features[4],feature)

        big = self.to_big(feature)
        
        #big_128 = self.to_128(features[5])

        if self.training:
            return big
        else:
            return big






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

        nfc_multi = {4:32, 8:16, 16:8, 32:4, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        self.nfc = {}
        for k, v in nfc_multi.items():
            self.nfc[k] = int(v*ndf)

        self.features = nn.ModuleList()

        min_channel = 16

        for i in self.nfc:

            if i >= min_channel and i < layer:
                self.features.append(downBlock(self.nfc[i*2], self.nfc[i]))


        self.down_from_big = downBlockHead(3, self.nfc[self.layer])
        self.down_from_small = nn.Sequential(
            spectral_norm(nn.Conv2d(nc, self.nfc[256], 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2),
            downBlock(self.nfc[256], self.nfc[128]),
            downBlock(self.nfc[128], self.nfc[64]),
            downBlock(self.nfc[64], self.nfc[32]),
        )

        self.rf = nn.Sequential(
            spectral_norm(nn.Conv2d(self.nfc[16], self.nfc[8], 1, 1, 0, bias=False)),
            nn.BatchNorm2d(self.nfc[8]), nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(self.nfc[8], 1, 4, 1, 0, bias=False)),
        )

        self.rf_small = nn.Sequential(spectral_norm(nn.Conv2d(self.nfc[32], 1, 4, 1, 0, bias=False)))


        self.decoder_small = SimpleDecoder(self.nfc[32], nc)
        self.decoder_part = SimpleDecoder(self.nfc[32], nc)
        self.decoder_big = SimpleDecoder(self.nfc[16], nc)



    def forward(self, input, input_128, label="fake", part=(0,0)):
        # Main Big Image
        feature = self.down_from_big(input)
        features = [feature]

        for i in reversed(range(len(self.features))):
            feature = self.features[i](feature)
            features.append(feature)

        # 128 Small Image
        feature_small = self.down_from_small(input_128)
        rf = self.rf(feature).view(-1)
        rf_small = self.rf_small(feature_small).view(-1)

        if label == "real":
            print(features[len(features)-1].size())
            print(features[len(features)-2].size())
            print(features[len(features)-2][:,:,part[0]:(part[0]+8),part[1]:(part[1]+8)])
            rec_big = self.decoder_big(features[len(features)-1])
            rec_small = self.decoder_small(features[len(features)-2])
            rec_part = self.decoder_part(features[len(features)-2][:,:,part[0]:(part[0]+8),part[1]:(part[1]+8)])
            return torch.cat([rf, rf_small]), [rec_small, rec_big, rec_part]
             

        return torch.cat([rf, rf_small])

class SimpleDecoder(nn.Module):
    """docstring for CAN_SimpleDecoder"""
    def __init__(self, nfc_in=64, nc=3):
        super(SimpleDecoder, self).__init__()

        nfc_multi = {4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*32)

        def upBlock(in_planes, out_planes):
            block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                spectral_norm(nn.Conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False)),
                nn.BatchNorm2d(out_planes*2), GLU())
            return block

        self.main = nn.Sequential(  nn.AdaptiveAvgPool2d(8),
                                    upBlock(nfc_in, nfc[16]) ,
                                    upBlock(nfc[16], nfc[32]),
                                    upBlock(nfc[32], nfc[64]),
                                    upBlock(nfc[64], nfc[128]),
                                    spectral_norm(nn.Conv2d(nfc[128], nc, 3, 1, 1, bias=False)),
                                    nn.Tanh() )

    def forward(self, input):
        # input shape: c x 4 x 4
        return self.main(input)

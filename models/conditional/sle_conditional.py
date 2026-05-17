import math
import torch.nn as nn
import torch
from torch.nn.utils import spectral_norm

class ConditionalBatchNorm(nn.Module):
    def __init__(self, num_features, z_dim, class_emb_dim):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.linear = spectral_norm(
            nn.Linear(z_dim + class_emb_dim, num_features * 2)
        )

    def forward(self, x, z_chunk, class_emb):
        x = self.bn(x)

        cond = torch.cat([z_chunk, class_emb], dim=1)
        
        cond = cond.view(x.size()[0],-1)

        gamma_beta = self.linear(cond)         
        gamma, beta = gamma_beta.chunk(2, dim=1)

        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta  = beta.unsqueeze(-1).unsqueeze(-1)

        return (1 + gamma) * x + beta

class InitLayer(nn.Module):
    def __init__(self, nz, channel):
        super().__init__()

        self.init = nn.Sequential(
                        spectral_norm(nn.ConvTranspose2d(nz, channel*2, 4, 1, 0, bias=False)),
                        nn.BatchNorm2d(channel*2), GLU()
        )

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

class UpBlock(nn.Module):
    def __init__(self, in_planes, out_planes, size, z_dim, class_emb_dim, seperable):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        if seperable == True:
            self.conv = nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(in_planes, in_planes, 3, 1, 1, groups=in_planes,bias = False)),
                nn.utils.spectral_norm(nn.Conv2d(in_planes, out_planes*2, 1, 1, 0, bias=False)),
            )
        else:
            self.conv = nn.utils.spectral_norm(nn.Conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False))

        self.noise = NoiseInjection(size)
        self.cbn = ConditionalBatchNorm(out_planes*2, z_dim, class_emb_dim)
        self.glu = GLU()

    def forward(self, x, z_chunk, class_emb):
        x = self.up(x)
        x = self.conv(x)
        x = self.noise(x)
        x = self.cbn(x, z_chunk, class_emb)
        x = self.glu(x)
        return x


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
    def __init__(self, nz, ngf, nc, img_size, num_classes, skip_layer):
        super().__init__()

        self.img_size = img_size
        self.nc = nc
        self.num_classes = num_classes
        self.skip_layer = skip_layer
        self.class_emb_dim = 256
        self.splits = int(math.log(self.img_size, 2) - 1)
        self.chunk_size = nz // self.splits

        nfc_multi = {4:32, 8:16, 16:8, 32:4, 64:4, 128:2, 256:1, 512:0.5, 1024:0.25}
        self.nfc = {}
        for k, v in nfc_multi.items():
            self.nfc[k] = int(v*ngf)

        self.init = InitLayer(self.chunk_size, self.nfc[4])
        self.features = nn.ModuleList()
        for i in self.nfc:
            if i < self.img_size:
                self.features.append(UpBlock(self.nfc[i], self.nfc[i*2], i*2, self.chunk_size, self.class_emb_dim, seperable=False))


        self.to_big = nn.Sequential(
            spectral_norm(nn.Conv2d(self.nfc[self.img_size], self.nc, 3, 1, 1, bias=False))
        )
        self.to_128 = nn.Sequential(
            spectral_norm(nn.Conv2d(self.nfc[128], self.nc, 3, 1, 1, bias=False))
        )

        self.sle = nn.ModuleList()
        print(self.splits)
        print(len(self.features))


        for i in self.nfc:
            if i >= 4 * pow(2, self.skip_layer + 1) and i <= self.img_size:
                self.sle.append(SLE(self.nfc[i // pow(2, self.skip_layer + 1)], self.nfc[i]))

        self.embedding = nn.Linear(self.num_classes, self.class_emb_dim)



    def forward(self, input, y):
        #embedding = self.embedding(y).unsqueeze(-1).unsqueeze(-1)
        #y = y.view(input.size(0), self.num_classes, 1, 1)
        #input = torch.cat([input,y], dim=1)
        input = torch.split(input, self.chunk_size, dim=1)
        y = y.view(y.size()[0],-1)
        class_emb = self.embedding(y)
        class_emb = class_emb.view(class_emb.size()[0],class_emb.size()[1],1,1)
        feature = self.init(input[0])
        features = [feature]
        f = 0
        for i in self.nfc:
            if i < self.img_size:
                feature = self.features[f](feature, input[f+1], class_emb)
                if i >= 2 * pow(2, self.skip_layer + 1):
                    feature = self.sle[f-self.skip_layer](features[f - self.skip_layer], feature)
                features.append(feature)
                f += 1

        big = self.to_big(feature)

        if self.training:
            big_128 = self.to_128(features[5])
            return big, big_128
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
    def __init__(self, ndf, nc, img_size, num_classes, skip_layer):
        super().__init__()
        
        self.img_size = img_size
        self.num_classes = num_classes
        self.nc = nc
        self.skip_layer = skip_layer

        nfc_multi = {4:16, 8:16, 16:8, 32:4, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        self.nfc = {}
        for k, v in nfc_multi.items():
            self.nfc[k] = int(v*ndf)

        self.features = nn.ModuleList()

        min_channel = 16

        for i in self.nfc:

            if i >= min_channel and i < self.img_size:
                self.features.append(DownBlockComp(self.nfc[i*2], self.nfc[i]))


        self.down_from_big = downBlockHead(nc, self.nfc[self.img_size])
        self.down_from_small = nn.Sequential(
            spectral_norm(nn.Conv2d(nc, self.nfc[256], 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2),
            DownBlockComp(self.nfc[256], self.nfc[128]),
            DownBlockComp(self.nfc[128], self.nfc[64]),
            DownBlockComp(self.nfc[64], self.nfc[32]),
        )

        self.rf = nn.Sequential(
            spectral_norm(nn.Conv2d(self.nfc[16], self.nfc[8], 1, 1, 0, bias=False)),
            nn.BatchNorm2d(self.nfc[8]), nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(self.nfc[8], 1, 4, 1, 0, bias=False)),
        )

        self.rf_small = nn.Sequential(spectral_norm(nn.Conv2d(self.nfc[32], 1, 4, 1, 0, bias=False)))


        self.decoder_small = SimpleDecoder(self.nfc[32], nc, ndf)
        self.decoder_part = SimpleDecoder(self.nfc[32], nc, ndf)
        self.decoder_big = SimpleDecoder(self.nfc[16], nc, ndf)

        self.embedding = nn.Linear(num_classes, self.nfc[16])
        self.embedding_small = nn.Linear(num_classes, self.nfc[32])
        nn.init.orthogonal_(self.embedding.weight)

        self.sle = nn.ModuleList()

        for i in self.nfc:
            if i >= 16 and i <= self.img_size // pow(2, self.skip_layer + 1):
                self.sle.append(SLE(self.nfc[i* pow(2, self.skip_layer + 1)], self.nfc[i]))



    def forward(self, input, input_128, class_label, label="fake", part=(0,0)):
        # Main Big Image
        feature = self.down_from_big(input)
        features = [feature]

        f = 0
        for i in reversed(range(len(self.features))):
            feature = self.features[i](feature)
            if i < len(self.features)-self.skip_layer:
                feature = self.sle[i](features[f- self.skip_layer],feature)
            features.append(feature)
            f += 1

        # 128 Small Image
        feature_small = self.down_from_small(input_128)
        rf = self.rf(feature).view(feature.size()[0],-1)
        rf_small = self.rf_small(feature_small).view(feature.size()[0],-1)
        # rf and rf_small are 5x5 logits

        class_label = class_label.view(class_label.size()[0], -1)
        v_y = self.embedding(class_label)
        v_y = v_y.view(v_y.size(0), -1)
        phi = torch.sum(feature, dim=[2,3])
        proj = (phi * v_y).sum(dim=1).view(phi.size(0), -1)
        rf = rf + proj

        v_y_small = self.embedding_small(class_label)
        v_y_small = v_y_small.view(v_y_small.size(0), -1)
        phi_small = torch.sum(feature_small, dim=[2,3])
        proj_small = (phi_small * v_y_small).sum(dim=1).view(phi_small.size(0), 1)
        rf_small = rf_small + proj_small



        if label == "real": 
            rec_small = self.decoder_small(feature_small)
            rec_big = self.decoder_big(features[len(features)-1])
            rec_part = self.decoder_part(features[len(features)-2][:,:,part[0]:(part[0]+8),part[1]:(part[1]+8)])
            return torch.cat([rf, rf_small]), [rec_small, rec_big, rec_part]
             

        return torch.cat([rf, rf_small])

class SimpleDecoder(nn.Module):
    """docstring for CAN_SimpleDecoder"""
    def __init__(self, nfc_in=16, nc=3, features=16):
        super(SimpleDecoder, self).__init__()

        nfc_multi = {4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*features)

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

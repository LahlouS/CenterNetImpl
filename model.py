import numpy as np
import torch
from torch import nn
import torchvision
from torchvision.models import ResNet50_Weights

###################################### BACKBONE == ResNet only for the moment ###############################################

class BackBone(nn.Module):
    def __init__(self): # TODO later add config parameter to choose differente architectures
        super().__init__()
        self.model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.out_channels = 2048 # only if resnet50
    
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        # we keep intermediates outputs for the FPN to do his job
        x1 = self.model.layer1(x)
        x2 = self.model.layer2(x1)
        x3 = self.model.layer3(x2)
        x4 = self.model.layer4(x3)
        
        # x4_bis = x4

        # print('\n\ntest BackBone DEBUG ----> x4.min:', round(x4_bis[0].min().item(), 4), 'x4.max:', round(x4_bis[0].max().item(), 4), end='\n\n')
        return x1, x2, x3, x4
    
    def _freezer(self):
        print('LOG: freezing had been called')
        self.model.conv1.eval()
        self.model.bn1.eval()
        self.model.relu.eval()
        self.model.maxpool.eval()

        self.model.layer1.eval()
        self.model.layer2.eval()
        self.model.layer3.eval()

###############################################################################################################################


################################################ Feature Pyramid Network  #####################################################

# LATERAL CONNECTIONS FOR THE FPN
class Conv1x1(nn.Module):
    def __init__(self, num_in, num_out):
        super().__init__()
        self.conv = nn.Conv2d(num_in, num_out, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(num_out)
        self.active = nn.ReLU(True)
        self.block = nn.Sequential(self.conv, self.norm, self.active)

    def forward(self, x):
        return self.block(x)


# ANTI ALIASING CONVOLUTIONS FOR THE FPN 
class Conv3x3(nn.Module):
    def __init__(self, num_in, num_out):
        super().__init__()
        self.conv = nn.Conv2d(num_in, num_out, kernel_size=3, padding=1,
                              bias=False)
        self.norm = nn.BatchNorm2d(num_out)
        self.active = nn.ReLU(True)
        self.block = nn.Sequential(self.conv, self.norm, self.active)

    def forward(self, x):
        return self.block(x)


class FPN(nn.Module):
    def __init__(self, in_channels = 2048, outplanes = 256):
        super(FPN, self).__init__()

        self.laterals = nn.Sequential(*[Conv1x1(in_channels // (2 ** c), outplanes) for c in range(4)])
        self.smooths = nn.Sequential(*[Conv3x3(outplanes * c, outplanes * c) for c in range(1, 5)])
        self.pooling = nn.MaxPool2d(2)

        self.out_channels = outplanes * 4 # because our top-down pathway is composed of 4 layers


    def forward(self, features):
        laterals = [lateral(features[f]) for f, lateral in enumerate(self.laterals)]

        map4 = laterals[0]

        map3 = laterals[1] + nn.functional.interpolate(map4, scale_factor=2, mode="nearest")
        map2 = laterals[2] + nn.functional.interpolate(map3, scale_factor=2, mode="nearest")
        map1 = laterals[3] + nn.functional.interpolate(map2, scale_factor=2, mode="nearest")

        map1 = self.smooths[0](map1)
        map2 = self.smooths[1](torch.cat([map2, self.pooling(map1)], dim=1))
        map3 = self.smooths[2](torch.cat([map3, self.pooling(map2)], dim=1))
        map4 = self.smooths[3](torch.cat([map4, self.pooling(map3)], dim=1))

        return map4
    
###############################################################################################################################

##################################################### UP SAMPLER ##############################################################

class Decoder(nn.Module):
    def __init__(self, inplanes, bn_momentum=0.1):
        super(Decoder, self).__init__()
        self.bn_momentum = bn_momentum
        # backbone output: [b, 2048, _h, _w]
        self.inplanes = inplanes
        self.deconv_with_bias = False
        self.deconv_layers = self._make_deconv_layer(
            num_layers=5,
            num_filters=[256, 256, 256, 256, 256],
            num_kernels=[4, 4, 4, 4, 4],
        )

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        layers = []
        for i in range(num_layers):
            kernel = num_kernels[i]
            padding = 0 if kernel == 2 else 1
            output_padding = 1 if kernel == 3 else 0
            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=self.bn_momentum))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.deconv_layers(x)
        return x

###############################################################################################################################

##################################################### HEADS CONSTRUCTION ######################################################

class Heads(nn.Module):
    def __init__(self, nclasses=53, in_channels = 256):
        super(Heads, self).__init__()

        self.nclasses = nclasses

        self.heat_maps = nn.Sequential(
                                            nn.Conv2d(in_channels, out_channels = 64, kernel_size = 3, stride=2, padding=1, bias=True),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(64, self.nclasses, kernel_size = 1, stride=2, padding=0, bias=True),
                                            nn.Softmax()
                                        )
        
        self.offset_maps = nn.Sequential(
                                            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(in_channels, 2, kernel_size=1, stride=2, padding=0),
                                        )
        
        self.size_maps = nn.Sequential(
                                            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(in_channels, 2, kernel_size=1, stride=2, padding=0),
                                        )

    def forward(self, x):
        heat = self.heat_maps(x)
        offset = self.offset_maps(x)
        size = self.size_maps(x)

        return heat, offset, size
    
###############################################################################################################################


######################################### GATHERING EVERYTHING IN CENTERNET ###################################################

class CenterNet(nn.Module):

    def __init__(self, input_shape = (3, 256, 256), nclasses = 53, need_fpn=True):
        super(CenterNet, self).__init__()

        self.input_shape = input_shape
        self.nclasses = nclasses

        self.backbone = BackBone()
        # self.backbone._freezer()
        self.fpn = FPN(in_channels = self.backbone.out_channels, outplanes = 256) # maybe try with 512
        self.up_sampler = Decoder(self.fpn.out_channels if need_fpn else self.backbone.out_channels)
        self.heads = Heads(self.nclasses, in_channels = 256)

        self.need_fpn = need_fpn

        if not self.need_fpn:
            print('LOG --> FPN is desactivated')
        else:
            print('LOG --> FPN is activated')

    def forward(self, x):

        features = self.backbone(x)

        if self.need_fpn:
            rescaled_features = self.fpn(features[::-1])
            upsampled_features = self.up_sampler(rescaled_features)
        else:
            upsampled_features = self.up_sampler(features[-1])

        heads = self.heads(upsampled_features)

        return heads
    

if __name__ == '__main__':
    random_input = torch.rand(32, 3, 256, 256)
    print(f'TEST centernet models pipeline:\n->creating random input of size {tuple(random_input.shape)} (sample shape from Dataset)\n')

    cn = CenterNet(random_input.shape, nclasses=53)
    print('**Centernet successfully initialized**\n')
    hmaps, offset, size = cn(random_input)

    print('Predictions shape:\n', end='')
    print('hmaps:', hmaps.shape)
    print('offset:', offset.shape)
    print('size:', size.shape)




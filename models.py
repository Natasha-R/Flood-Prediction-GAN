import torch
from torch import nn

## Pix2Pix ##########################################################################################

class Pix2PixGenerator(nn.Module):
    def __init__(self):
        super(Pix2PixGenerator, self).__init__()

        unet_block = SkipConnectionBlock(outer_nc=512, inner_nc=512, input_nc=None, submodule=None, outermost=False, innermost=True, use_dropout=False)
        for i in range(3): 
            unet_block = SkipConnectionBlock(outer_nc=512, inner_nc=512, input_nc=None, submodule=unet_block, outermost=False, innermost=False, use_dropout=True)
        unet_block = SkipConnectionBlock(outer_nc=256, inner_nc=512, input_nc=None, submodule=unet_block, outermost=False, innermost=False, use_dropout=False)
        unet_block = SkipConnectionBlock(outer_nc=128, inner_nc=256, input_nc=None, submodule=unet_block, outermost=False, innermost=False, use_dropout=False)
        unet_block = SkipConnectionBlock(outer_nc=64, inner_nc=128, input_nc=None, submodule=unet_block, outermost=False, innermost=False, use_dropout=False)
        self.model = SkipConnectionBlock(outer_nc=3, inner_nc=64, input_nc=9, submodule=unet_block, outermost=True, innermost=False, use_dropout=False)

    def forward(self, input):
        return self.model(input)

class SkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc, submodule, outermost, innermost, use_dropout):
        super(SkipConnectionBlock, self).__init__()
        self.outermost = outermost
        
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        uprelu = nn.ReLU(True)
        downnorm = nn.BatchNorm2d(inner_nc)
        upnorm = nn.BatchNorm2d(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)
        
class Pix2PixDiscriminator(nn.Module):
    def __init__(self):
        super(Pix2PixDiscriminator, self).__init__()
        
        sequence = [nn.Conv2d(12, 64, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, 3):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [nn.Conv2d(64 * nf_mult_prev, 64 * nf_mult, kernel_size=4, stride=2, padding=1, bias=False),
                         nn.BatchNorm2d(64 * nf_mult),
                         nn.LeakyReLU(0.2, True)]
        nf_mult_prev = nf_mult
        sequence += [nn.Conv2d(64 * nf_mult_prev, 512, kernel_size=4, stride=1, padding=1, bias=False),
                     nn.BatchNorm2d(512),
                     nn.LeakyReLU(0.2, True)]
        sequence += [nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)
    
## CycleGAN ##########################################################################################

class CycleGANGenerator(nn.Module):
    def __init__(self, input_channels):
        super(CycleGANGenerator, self).__init__()

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_channels, 64, kernel_size=7, padding=0, bias=True),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(True)]
        for i in range(2):
            mult = 2 ** i
            model += [nn.Conv2d(64 * mult, 64 * mult * 2, kernel_size=3, stride=2, padding=1, bias=True),
                      nn.InstanceNorm2d(64 * mult * 2),
                      nn.ReLU(True)]
        for i in range(9):
            model += [ResnetBlock(dim=256)]
        for i in range(2):
            mult = 2 ** (2 - i)
            model += [nn.ConvTranspose2d(64 * mult, int(64 * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=True),
                      nn.InstanceNorm2d(int(64 * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(64, 3, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(*[nn.ReflectionPad2d(1),
                                          nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True), 
                                          nn.InstanceNorm2d(dim), 
                                          nn.ReLU(True),
                                          nn.ReflectionPad2d(1),
                                          nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True), 
                                          nn.InstanceNorm2d(dim)])
    def forward(self, x):
        out = x + self.conv_block(x)
        return out
    
class CycleGANDiscriminator(nn.Module):
    def __init__(self):
        super(CycleGANDiscriminator, self).__init__()
        
        sequence = [nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, 3):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [nn.Conv2d(64 * nf_mult_prev, 64 * nf_mult, kernel_size=4, stride=2, padding=1, bias=True),
                         nn.InstanceNorm2d(64 * nf_mult),
                         nn.LeakyReLU(0.2, True)]
        nf_mult_prev = nf_mult
        sequence += [nn.Conv2d(64 * nf_mult_prev, 512, kernel_size=4, stride=1, padding=1, bias=True),
                     nn.InstanceNorm2d(512),
                     nn.LeakyReLU(0.2, True)]
        sequence += [nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)] 
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)
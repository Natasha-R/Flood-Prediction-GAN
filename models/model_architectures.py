import torch
from torch import nn
import torch.nn.functional as F

## Pix2Pix 
## Architecture adapted from: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/
##########################################################################################

class Pix2PixGenerator(nn.Module):
    def __init__(self, input_channels):
        super(Pix2PixGenerator, self).__init__()

        unet_block = Pix2PixBlock(outer_nc=512, inner_nc=512, input_nc=None, submodule=None, outermost=False, innermost=True, use_dropout=False)
        for i in range(3): 
            unet_block = Pix2PixBlock(outer_nc=512, inner_nc=512, input_nc=None, submodule=unet_block, outermost=False, innermost=False, use_dropout=True)
        unet_block = Pix2PixBlock(outer_nc=256, inner_nc=512, input_nc=None, submodule=unet_block, outermost=False, innermost=False, use_dropout=False)
        unet_block = Pix2PixBlock(outer_nc=128, inner_nc=256, input_nc=None, submodule=unet_block, outermost=False, innermost=False, use_dropout=False)
        unet_block = Pix2PixBlock(outer_nc=64, inner_nc=128, input_nc=None, submodule=unet_block, outermost=False, innermost=False, use_dropout=False)
        self.model = Pix2PixBlock(outer_nc=3, inner_nc=64, input_nc=input_channels, submodule=unet_block, outermost=True, innermost=False, use_dropout=False)

    def forward(self, input):
        return self.model(input)

class Pix2PixBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc, submodule, outermost, innermost, use_dropout):
        super(Pix2PixBlock, self).__init__()
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
    def __init__(self, input_channels):
        super(Pix2PixDiscriminator, self).__init__()
        
        sequence = [nn.Conv2d(input_channels + 3, 64, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
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
    
## CycleGAN 
## Architecture adapted from: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/
##########################################################################################

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
            model += [CycleGANBlock(dim=256)]
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

class CycleGANBlock(nn.Module):
    def __init__(self, dim):
        super(CycleGANBlock, self).__init__()
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
    def __init__(self, input_channels):
        super(CycleGANDiscriminator, self).__init__()
        
        sequence = [nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
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
    
## AttentionGAN 
## Architecture adapted from: https://github.com/Ha0Tang/AttentionGAN
##########################################################################################

class AttentionGANGenerator(nn.Module):
    def __init__(self, input_channels):
        super(AttentionGANGenerator, self).__init__()
        
        self.input_channels = input_channels
        self.last_attention_mask = None
        
        self.conv1 = nn.Conv2d(input_channels, 64, 7, 1, 0)
        self.conv1_norm = nn.InstanceNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv2_norm = nn.InstanceNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, 2, 1)
        self.conv3_norm = nn.InstanceNorm2d(256)

        self.resnet_blocks = []
        for i in range(9):
            self.resnet_blocks.append(AttentionGANBlock(256, 3, 1, 1))
        self.resnet_blocks = nn.Sequential(*self.resnet_blocks)

        self.deconv1_content = nn.ConvTranspose2d(256, 128, 3, 2, 1, 1)
        self.deconv1_norm_content = nn.InstanceNorm2d(128)
        self.deconv2_content = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1)
        self.deconv2_norm_content = nn.InstanceNorm2d(64)
        self.deconv3_content = nn.Conv2d(64, 27, 7, 1, 0)

        self.deconv1_attention = nn.ConvTranspose2d(256, 128, 3, 2, 1, 1)
        self.deconv1_norm_attention = nn.InstanceNorm2d(128)
        self.deconv2_attention = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1)
        self.deconv2_norm_attention = nn.InstanceNorm2d(64)
        self.deconv3_attention = nn.Conv2d(64, 10, 1, 1, 0)
        
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, input):
        
        # process the input image
        x = F.pad(input, (3, 3, 3, 3), "reflect")
        x = F.relu(self.conv1_norm(self.conv1(x)))
        x = F.relu(self.conv2_norm(self.conv2(x)))
        x = F.relu(self.conv3_norm(self.conv3(x)))
        x = self.resnet_blocks(x)
        
        # create 9 content masks
        # each content mask has 3 channels
        x_content = F.relu(self.deconv1_norm_content(self.deconv1_content(x)))
        x_content = F.relu(self.deconv2_norm_content(self.deconv2_content(x_content)))
        x_content = F.pad(x_content, (3, 3, 3, 3), "reflect")
        content = self.deconv3_content(x_content)
        content = self.tanh(content)
        content1 = content[:, 0:3, :, :]
        content2 = content[:, 3:6, :, :]
        content3 = content[:, 6:9, :, :]
        content4 = content[:, 9:12, :, :]
        content5 = content[:, 12:15, :, :]
        content6 = content[:, 15:18, :, :]
        content7 = content[:, 18:21, :, :]
        content8 = content[:, 21:24, :, :]
        content9 = content[:, 24:27, :, :]

        # create 10 attention masks (9 foreground and 1 background)
        # each attention mask has one channel, which is then duplicated 3 times
        x_attention = F.relu(self.deconv1_norm_attention(self.deconv1_attention(x)))
        x_attention = F.relu(self.deconv2_norm_attention(self.deconv2_attention(x_attention)))
        attention = self.deconv3_attention(x_attention)
        attention = self.softmax(attention)
        attention1 = attention[:, 0:1, :, :].repeat(1, 3, 1, 1)
        attention2 = attention[:, 1:2, :, :].repeat(1, 3, 1, 1)
        attention3 = attention[:, 2:3, :, :].repeat(1, 3, 1, 1)
        attention4 = attention[:, 3:4, :, :].repeat(1, 3, 1, 1)
        attention5 = attention[:, 4:5, :, :].repeat(1, 3, 1, 1)
        attention6 = attention[:, 5:6, :, :].repeat(1, 3, 1, 1)
        attention7 = attention[:, 6:7, :, :].repeat(1, 3, 1, 1)
        attention8 = attention[:, 7:8, :, :].repeat(1, 3, 1, 1)
        attention9 = attention[:, 8:9, :, :].repeat(1, 3, 1, 1)
        attention10 = attention[:, 9:10, :, :].repeat(1, 3, 1, 1) # background mask
        
        # multiply each content mask to each foreground attention mask
        output1 = content1 * attention1
        output2 = content2 * attention2
        output3 = content3 * attention3
        output4 = content4 * attention4
        output5 = content5 * attention5
        output6 = content6 * attention6
        output7 = content7 * attention7
        output8 = content8 * attention8
        output9 = content9 * attention9
        # multiply the original input image to the background mask
        output10 = input[:, :3, :, :] * attention10
        
        # save the attention mask
        self.last_attention_mask = attention10[:, 0, :, :]
        
        # sum the results of all of the content masks multiplied to all of the attention masks
        overall_output = output1 + output2 + output3 + output4 + output5 + output6 + output7 + output8 + output9 + output10
        return overall_output
    
class AttentionGANBlock(nn.Module):
    def __init__(self, channel, kernel, stride, padding):
        super(AttentionGANBlock, self).__init__()
        
        self.padding = padding
        self.conv1 = nn.Conv2d(channel, channel, kernel, stride, 0)
        self.conv1_norm = nn.InstanceNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, kernel, stride, 0)
        self.conv2_norm = nn.InstanceNorm2d(channel)

    def forward(self, input):
        x = F.pad(input, (self.padding, self.padding, self.padding, self.padding), "reflect")
        x = F.relu(self.conv1_norm(self.conv1(x)))
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), "reflect")
        x = self.conv2_norm(self.conv2(x))

        return input + x
    
class AttentionGANDiscriminator(nn.Module):
    def __init__(self, input_channels):
        super(AttentionGANDiscriminator, self).__init__()
        
        sequence = [nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
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
    
## PairedAttention
## Architecture adapted from: https://github.com/Ha0Tang/AttentionGAN
##########################################################################################

class PairedAttentionGenerator(nn.Module):
    def __init__(self, input_channels):
        super(PairedAttentionGenerator, self).__init__()
        
        self.input_channels = input_channels
        self.last_attention_mask = None
        
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=1, padding=0)
        self.conv1_norm = nn.InstanceNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv2_norm = nn.InstanceNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv3_norm = nn.InstanceNorm2d(256)

        self.resnet_blocks = []
        for i in range(9):
            self.resnet_blocks.append(PairedAttentionBlock(channel=256, kernel=3, stride=1, padding=1))
        self.resnet_blocks = nn.Sequential(*self.resnet_blocks)

        self.deconv1_content = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv1_norm_content = nn.InstanceNorm2d(128)
        self.deconv2_content = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2_norm_content = nn.InstanceNorm2d(64)
        self.deconv3_content = nn.Conv2d(64, 27, kernel_size=7, stride=1, padding=0)

        self.deconv1_attention = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv1_norm_attention = nn.InstanceNorm2d(128)
        self.deconv2_attention = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2_norm_attention = nn.InstanceNorm2d(64)
        self.deconv3_attention = nn.Conv2d(64, 10, kernel_size=1, stride=1, padding=0)
        
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, input):
        
        # process the input image
        x = F.pad(input, (3, 3, 3, 3), "reflect")
        x = F.relu(self.conv1_norm(self.conv1(x)))
        x = F.relu(self.conv2_norm(self.conv2(x)))
        x = F.relu(self.conv3_norm(self.conv3(x)))
        x = self.resnet_blocks(x)
        
        # create 9 content masks
        # each content mask has 3 channels
        x_content = F.relu(self.deconv1_norm_content(self.deconv1_content(x)))
        x_content = F.relu(self.deconv2_norm_content(self.deconv2_content(x_content)))
        x_content = F.pad(x_content, (3, 3, 3, 3), "reflect")
        content = self.deconv3_content(x_content)
        content = self.tanh(content)
        content1 = content[:, 0:3, :, :]
        content2 = content[:, 3:6, :, :]
        content3 = content[:, 6:9, :, :]
        content4 = content[:, 9:12, :, :]
        content5 = content[:, 12:15, :, :]
        content6 = content[:, 15:18, :, :]
        content7 = content[:, 18:21, :, :]
        content8 = content[:, 21:24, :, :]
        content9 = content[:, 24:27, :, :]

        # create 10 attention masks (9 foreground and 1 background)
        # each attention mask has one channel, which is then duplicated 3 times
        x_attention = F.relu(self.deconv1_norm_attention(self.deconv1_attention(x)))
        x_attention = F.relu(self.deconv2_norm_attention(self.deconv2_attention(x_attention)))
        attention = self.deconv3_attention(x_attention)
        attention = self.softmax(attention)
        attention1 = attention[:, 0:1, :, :].repeat(1, 3, 1, 1)
        attention2 = attention[:, 1:2, :, :].repeat(1, 3, 1, 1)
        attention3 = attention[:, 2:3, :, :].repeat(1, 3, 1, 1)
        attention4 = attention[:, 3:4, :, :].repeat(1, 3, 1, 1)
        attention5 = attention[:, 4:5, :, :].repeat(1, 3, 1, 1)
        attention6 = attention[:, 5:6, :, :].repeat(1, 3, 1, 1)
        attention7 = attention[:, 6:7, :, :].repeat(1, 3, 1, 1)
        attention8 = attention[:, 7:8, :, :].repeat(1, 3, 1, 1)
        attention9 = attention[:, 8:9, :, :].repeat(1, 3, 1, 1)
        attention10 = attention[:, 9:10, :, :].repeat(1, 3, 1, 1) # background mask
        
        # multiply each content mask to each foreground attention mask
        output1 = content1 * attention1
        output2 = content2 * attention2
        output3 = content3 * attention3
        output4 = content4 * attention4
        output5 = content5 * attention5
        output6 = content6 * attention6
        output7 = content7 * attention7
        output8 = content8 * attention8
        output9 = content9 * attention9
        # multiply the original input image to the background mask
        output10 = input[:, :3, :, :] * attention10
        
        # save the attention mask
        self.last_attention_mask = attention10[:, 0, :, :]
        
        # sum the results of all of the content masks multiplied to all of the attention masks
        overall_output = output1 + output2 + output3 + output4 + output5 + output6 + output7 + output8 + output9 + output10
        return overall_output
    
class PairedAttentionBlock(nn.Module):
    def __init__(self, channel, kernel, stride, padding):
        super(PairedAttentionBlock, self).__init__()
        
        self.padding = padding
        self.conv1 = nn.Conv2d(channel, channel, kernel, stride, 0)
        self.conv1_norm = nn.InstanceNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, kernel, stride, 0)
        self.conv2_norm = nn.InstanceNorm2d(channel)

    def forward(self, input):
        x = F.pad(input, (self.padding, self.padding, self.padding, self.padding), "reflect")
        x = F.relu(self.conv1_norm(self.conv1(x)))
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), "reflect")
        x = self.conv2_norm(self.conv2(x))

        return input + x
    
class PairedAttentionDiscriminator(nn.Module):
    def __init__(self, input_channels):
        super(PairedAttentionDiscriminator, self).__init__()
        
        sequence = [nn.Conv2d(input_channels + 3, 64, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
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
    
## Pix2PixUNet
## Architecture adapted from: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/
##########################################################################################

class Pix2PixUNet(nn.Module):
    def __init__(self):
        super(Pix2PixUNet, self).__init__()

        unet_block = Pix2PixUNetBlock(outer_nc=512, inner_nc=512, input_nc=None, submodule=None, outermost=False, innermost=True, use_dropout=False)
        for i in range(3): 
            unet_block = Pix2PixUNetBlock(outer_nc=512, inner_nc=512, input_nc=None, submodule=unet_block, outermost=False, innermost=False, use_dropout=True)
        unet_block = Pix2PixUNetBlock(outer_nc=256, inner_nc=512, input_nc=None, submodule=unet_block, outermost=False, innermost=False, use_dropout=False)
        unet_block = Pix2PixUNetBlock(outer_nc=128, inner_nc=256, input_nc=None, submodule=unet_block, outermost=False, innermost=False, use_dropout=False)
        unet_block = Pix2PixUNetBlock(outer_nc=64, inner_nc=128, input_nc=None, submodule=unet_block, outermost=False, innermost=False, use_dropout=False)
        self.model = Pix2PixUNetBlock(outer_nc=1, inner_nc=64, input_nc=3, submodule=unet_block, outermost=True, innermost=False, use_dropout=False)

    def forward(self, input):
        return self.model(input)

class Pix2PixUNetBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc, submodule, outermost, innermost, use_dropout):
        super(Pix2PixUNetBlock, self).__init__()
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
            up = [uprelu, upconv, nn.Sigmoid()]
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
        
import torch.nn.functional as F

### Original U-Net
## Architecture adapted from: https://github.com/milesial/Pytorch-UNet
##########################################################################################

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels))
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)
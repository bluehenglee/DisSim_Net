import torch
import os
from abc import ABC
import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision import models
import torch.nn.functional as F
from utils.practical_function import *
from functools import partial
f = torch.cuda.is_available()
from einops import rearrange, repeat, reduce
import numpy as np
from collections import OrderedDict

device = torch.device("cuda" if f else "cpu")

# import Constants
nonlinearity = partial(F.relu, inplace=True)


class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.max_pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.max_pool_conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2), diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out

class SPPblock(nn.Module):
    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.upsample(self.conv(self.pool1(x)), size=(h, w), mode='bilinear')
        self.layer2 = F.upsample(self.conv(self.pool2(x)), size=(h, w), mode='bilinear')
        self.layer3 = F.upsample(self.conv(self.pool3(x)), size=(h, w), mode='bilinear')
        self.layer4 = F.upsample(self.conv(self.pool4(x)), size=(h, w), mode='bilinear')

        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)

        return out

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity
        #棋盘格效应通常是由于转置卷积层（nn.ConvTranspose2d）的特定配置引起的，尤其是当使用奇数大小的卷积核和较大的步长时。
        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        # self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 4, stride=2, padding=1)
        # self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 2, stride=2)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class MLP(nn.Module):
    def __init__(self, dim, projection_size=128, hidden_size=4096):
        super(MLP, self).__init__()
        
        # 定义网络结构
        self.network = nn.Sequential(
            nn.Linear(dim, hidden_size),       # 输入层到隐藏层
            nn.BatchNorm1d(hidden_size),       # 批量归一化
            nn.ReLU(inplace=True),             # 激活函数
            nn.Linear(hidden_size, projection_size)  # 隐藏层到输出层
        )

    def forward(self, x):
        return self.network(x)

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, drop_rate):
        super(DenseLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        self.drop_rate = drop_rate
    
    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        if self.drop_rate > 0:
            out = nn.Dropout2d(self.drop_rate)(out)
        return torch.cat([x, out], 1)
    
class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, drop_rate):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate, drop_rate))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        return self.pool(self.conv(self.relu(self.bn(x))))
    
class PAplusNet_unet(nn.Module):
    def __init__(self, num_channels=4, num_classes=1, patch=3):
        super(PAplusNet_unet, self).__init__()
        self.inc = inconv(num_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, num_classes)
        self.relu = nn.ReLU()
        self.get_projectors = nn.ModuleDict({
            '1': self.create_projector(64),  # patch_size=1*1*64
            '3': self.create_projector(576),  # patch_size=3
            '5': self.create_projector(1600),  # patch_size=5
            '7': self.create_projector(3136),  # patch_size=7
            '9': self.create_projector(5184)   # patch_size=9
        })
        patch_size = str(patch)
        if patch_size in self.get_projectors:
            self.projector = self.get_projectors[patch_size]
        else:
            raise ValueError(f"Invalid patch size: {patch_size}. Must be one of {list(self.get_projectors.keys())}")
        
    def find_nonzero_regions(self, image, m):
        binary_image = image > 0
        kernel = torch.ones((1, 1, m, m), device=image.device)
        conv_result = F.conv2d(binary_image.float(), kernel, stride=1, padding=0)
        nonzero_coords = torch.nonzero(conv_result[:] == m * m, as_tuple=False)

        return nonzero_coords

    def generate_samples(self, feature, mask, threshold=0.5, num_samples_per_batch=10, m=1):
        mask = (mask > threshold)
        b, c, h, w = feature.shape
        if mask.sum() == 0:
            return torch.tensor([]).to(feature.device)
        all_indices = self.find_nonzero_regions(mask, m)
        sample_indices = torch.randperm(all_indices.size(0))[:num_samples_per_batch * b]
        selected_indices = all_indices[sample_indices]
        batch_indices, y_indices, x_indices = selected_indices[:, 0], selected_indices[:, 2], selected_indices[:, 3]
        # [b, c, h, w] -> [b, c, num_patches_h, num_patches_w, m, m]
        patches_unfolded = feature.unfold(2, m, 1).unfold(3, m, 1)
        patches = patches_unfolded[batch_indices, :, y_indices, x_indices]
        flattened_representation = rearrange(patches, 'n ... -> n (...)')
        projection = self.projector(flattened_representation)

        return projection

    def contrast_nce_fast(self, anchor, positive, negative, temperature=0.3):
        anchor = F.normalize(anchor, dim=1)  # [b*n, 64]
        positive = F.normalize(positive, dim=1)  # [b*n, 64]
        negative = F.normalize(negative, dim=1)  # [b*n, 64]
        anchor_positive_dot = (anchor * positive).sum(dim=1, keepdim=True) / temperature  # [b*n, 1]
        anchor_negative_dot = torch.matmul(anchor, negative.t()) / temperature  # [b*n, b*n]
        exp_pos = torch.exp(anchor_positive_dot)  # [b*n, 1]
        exp_neg = torch.exp(anchor_negative_dot)  # [b*n, b*n]
        neg_exp_sum = exp_neg.sum(dim=1, keepdim=True)  # [b*n, 1]
        loss = -torch.log(exp_pos / (exp_pos + neg_exp_sum + 1e-9))  # [b*n, 1]
        total_loss = loss.mean() 

        return total_loss

    def compute_loss(self, feat, fore, back, num, patchsize, temperature=0.07):
        anchor_feats= self.generate_samples(feat, fore, num_samples_per_batch=num, m=patchsize)
        pos_feats = self.generate_samples(feat, fore, num_samples_per_batch=num, m=patchsize)
        neg_feats = self.generate_samples(feat, back, num_samples_per_batch=num, m=patchsize)#n*b, c*h*w
        anchor_feats = self.projector(anchor_feats)
        pos_feats = self.projector(pos_feats)
        neg_feats = self.projector(neg_feats)
        loss = self.contrast_nce_fast(anchor_feats, pos_feats, neg_feats, temperature)
        return loss

    
    def forward(self, img, fore, back):
        x1 = self.inc(img)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        d1 = self.up1(x5, x4)
        d2 = self.up2(d1, x3)
        d3 = self.up3(d2, x2)
        out = self.up4(d3, x1)#feature
        output = self.outc(out)

        loss_contrast_pixel = self.compute_loss(out, fore, back, num=2048, patchsize=1, temperature=0.07)
        loss_contrast_patch = self.compute_loss(out, fore, back, num=512, patchsize=3, temperature=0.07)
        loss_contrast = 0.5*loss_contrast_pixel + 0.5*loss_contrast_patch

        return output,loss_contrast
    
class PAplusNet_cenet(nn.Module):
    def __init__(self, num_channels=4, num_classes=1, patch=3):
        super(PAplusNet_cenet, self).__init__()
        # Define the filters for each layer
        filters = [64, 128, 256, 512]
        # Use a pre-trained ResNet34 as the encoder
        resnet = models.resnet34(pretrained=True)
        self.convdata = nn.Conv2d(num_channels, 3, kernel_size=3, stride=1, padding=1)
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            DACblock(512),
            SPPblock(512)
        )
        self.decoder4 = DecoderBlock(516, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalbn = nn.BatchNorm2d(32)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

        self.get_projectors = nn.ModuleDict({
            '1': self.create_projector(32),  # patch_size=1*1*32
            '3': self.create_projector(288),  # patch_size=3
            '5': self.create_projector(800),  # patch_size=5
            '7': self.create_projector(1568),  # patch_size=7
            '9': self.create_projector(2592)   # patch_size=9
        })
        patch_size = str(patch)
        if patch_size in self.get_projectors:
            self.projector = self.get_projectors[patch_size]
        else:
            raise ValueError(f"Invalid patch size: {patch_size}. Must be one of {list(self.get_projectors.keys())}")
        
    def find_nonzero_regions(self, image, m):
        binary_image = image > 0
        kernel = torch.ones((1, 1, m, m), device=image.device)
        conv_result = F.conv2d(binary_image.float(), kernel, stride=1, padding=0)
        nonzero_coords = torch.nonzero(conv_result[:] == m * m, as_tuple=False)

        return nonzero_coords

    def generate_samples(self, feature, mask, threshold=0.5, num_samples_per_batch=10, m=1):
        mask = (mask > threshold)
        b, c, h, w = feature.shape
        if mask.sum() == 0:
            return torch.tensor([]).to(feature.device)
        all_indices = self.find_nonzero_regions(mask, m)
        sample_indices = torch.randperm(all_indices.size(0))[:num_samples_per_batch * b]
        selected_indices = all_indices[sample_indices]
        batch_indices, y_indices, x_indices = selected_indices[:, 0], selected_indices[:, 2], selected_indices[:, 3]
        # [b, c, h, w] -> [b, c, num_patches_h, num_patches_w, m, m]
        patches_unfolded = feature.unfold(2, m, 1).unfold(3, m, 1)
        patches = patches_unfolded[batch_indices, :, y_indices, x_indices]
        flattened_representation = rearrange(patches, 'n ... -> n (...)')
        projection = self.projector(flattened_representation)

        return projection

    def contrast_nce_fast(self, anchor, positive, negative, temperature=0.3):
        anchor = F.normalize(anchor, dim=1)  # [b*n, 64]
        positive = F.normalize(positive, dim=1)  # [b*n, 64]
        negative = F.normalize(negative, dim=1)  # [b*n, 64]
        anchor_positive_dot = (anchor * positive).sum(dim=1, keepdim=True) / temperature  # [b*n, 1]
        anchor_negative_dot = torch.matmul(anchor, negative.t()) / temperature  # [b*n, b*n]
        exp_pos = torch.exp(anchor_positive_dot)  # [b*n, 1]
        exp_neg = torch.exp(anchor_negative_dot)  # [b*n, b*n]
        neg_exp_sum = exp_neg.sum(dim=1, keepdim=True)  # [b*n, 1]
        loss = -torch.log(exp_pos / (exp_pos + neg_exp_sum + 1e-9))  # [b*n, 1]
        total_loss = loss.mean() 

        return total_loss

    def compute_loss(self, feat, fore, back, num, patchsize, temperature=0.07):
        anchor_feats= self.generate_samples(feat, fore, num_samples_per_batch=num, m=patchsize)
        pos_feats = self.generate_samples(feat, fore, num_samples_per_batch=num, m=patchsize)
        neg_feats = self.generate_samples(feat, back, num_samples_per_batch=num, m=patchsize)#n*b, c*h*w
        anchor_feats = self.projector(anchor_feats)
        pos_feats = self.projector(pos_feats)
        neg_feats = self.projector(neg_feats)
        loss = self.contrast_nce_fast(anchor_feats, pos_feats, neg_feats, temperature)
        return loss
    
    def forward(self, img, fore, back):
        x0 = self.convdata(img)
        # Encoder
        x1 = self.encoder[0](x0)        # x = self.firstconv(x)#12x64x128x128
        x2 = self.encoder[1](x1)        # x = self.firstbn(x)#12x64x128x128
        x3 = self.encoder[2](x2)        # x = self.firstrelu(x)#12x64x128x128
        x4 = self.encoder[3](x3)        # x = self.firstmaxpool(x)#12x64x64x64
        e1 = self.encoder[4](x4)        # e1 = self.encoder1(x) #12x64x64x64
        e2 = self.encoder[5](e1)        # e2 = self.encoder2(e1)#12x128x32x32
        e3 = self.encoder[6](e2)        # e3 = self.encoder3(e2)#12x256x16x16
        e4 = self.encoder[7](e3)        # e4 = self.encoder4(e3)#12x512x8x8
        # # Center
        e5 = self.encoder[8](e4)        # e4 = self.dblock(e4)
        e5 = self.encoder[9](e5)        # e4 = self.spp(e4)

        # Decoder
        d4 = self.decoder4(e5) + e3#12x256x16x16
        d3 = self.decoder3(d4) + e2#12x128x32x32
        d2 = self.decoder2(d3) + e1#12x64x64x64
        d1 = self.decoder1(d2)#12x64x128x128

        out = self.finaldeconv1(d1)#12x32x256x256
        out = self.finalbn(out)
        out = self.finalrelu1(out)#12x32x256x256
        out = self.finalconv2(out)#12x32x256x256
        out = self.finalrelu2(out.clone())#12x32x256x256

        output = self.finalconv3(out)#12x1x256x256
        output = nn.Sigmoid()(output)
    
        loss_contrast_pixel = self.compute_loss(out, fore, back, num=2048, patchsize=1, temperature=0.07)
        loss_contrast_patch = self.compute_loss(out, fore, back, num=512, patchsize=3, temperature=0.07)
        loss_contrast = 0.5*loss_contrast_pixel + 0.5*loss_contrast_patch

        return output,loss_contrast

class PAplusNet_densenet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, drop_rate=0, num_channels=4, num_classes=1, patch=3):
        super(PAplusNet_densenet, self).__init__()
        nb_filter = num_init_features
        eps = 1.1e-5
        
        # Initial convolution layer
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(num_channels, nb_filter, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(nb_filter, eps=eps)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        
        # Dense blocks and transition layers
        for i, num_layer in enumerate(block_config):
            block = DenseBlock(num_layer, nb_filter, growth_rate, drop_rate)
            nb_filter += num_layer * growth_rate
            self.features.add_module('denseblock%d' % (i + 1), block)
            if i != len(block_config) - 1:
                trans = TransitionLayer(nb_filter, nb_filter // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                nb_filter = nb_filter // 2
        
        self.features.add_module('norm5', nn.BatchNorm2d(nb_filter, eps=eps))
        self.features.add_module('relu5', nn.ReLU(inplace=True))
        self.conv2d5 = nn.Conv2d(32, 1, (1, 1), padding=0)
        # Decoder
        self.decode = nn.Sequential(OrderedDict([
            ('up0', nn.ConvTranspose2d(nb_filter, 512, kernel_size=2, stride=2)),
            ('conv2d0', nn.Conv2d(512, 256, kernel_size=3, padding=1)),
            ('bn0', nn.BatchNorm2d(256)),
            ('ac0', nn.ReLU(inplace=True)),

            ('up1', nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)),
            ('conv2d1', nn.Conv2d(256, 128, kernel_size=3, padding=1)),
            ('bn1', nn.BatchNorm2d(128)),
            ('ac1', nn.ReLU(inplace=True)),

            ('up2', nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)),
            ('conv2d2', nn.Conv2d(128, 64, kernel_size=3, padding=1)),
            ('bn2', nn.BatchNorm2d(64)),
            ('ac2', nn.ReLU(inplace=True)),

            ('up3', nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)),
            ('conv2d3', nn.Conv2d(64, 32, kernel_size=3, padding=1)),
            ('bn3', nn.BatchNorm2d(32)),
            ('ac3', nn.ReLU(inplace=True)),

            ('up4', nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)),
        ]))
        self.get_projectors = nn.ModuleDict({
            '1': self.create_projector(32),  # patch_size=1*1*32
            '3': self.create_projector(288),  # patch_size=3
            '5': self.create_projector(800),  # patch_size=5
            '7': self.create_projector(1568),  # patch_size=7
            '9': self.create_projector(2592)   # patch_size=9
        })
        patch_size = str(patch)
        if patch_size in self.get_projectors:
            self.projector = self.get_projectors[patch_size]
        else:
            raise ValueError(f"Invalid patch size: {patch_size}. Must be one of {list(self.get_projectors.keys())}")
        
    def find_nonzero_regions(self, image, m):
        binary_image = image > 0
        kernel = torch.ones((1, 1, m, m), device=image.device)
        conv_result = F.conv2d(binary_image.float(), kernel, stride=1, padding=0)
        nonzero_coords = torch.nonzero(conv_result[:] == m * m, as_tuple=False)

        return nonzero_coords

    def generate_samples(self, feature, mask, threshold=0.5, num_samples_per_batch=10, m=1):
        mask = (mask > threshold)
        b, c, h, w = feature.shape
        if mask.sum() == 0:
            return torch.tensor([]).to(feature.device)
        all_indices = self.find_nonzero_regions(mask, m)
        sample_indices = torch.randperm(all_indices.size(0))[:num_samples_per_batch * b]
        selected_indices = all_indices[sample_indices]
        batch_indices, y_indices, x_indices = selected_indices[:, 0], selected_indices[:, 2], selected_indices[:, 3]
        # [b, c, h, w] -> [b, c, num_patches_h, num_patches_w, m, m]
        patches_unfolded = feature.unfold(2, m, 1).unfold(3, m, 1)
        patches = patches_unfolded[batch_indices, :, y_indices, x_indices]
        flattened_representation = rearrange(patches, 'n ... -> n (...)')
        projection = self.projector(flattened_representation)

        return projection

    def contrast_nce_fast(self, anchor, positive, negative, temperature=0.3):
        anchor = F.normalize(anchor, dim=1)  # [b*n, 64]
        positive = F.normalize(positive, dim=1)  # [b*n, 64]
        negative = F.normalize(negative, dim=1)  # [b*n, 64]
        anchor_positive_dot = (anchor * positive).sum(dim=1, keepdim=True) / temperature  # [b*n, 1]
        anchor_negative_dot = torch.matmul(anchor, negative.t()) / temperature  # [b*n, b*n]
        exp_pos = torch.exp(anchor_positive_dot)  # [b*n, 1]
        exp_neg = torch.exp(anchor_negative_dot)  # [b*n, b*n]
        neg_exp_sum = exp_neg.sum(dim=1, keepdim=True)  # [b*n, 1]
        loss = -torch.log(exp_pos / (exp_pos + neg_exp_sum + 1e-9))  # [b*n, 1]
        total_loss = loss.mean() 

        return total_loss

    def compute_loss(self, feat, fore, back, num, patchsize, temperature=0.07):
        anchor_feats= self.generate_samples(feat, fore, num_samples_per_batch=num, m=patchsize)
        pos_feats = self.generate_samples(feat, fore, num_samples_per_batch=num, m=patchsize)
        neg_feats = self.generate_samples(feat, back, num_samples_per_batch=num, m=patchsize)#n*b, c*h*w
        anchor_feats = self.projector(anchor_feats)
        pos_feats = self.projector(pos_feats)
        neg_feats = self.projector(neg_feats)
        loss = self.contrast_nce_fast(anchor_feats, pos_feats, neg_feats, temperature)
        return loss

    
    def forward(self, img, fore, back):
        out = self.features(img)  #[12, 2208, 8, 8]
        out = self.decode(out)  #[12,32,256,256]
        output = self.conv2d5(out)
        output = nn.Sigmoid()(output)

        loss_contrast_pixel = self.compute_loss(out, fore, back, num=2048, patchsize=1, temperature=0.07)
        loss_contrast_patch = self.compute_loss(out, fore, back, num=512, patchsize=3, temperature=0.07)
        loss_contrast = 0.5*loss_contrast_pixel + 0.5*loss_contrast_patch

        return output,loss_contrast

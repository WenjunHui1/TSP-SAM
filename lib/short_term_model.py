import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
from abc import ABC
import torch_dct as DCT
from einops import rearrange

from lib.pvtv2_afterTEM import Network
from lib.long_term import *
from segment_anything import SamPredictor, sam_model_registry
    
class two_ConvBnRule(nn.Module):
    def __init__(self, in_chan, out_chan=64):
        super(two_ConvBnRule, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_chan,
            out_channels=out_chan,
            kernel_size=3,
            padding=1
        )
        self.BN1 = nn.BatchNorm2d(out_chan)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=out_chan,
            out_channels=out_chan,
            kernel_size=3,
            padding=1
        )
        self.BN2 = nn.BatchNorm2d(out_chan)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.BN1(feat)
        feat = self.relu1(feat)

        feat = self.conv2(feat)
        feat = self.BN2(feat)
        feat = self.relu2(feat)

        return feat

def dct_grid(img_ycbcr, grid=8):
    
    num_batchsize, c, h, w = img_ycbcr.shape
    
    img_ycbcr = img_ycbcr.reshape(num_batchsize, c, h // grid, grid, w // grid, grid).permute(0, 2, 4, 1, 3, 5)
    img_freq = DCT.dct_2d(img_ycbcr, norm='ortho')
    img_freq = img_freq.reshape(num_batchsize, h // grid, w // grid, -1).permute(0, 3, 1, 2)
    
    return img_freq

class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 base_width=64):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.))
        self.conv1 = nn.Conv2d(inplanes, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    
class Attention(nn.Module, ABC):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.to_q = nn.Linear(heads * dim_head, heads * dim_head, bias=False)
        self.to_kv = nn.Linear(heads * dim_head, heads * dim_head * 2, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity() 
        
    def forward(self, x_q, x_kv):

        b, c, _, h = *x_q.shape, self.heads
        q = self.to_q(x_q)
        q = rearrange(q, 'b c (h d) -> b h c d', h=h)
        
        kv = self.to_kv(x_kv).chunk(2, dim=-1) 
        k, v = map(lambda t: rearrange(t, 'b c (h d) -> b h c d', h=h), kv) 
        
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale 

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h c d -> b c (h d)')
        out = self.to_out(out)
        
        return out
    
class Transformer(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        
        self.norm = nn.LayerNorm([dim])
        self.cross_attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        
        self.net = nn.Sequential(
            nn.Linear(mlp_dim, mlp_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim // 2, mlp_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x_q, x_kv):
        
        x = self.cross_attn(x_q, x_kv) 
        x = self.net(x) 
        
        return self.norm(x)

class FreqEnhance(nn.Module):
    def __init__(self, args):
        super(FreqEnhance, self).__init__()

        self.conv_freq1 = two_ConvBnRule(96, 96)
        self.conv_freq2 = two_ConvBnRule(96, 96)
        self.conv_freq3 = two_ConvBnRule(192, 192)

        self.high_band = Transformer(dim=96, heads=2, dim_head=48, mlp_dim=96, dropout=0.)
        self.low_band = Transformer(dim=96, heads=2, dim_head=48, mlp_dim=96, dropout=0.)

    def forward(self, img1_dct, img2_dct):
        # YCbCr
        feat1_y, feat1_Cb, feat1_Cr = img1_dct[:, :64, :, :], img1_dct[:, 64:128, :, :], img1_dct[:, 128:, :, :]
        ori_feat1_DCT = img1_dct
        
        feat2_y, feat2_Cb, feat2_Cr = img2_dct[:, :64, :, :], img2_dct[:, 64:128, :, :], img2_dct[:, 128:, :, :]
        
        # high-low freq 
        high1 = torch.cat([feat1_y[:, 32:, :, :], feat1_Cb[:, 32:, :, :], feat1_Cr[:, 32:, :, :]], 1)
        low1 = torch.cat([feat1_y[:, :32, :, :], feat1_Cb[:, :32, :, :], feat1_Cr[:, :32, :, :]], 1)
        
        high2 = torch.cat([feat2_y[:, 32:, :, :], feat2_Cb[:, 32:, :, :], feat2_Cr[:, 32:, :, :]], 1)
        low2 = torch.cat([feat2_y[:, :32, :, :], feat2_Cb[:, :32, :, :], feat2_Cr[:, :32, :, :]], 1)
        
        # band-wise freq corr
        high1 = rearrange(high1, 'b c h w -> b c (h w)').transpose(1, 2)
        high2 = rearrange(high2, 'b c h w -> b c (h w)').transpose(1, 2)
        
        low1 = rearrange(low1, 'b c h w -> b c (h w)').transpose(1, 2)
        low2 = rearrange(low2, 'b c h w -> b c (h w)').transpose(1, 2)
        
        high_corr = self.high_band(high2, high1).transpose(1, 2)
        low_corr = self.low_band(low2, low1).transpose(1, 2)
        
        high_corr = rearrange(high_corr, 'b c (h w) -> b c h w', h=img1_dct.shape[-2])
        low_corr = rearrange(low_corr, 'b c (h w) -> b c h w', h=img1_dct.shape[-2])
        
        high_corr = self.conv_freq1(high_corr)
        low_corr = self.conv_freq2(low_corr)
        
        high_y, high_b, high_r = torch.split(high_corr, 32, 1)
        low_y, low_b, low_r = torch.split(low_corr, 32, 1)
        
        feat_y = torch.cat([low_y, high_y], 1)
        feat_Cb = torch.cat([low_b, high_b], 1)
        feat_Cr = torch.cat([low_r, high_r], 1)
        
        feat_DCT = torch.cat((torch.cat((feat_y, feat_Cb), 1), feat_Cr), 1) # concat
        feat_DCT = self.conv_freq3(feat_DCT) + ori_feat1_DCT
        
        return feat_DCT
        
class SpaFreq(nn.Module):
    def __init__(self, args):
        super(SpaFreq, self).__init__()

        self.conv_freq20 = two_ConvBnRule(192, 64)
        self.conv_freq21 = two_ConvBnRule(64)
        
        self.conv_freq30 = two_ConvBnRule(192, 64)
        self.conv_freq31 = two_ConvBnRule(64)
        
        self.conv_freq40 = two_ConvBnRule(192, 64)
        self.conv_freq41 = two_ConvBnRule(64)

        self.fusion2 = Transformer(dim=64, heads=2, dim_head=32, mlp_dim=64, dropout=0.)
        self.fusion3 = Transformer(dim=64, heads=2, dim_head=32, mlp_dim=64, dropout=0.)
        self.fusion4 = Transformer(dim=64, heads=2, dim_head=32, mlp_dim=64, dropout=0.)

    def forward(self, fmap, freq12):
    
        x2_rfb, x3_rfb, x4_rfb = fmap
        
        freq_to2 = self.conv_freq20(freq12)
        freq_to2 = F.interpolate(freq_to2, size=x2_rfb.shape[-2:], mode='bilinear', align_corners=True)
        freq_to2 = rearrange(freq_to2, 'b c h w -> b c (h w)').transpose(1, 2)
        feat2 = rearrange(x2_rfb, 'b c h w -> b c (h w)').transpose(1, 2)
        feat2 = self.fusion2(freq_to2, feat2).transpose(1, 2)
        feat2 = rearrange(feat2, 'b c (h w) -> b c h w', h=x2_rfb.shape[-2])
        feat2 = self.conv_freq21(feat2) + x2_rfb 
        
        freq_to3 = self.conv_freq30(freq12)
        freq_to3 = F.interpolate(freq_to3, size=x3_rfb.shape[-2:], mode='bilinear', align_corners=True)
        freq_to3 = rearrange(freq_to3, 'b c h w -> b c (h w)').transpose(1,2)
        feat3 = rearrange(x3_rfb, 'b c h w -> b c (h w)').transpose(1,2)
        feat3 = self.fusion3(freq_to3, feat3).transpose(1, 2)
        feat3 = rearrange(feat3, 'b c (h w) -> b c h w', h=x3_rfb.shape[-2])
        feat3 = self.conv_freq31(feat3) + x3_rfb 
        
        freq_to4 = self.conv_freq40(freq12)
        freq_to4 = F.interpolate(freq_to4, size=x4_rfb.shape[-2:], mode='bilinear', align_corners=True)
        freq_to4 = rearrange(freq_to4, 'b c h w -> b c (h w)').transpose(1,2)
        feat4 = rearrange(x4_rfb, 'b c h w -> b c (h w)').transpose(1,2)
        feat4 = self.fusion4(freq_to4, feat4).transpose(1, 2)
        feat4 = rearrange(feat4, 'b c (h w) -> b c h w', h=x4_rfb.shape[-2])
        feat4 = self.conv_freq41(feat4) + x4_rfb 

        return feat2, feat3, feat4
        
class Temporal_injector(nn.Module):
    def __init__(self, args):
        super(Temporal_injector, self).__init__()

        self.conv_temp0 = two_ConvBnRule(256, 64)
        self.conv_temp1 = two_ConvBnRule(256, 64)
        self.conv_temp2 = two_ConvBnRule(64, 256)

        self.injector = Transformer(dim=64, heads=2, dim_head=32, mlp_dim=64, dropout=0.)
        
    def forward(self, feat_sam, feat_seq):

        feat_seq = F.interpolate(feat_seq, size=feat_sam.shape[-2:], mode='bilinear', align_corners=True)
        
        feat_sam = self.conv_temp0(feat_sam)
        feat_seq = self.conv_temp1(feat_seq)
        
        feat_sam = rearrange(feat_sam, 'b c h w -> b c (h w)').transpose(1,2)
        feat_seq = rearrange(feat_seq, 'b c h w -> b c (h w)').transpose(1,2)
        
        feat = self.injector(feat_sam, feat_seq).transpose(1, 2)
        feat = rearrange(feat, 'b c (h w) -> b c h w', h=feat.shape[-2])
        feat = self.conv_temp2(feat) 
        
        return feat
        
class Location_rectification(nn.Module):
    def __init__(self, args):
        super(Location_rectification, self).__init__()
        
        self.rect_query = nn.Parameter(torch.rand((1, 256, 4)),requires_grad=True)
        self.rect_extractor = Transformer(dim=4, heads=4, dim_head=64, mlp_dim=4, dropout=0.3)
        
    def forward(self, temp): 
        
        q = (self.rect_query.repeat_interleave(temp.shape[0], dim=0)).transpose(1, 2)
        kv = rearrange(temp, 'b c h w -> b c (h w)').transpose(1,2)
        
        rect = self.rect_extractor(q, kv).transpose(1, 2)
        rect = torch.mean(rect, dim=1, keepdim=True)
        
        return rect
        
model_type = "vit_h"
sam_checkpoint = "./model_checkpoint/sam_vit_h_4b8939.pth"
        
#short-term网络模型
class VideoModel(nn.Module):
    def __init__(self, args):
        super(VideoModel, self).__init__()

        self.args = args

        # backbone
        self.backbone = Network(channel=64, pretrained=True, imgsize=self.args.trainsize)
        
        self.freq = FreqEnhance(self.args)
        self.fusion = SpaFreq(self.args)
        self.fusion_conv = nn.Sequential(nn.Conv2d(2, 64, 3, 1, 1),
                                         nn.Conv2d(64, 64, 3, 1, 1),
                                         nn.Conv2d(64, 1, 3, 1, 1),
            )
        
        # sparse prompt head
        self.fusion_conv0 =  two_ConvBnRule(64, 64)
        self.fusion_conv1 =  two_ConvBnRule(64, 64)
        self.fusion_conv2 =  two_ConvBnRule(64, 64)
        
        self.convblock0 = self._make_layer(Bottleneck, 64, 2, stride=1) 
        self.convblock1 = self._make_layer(Bottleneck, 64, 2, stride=1) 
        self.convblock2 = self._make_layer(Bottleneck, 64, 2, stride=1)
        
        self.out_conv = nn.Sequential(nn.Conv2d(64, 4, 3, 1, 1),
                                      nn.BatchNorm2d(4),
                                      nn.ReLU(inplace=True),)
        self.avgpool = nn.AdaptiveAvgPool2d((1))

        # temporal-spatial injector
        self.resnet3D = Resnet3D()
        self.temporal_injector = Temporal_injector(self.args)
        self.location_rect = Location_rectification(args)

        # sam 
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).cuda(self.args.gpu_ids[0])
       
    def forward(self, x, x_ycbcr, x_sam):
        
        image1 = x[-1]
        fmap1 = self.backbone.feat_net(image1)

        # dct
        image1_ycbcr, image2_ycbcr, image3_ycbcr = x_ycbcr[-1], x_ycbcr[-2], x_ycbcr[-3]

        image1_dct = dct_grid(image1_ycbcr, self.args.grid)
        image2_dct = dct_grid(image2_ycbcr, self.args.grid)
        image3_dct = dct_grid(image3_ycbcr, self.args.grid)
        
        # Motion in freqency
        freq12 = self.freq(image1_dct, image2_dct)
        freq13 = self.freq(image1_dct, image3_dct)

        # frequency-spatial fusion
        corr_vol12 = self.fusion(fmap1, freq12)
        corr_vol13 = self.fusion(fmap1, freq13)

        # dense prompt head
        out12 = self.backbone.decoder(corr_vol12)
        out13 = self.backbone.decoder(corr_vol13)

        concated = torch.cat([out12[-1], out13[-1]], dim=1)
        out = self.fusion_conv(concated)
        
        # sparse prompt head
        x20, x30, x40 = corr_vol12 
        x21, x31, x41 = corr_vol13
        
        concated = x20 + x21 
        out2 = self.fusion_conv0(concated)
        
        concated = x30 + x31 
        out3 = self.fusion_conv1(concated)
        
        concated = x40 + x41 
        out4 = self.fusion_conv2(concated)
        
        out4 = self.convblock0(out4)
        out4 = F.interpolate(out4, out3.shape[-2:], mode='bilinear', align_corners=True)
        out34 = out3 + out4 
        out34 = self.convblock1(out34)
        out34 = F.interpolate(out34, out2.shape[-2:], mode='bilinear', align_corners=True)
        out234 = out2 + out34 
        out234 = self.convblock2(out234)
        
        out_prompt = self.out_conv(out234)
        out_prompt = self.avgpool(out_prompt).squeeze(2).squeeze(2)
        
        # sam 
        self.predictor = SamPredictor(self.sam)
        
        original_image_size = x_sam.shape[-2:]
        target_size = self.sam.image_encoder.img_size
        image_sam = F.interpolate(x_sam, (target_size, target_size), mode='bilinear')
        
        self.predictor.set_torch_image(image_sam, image_sam.shape[-2:])
        
        # long-range temporal-spatial injection and rect
        image_seq = torch.stack(x).permute(1, 2, 0, 3, 4) 
        feat_seq = self.resnet3D(image_seq)

        feat_sam = self.predictor.get_image_embedding()
        feat_temp = self.temporal_injector(feat_sam, feat_seq)
        
        rect = self.location_rect(feat_temp)
        sparse_prompt = (out_prompt * target_size).unsqueeze(1) + rect
        
        dense_prompt = F.interpolate(out, (256, 256), mode='bilinear')
        _, _, low_masks = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=sparse_prompt,
            mask_input=dense_prompt,
            multimask_output=False,
            feat_sam_temp=(feat_sam+feat_temp))

        out_sam = F.interpolate(low_masks, original_image_size, mode='bilinear')

        return out12, out13, out, sparse_prompt, out_sam
    
    def _make_layer(self, block, planes, blocks, stride): 
        layers = self._layer(block, planes, blocks, stride)
        return nn.Sequential(*layers)

    def _layer(self, block, planes, blocks, stride):
        downsample = get_downsampling_layer(64, block, planes, stride)

        layers = [block(64, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return layers
        
def get_downsampling_layer(inplanes, block, planes, stride):
    
    outplanes = planes * block.expansion
    
    if stride == 1 and inplanes == outplanes:
        return
    else:
        return nn.Sequential(
            nn.Conv2d(inplanes, outplanes, 1, stride, bias=False),
            nn.BatchNorm2d(outplanes),
        )


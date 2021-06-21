import torch
from torch import nn
from einops import repeat
from einops.layers.torch import Rearrange
from modules.vit_mod import ConvAttention, PreNorm, FeedForward
import numpy as np
import torch.nn.functional as F
import math

class Transformer(nn.Module):
    def __init__(self, dim, img_size, depth, heads, dim_head, mlp_dim, dropout=0., last_stage=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, ConvAttention(dim, img_size, heads=heads, dim_head=dim_head, dropout=dropout, last_stage=last_stage)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)
        # print('out_norm',out_normal.shape)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            #pdb.set_trace()
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, dilation=self.conv.dilation, groups=self.conv.groups)

            return out_normal - self.theta * out_diff

"""
basic_conv(64, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU())   
"""

class mutli_conv(nn.Module):

    def __init__(self, in_channels,out_channels,kernel_size,stride,padding,theta =0.7):
        super(mutli_conv,self).__init__()
        self.conv_cd =  nn.Sequential(Conv2d_cd(in_channels//2, out_channels, kernel_size=kernel_size,
                                                stride=stride, padding=padding, bias=False, theta= theta),
                                      nn.LeakyReLU(0.1))
        self.conv_f = nn.Sequential(nn.Conv2d(in_channels//2, out_channels, kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=1, groups=3),
                                    nn.LeakyReLU(0.1))
        self.frag= in_channels // 2


    def forward(self, x):
        texture, frequence = x.split(self.frag,1)
        # print('txture', texture.shape)
        # print('frequence', frequence.shape)
        texture = self.conv_cd(texture)
        frequence = self.conv_f(frequence)
        # print('txture', texture.shape)
        # print('frequence', frequence.shape)
        feat = torch.add(texture,frequence)
        return feat





class CvT(nn.Module):
    def __init__(self, image_size, in_channels, num_classes, dim=72, kernels=[5, 3, 3], strides=[4, 2, 2],
                 heads=[1, 3, 6] , depth = [1, 2, 10], pool='cls', dropout=0.1, emb_dropout=0., scale_dim=4):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = pool
        self.dim = dim

        ##### Stage 1 #######
        self.stage1_conv_embed = nn.Sequential(
            # nn.Conv2d(in_channels, dim, kernels[0], strides[0], 2),
            mutli_conv(in_channels,dim,kernels[0],strides[0],padding=2),
            Rearrange('b c h w -> b (h w) c', h=image_size // 4, w=image_size // 4),
            nn.LayerNorm(dim)
        )
        self.stage1_transformer = nn.Sequential(
            Transformer(dim=dim, img_size=image_size // 4, depth=depth[0], heads=heads[0], dim_head=self.dim,
                        mlp_dim=dim * scale_dim, dropout=dropout),
            Rearrange('b (h w) c -> b c h w', h=image_size // 4, w=image_size // 4)
        )

        ##### Stage 2 #######
        in_channels = dim
        scale = heads[1] // heads[0]
        dim = scale * dim
        self.stage2_conv_embed = nn.Sequential(
            # nn.Conv2d(in_channels, dim, kernels[1], strides[1], 1),
            mutli_conv(in_channels, dim, kernels[1], strides[1], padding=1),
            Rearrange('b c h w -> b (h w) c', h=image_size // 8, w=image_size // 8),
            nn.LayerNorm(dim)
        )
        self.stage2_transformer = nn.Sequential(
            Transformer(dim=dim, img_size=image_size // 8, depth=depth[1], heads=heads[1], dim_head=self.dim,
                        mlp_dim=dim * scale_dim, dropout=dropout),
            Rearrange('b (h w) c -> b c h w', h=image_size // 8, w=image_size // 8)
        )

        ##### Stage 3 #######
        in_channels = dim
        scale = heads[2] // heads[1]
        dim = scale * dim
        self.stage3_conv_embed = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernels[2], strides[2], 1),
            Rearrange('b c h w -> b (h w) c', h=image_size // 16, w=image_size // 16),
            nn.LayerNorm(dim)
        )
        self.stage3_transformer = nn.Sequential(
            Transformer(dim=dim, img_size=image_size // 16, depth=depth[2], heads=heads[2], dim_head=self.dim,
                        mlp_dim=dim * scale_dim, dropout=dropout, last_stage=True),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout_large = nn.Dropout(emb_dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):

        xs = self.stage1_conv_embed(img)
        # print('stage1 conv {}'.format(xs.shape))
        xs = self.stage1_transformer(xs)
        # print('stage1 transf {}'.format(xs.shape))

        xs = self.stage2_conv_embed(xs)
        # print('stage2 conv {}'.format(xs.shape))
        xs = self.stage2_transformer(xs)
        # print('stage2 transf {}'.format(xs.shape))

        xs = self.stage3_conv_embed(xs)
        b, n, _ = xs.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        xs = torch.cat((cls_tokens, xs), dim=1)
        xs = self.stage3_transformer(xs)
        xs = xs.mean(dim=1) if self.pool == 'mean' else xs[:, 0]
        # feat = xs
        xs = self.mlp_head(xs)
        return xs


if __name__ == "__main__":
    img = torch.ones([128, 18, 256, 256])

    model = CvT(256, 18, 2)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)

    out = model(img)

    print("Shape of out :", out)  # [B, num_classes]
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



div_size = 16
div_flow = 20.0

def warp(img, flow):
    B, _, H, W = flow.shape
    xx = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1)
    yy = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W)
    grid = torch.cat([xx, yy], 1).to(img)
    flow_ = torch.cat([flow[:, 0:1, :, :] / ((W - 1.0) / 2.0), flow[:, 1:2, :, :] / ((H - 1.0) / 2.0)], 1)
    grid_ = (grid + flow_).permute(0, 2, 3, 1)
    output = F.grid_sample(input=img, grid=grid_, mode='bilinear', padding_mode='border', align_corners=True)
    return output

def weight_3expo_low_tog17(img):
    w = torch.zeros_like(img)
    mask1 = img < 0.5
    w[mask1] = 0.0
    mask2 = img >= 0.50
    w[mask2] = img[mask2] - 0.5
    w /= 0.5
    return w

def weight_3expo_mid_tog17(img):
    w = torch.zeros_like(img)
    mask1 = img < 0.5
    w[mask1] = img[mask1]
    mask2 = img >= 0.5
    w[mask2] = 1.0 - img[mask2]
    w /= 0.5
    return w

def weight_3expo_high_tog17(img):
    w = torch.zeros_like(img)
    mask1 = img < 0.5
    w[mask1] = 0.5 - img[mask1]
    mask2 = img >= 0.5
    w[mask2] = 0.0
    w /= 0.5
    return w

def merge_hdr(ldr_imgs, lin_imgs, mask0, mask2):
    sum_img = torch.zeros_like(ldr_imgs[1])
    sum_w = torch.zeros_like(ldr_imgs[1])
    w_low = weight_3expo_low_tog17(ldr_imgs[1]) * mask0
    w_mid = weight_3expo_mid_tog17(ldr_imgs[1]) + weight_3expo_low_tog17(ldr_imgs[1]) * (1.0 - mask0) + weight_3expo_high_tog17(ldr_imgs[1]) * (1.0 - mask2)
    w_high = weight_3expo_high_tog17(ldr_imgs[1]) * mask2
    w_list = [w_low, w_mid, w_high]
    for i in range(len(ldr_imgs)):
        sum_w += w_list[i]
        sum_img += w_list[i] * lin_imgs[i]
    hdr_img = sum_img / (sum_w + 1e-9)
    return hdr_img

def resize(x, scale_factor):
    return F.interpolate(x, scale_factor=scale_factor, mode="bilinear", align_corners=False, recompute_scale_factor=True)

def convrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias), 
        nn.PReLU(out_channels)
    )

def deconv(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)

def channel_shuffle(x, groups):
    b, c, h, w = x.size()
    channels_per_group = c // groups
    x = x.view(b, groups, channels_per_group, h, w)
    x = x.transpose(1, 2).contiguous()
    x = x.view(b, -1, h, w)
    return x


class Encoder(nn.Module):
    def __init__(self, in_channels=4):
        super(Encoder, self).__init__()
        self.pyramid1 = nn.Sequential(
            convrelu(in_channels, 40, 3, 2, 1), 
            convrelu(40, 40, 3, 1, 1)
        )
        self.pyramid2 = nn.Sequential(
            convrelu(40, 40, 3, 2, 1), 
            convrelu(40, 40, 3, 1, 1)
        )
        self.pyramid3 = nn.Sequential(
            convrelu(40, 40, 3, 2, 1), 
            convrelu(40, 40, 3, 1, 1)
        )
        self.pyramid4 = nn.Sequential(
            convrelu(40, 40, 3, 2, 1), 
            convrelu(40, 40, 3, 1, 1)
        )
        
    def forward(self, img_c):
        f1 = self.pyramid1(img_c)
        f2 = self.pyramid2(f1)
        f3 = self.pyramid3(f2)
        f4 = self.pyramid4(f3)
        return f1, f2, f3, f4


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = convrelu(126, 120)
        self.conv2 = convrelu(120, 120, groups=3)
        self.conv3 = convrelu(120, 120, groups=3)
        self.conv4 = convrelu(120, 120, groups=3)
        self.conv5 = convrelu(120, 120)
        self.conv6 = deconv(120, 6)

    def forward(self, f0, f1, f2, flow0, flow2, mask0, mask2):
        f0_warp = warp(f0, flow0)
        f2_warp = warp(f2, flow2)
        f_in = torch.cat([f0_warp, f1, f2_warp, flow0, flow2, mask0, mask2], 1)
        f_out = self.conv1(f_in)
        f_out = channel_shuffle(self.conv2(f_out), 3)
        f_out = channel_shuffle(self.conv3(f_out), 3)
        f_out = channel_shuffle(self.conv4(f_out), 3)
        f_out = self.conv5(f_out)
        f_out = self.conv6(f_out)
        up_flow0 = 2.0 * resize(flow0, scale_factor=2.0) + f_out[:, 0:2]
        up_flow2 = 2.0 * resize(flow2, scale_factor=2.0) + f_out[:, 2:4]
        up_mask0 = resize(mask0, scale_factor=2.0) + f_out[:, 4:5]
        up_mask2 = resize(mask2, scale_factor=2.0) + f_out[:, 5:6]
        return up_flow0, up_flow2, up_mask0, up_mask2


class ResBlock(nn.Module):
    def __init__(self, channels, dilation=1, bias=True):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=bias), 
            nn.PReLU(channels)
        )
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=bias)
        self.prelu = nn.PReLU(channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.prelu(x + out)
        return out


class RefineNet(nn.Module):
    def __init__(self, img_channels=4):
        super(RefineNet, self).__init__()
        # 适当增大模型规模 (目标: <100G, <0.5M RefineNet params)
        # img_channels=4
        c0 = 12
        c1 = 24
        c2 = 12
        total_c = c0 + c1 + c2 # 48 channels
        
        self.conv0 = nn.Sequential(convrelu(img_channels, c0), convrelu(c0, c0))
        # img1_c (4) + flow0 (2) + flow2 (2) + mask0 (1) + mask2 (1) + img_hdr_m (3) = 13
        self.conv1 = nn.Sequential(convrelu(img_channels+2+2+1+1+3, c1), convrelu(c1, c1))
        self.conv2 = nn.Sequential(convrelu(img_channels, c2), convrelu(c2, c2))

        # 恢复 ResBlock 数量至 5，通道数 48
        self.resblock1 = ResBlock(total_c, 1)
        self.resblock2 = ResBlock(total_c, 2)
        self.resblock3 = ResBlock(total_c, 4)
        self.resblock4 = ResBlock(total_c, 2)
        self.resblock5 = ResBlock(total_c, 1)
        
        self.conv3 = nn.Conv2d(total_c, 12, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        
    def forward(self, img0_c, img1_c, img2_c, flow0, flow2, mask0, mask2, img_hdr_m):
        feat0 = self.conv0(img0_c)
        feat1 = self.conv1(torch.cat([img1_c, flow0 / div_flow, flow2 / div_flow, mask0, mask2, img_hdr_m], 1))
        feat2 = self.conv2(img2_c)
        feat0_warp = warp(feat0, flow0)
        feat2_warp = warp(feat2, flow2)
        feat = torch.cat([feat0_warp, feat1, feat2_warp], 1)
        feat = self.resblock1(feat)
        feat = self.resblock2(feat)
        feat = self.resblock3(feat)
        feat = self.resblock4(feat)
        feat = self.resblock5(feat)
        res = self.pixel_shuffle(self.conv3(feat))
        
        # img_hdr_m is 384x768, we need 768x1536
        img_hdr_m_up = F.interpolate(img_hdr_m, scale_factor=2, mode="bilinear", align_corners=False)
        img_hdr_r = torch.clamp(img_hdr_m_up + res, 0, 1)
        return img_hdr_r


class SAFNet(nn.Module):
    def __init__(self):
        super(SAFNet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.refinenet = RefineNet()

    def forward_flow_mask(self, img0_c, img1_c, img2_c, scale_factor=0.5):
        h, w = img1_c.shape[-2:]
        org_size = (int(h), int(w))
        input_size = (int(div_size * np.ceil(h * scale_factor / div_size)), int(div_size * np.ceil(w * scale_factor / div_size)))

        if input_size != org_size:
            img0_c = F.interpolate(img0_c, size=input_size, mode='bilinear', align_corners=False)
            img1_c = F.interpolate(img1_c, size=input_size, mode='bilinear', align_corners=False)
            img2_c = F.interpolate(img2_c, size=input_size, mode='bilinear', align_corners=False)

        f0_1, f0_2, f0_3, f0_4 = self.encoder(img0_c)
        f1_1, f1_2, f1_3, f1_4 = self.encoder(img1_c)
        f2_1, f2_2, f2_3, f2_4 = self.encoder(img2_c)

        up_flow0_5 = torch.zeros_like(f1_4[:, 0:2, :, :])
        up_flow2_5 = torch.zeros_like(f1_4[:, 0:2, :, :])
        up_mask0_5 = torch.zeros_like(f1_4[:, 0:1, :, :])
        up_mask2_5 = torch.zeros_like(f1_4[:, 0:1, :, :])
        up_flow0_4, up_flow2_4, up_mask0_4, up_mask2_4 = self.decoder(f0_4, f1_4, f2_4, up_flow0_5, up_flow2_5, up_mask0_5, up_mask2_5)
        up_flow0_3, up_flow2_3, up_mask0_3, up_mask2_3 = self.decoder(f0_3, f1_3, f2_3, up_flow0_4, up_flow2_4, up_mask0_4, up_mask2_4)
        up_flow0_2, up_flow2_2, up_mask0_2, up_mask2_2 = self.decoder(f0_2, f1_2, f2_2, up_flow0_3, up_flow2_3, up_mask0_3, up_mask2_3)
        up_flow0_1, up_flow2_1, up_mask0_1, up_mask2_1 = self.decoder(f0_1, f1_1, f2_1, up_flow0_2, up_flow2_2, up_mask0_2, up_mask2_2)

        if input_size != org_size:
            scale_h = org_size[0] / input_size[0]
            scale_w = org_size[1] / input_size[1]
            up_flow0_1 = F.interpolate(up_flow0_1, size=org_size, mode='bilinear', align_corners=False)
            up_flow0_1[:, 0, :, :] *= scale_w
            up_flow0_1[:, 1, :, :] *= scale_h
            up_flow2_1 = F.interpolate(up_flow2_1, size=org_size, mode='bilinear', align_corners=False)
            up_flow2_1[:, 0, :, :] *= scale_w
            up_flow2_1[:, 1, :, :] *= scale_h
            up_mask0_1 = F.interpolate(up_mask0_1, size=org_size, mode='bilinear', align_corners=False)
            up_mask2_1 = F.interpolate(up_mask2_1, size=org_size, mode='bilinear', align_corners=False)

        up_mask0_1 = torch.sigmoid(up_mask0_1)
        up_mask2_1 = torch.sigmoid(up_mask2_1)

        return up_flow0_1, up_flow2_1, up_mask0_1, up_mask2_1
    
    def forward(self, x, scale_factor=0.5, refine=True):
        # x: (B, 36, 384, 768) -> 9 frames * 4 channels (Bayer packing)
        # 我们选取三个代表帧: 0 (Short), 4 (Mid), 8 (Long)
        img0_c = x[:, 0:4, :, :]
        img1_c = x[:, 16:20, :, :]
        img2_c = x[:, 32:36, :, :]

        # imgx_c[:, 0:3] linear domain, imgx_c[:, 3:6] ldr domain
        flow0, flow2, mask0, mask2 = self.forward_flow_mask(img0_c, img1_c, img2_c, scale_factor=scale_factor)

        img0_c_warp = warp(img0_c, flow0)
        img2_c_warp = warp(img2_c, flow2)
        
        # 针对 Bayer Packing (4通道)，原本 merge_hdr 是针对 6 通道 (ldr+linear) 的。
        # 这里取前 3 通道 (G1, R, B) 占位融合
        img_hdr_m = merge_hdr(
            [img0_c_warp[:, 0:3, :, :], img1_c[:, 0:3, :, :], img2_c_warp[:, 0:3, :, :]], 
            [img0_c_warp[:, 0:3, :, :], img1_c[:, 0:3, :, :], img2_c_warp[:, 0:3, :, :]], 
            mask0, mask2
            )
        
        if refine == True:
            img_hdr_r = self.refinenet(img0_c, img1_c, img2_c, flow0, flow2, mask0, mask2, img_hdr_m)
            return img_hdr_r
        else:
            return F.interpolate(img_hdr_m, scale_factor=2, mode="bilinear", align_corners=False)

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modelviz = SAFNet().to(device)
    # 4. get_model_complexity_info 打印运算浮点数和参数量
    from ptflops import get_model_complexity_info
    # 输入适配为 (36, 384, 768)
    macs, params = get_model_complexity_info(modelviz, (36, 384, 768), verbose=True, print_per_layer_stat=True)
    print(macs, params)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from fvcore.nn import FlopCountAnalysis, flop_count_table
import numbers
from einops import rearrange

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
    def __init__(self):
        super(Encoder, self).__init__()
        # 输入通道为 4 (RAW 图)
        self.pyramid1 = nn.Sequential(
            convrelu(4, 40, 3, 2, 1), 
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
        # 【修改点】：大幅缩减通道数和层数以降低 FLOPs
        # 输入通道: f_oth_warp(40) + f_ref(40) + flow(2) = 82
        self.conv1 = convrelu(82, 48)
        # 使用 groups=4 进行分组卷积，能削减此处 75% 的计算量
        self.conv2 = convrelu(48, 48, groups=4)
        self.conv3 = convrelu(48, 48, groups=4)
        # 降维过渡
        self.conv4 = convrelu(48, 32)
        # 最后一层直接使用反卷积上采样并输出2通道光流 (减少了原版的 conv5 过渡)
        self.conv5 = deconv(32, 2)

    def forward(self, f_oth, f_ref, flow):
        f_oth_warp = warp(f_oth, flow)
        f_in = torch.cat([f_oth_warp, f_ref, flow], 1)
        
        f_out = self.conv1(f_in)
        # 配合分组卷积使用 channel_shuffle 打乱通道特征
        f_out = channel_shuffle(self.conv2(f_out), 4)
        f_out = channel_shuffle(self.conv3(f_out), 4)
        f_out = self.conv4(f_out)
        f_out = self.conv5(f_out)
        
        up_flow = 2.0 * resize(flow, scale_factor=2.0) + f_out[:, 0:2]
        return up_flow

# ================= StarNet 组件开始 =================

class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


def Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias),
        # nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups=1, bias=bias)
    )


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, bias):
        super(FeedForward, self).__init__()


        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias, groups=dim),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=bias),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias, groups=dim),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=bias),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = F.gelu(x1) * x2
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias, groups=dim),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=bias),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias, groups=dim),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=bias),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias, groups=dim),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=bias),
        )

    def forward(self, x):
        b, c, h, w = x.shape


        q = rearrange(self.conv1(x), 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(self.conv2(x), 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(self.conv3(x), 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

# ================= StarNet 组件结束 =================

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class Restormer(nn.Module):
    def __init__(self,
                 out_channels=60,
                 dim=48,
                 num_blocks=[1, 1, 1, 1],
                 num_refinement_blocks=1,
                 heads=[1, 2, 4, 8],
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 dual_pixel_task=False,  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 ):

        super(Restormer, self).__init__()

        # self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1],
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2],
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3],
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2],
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1],
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0],
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0],
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################

        self.output = Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):

        # inp_enc_level1 = self.patch_embed(inp_img)
        inp_enc_level1 = inp_img
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1

class RawNet(nn.Module):
    def __init__(self, num_frames=9):
        super().__init__()
        self.num_frames = num_frames
        self.encoder = Encoder()
        self.decoder = Decoder()
        
        # 使用基于 StarNet 架构的融合网络
        #self.fusion = StarFusion(in_channels=self.num_frames * 4, out_channels=4, base_dim=32, depths=[2, 2, 2])
        self.fusion = Restormer(
            out_channels=4,
            dim = 36,
            num_blocks = [1,1,1,2],
            num_refinement_blocks = 1,
            heads = [1,2,4,8],
            bias = False,
            LayerNorm_type = 'withBias',   ## Other option 'BiasFree'
            dual_pixel_task = True       ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
        )
        # 转置卷积上采样模块 (放大2倍尺寸，并输出3通道RGB)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(4, 32, kernel_size=4, stride=2, padding=1),
            nn.PReLU(32),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        )

    def forward_flow(self, imgs, scale_factor=0.5):
        h, w = imgs[0].shape[-2:]
        org_size = (int(h), int(w))
        input_size = (int(div_size * np.ceil(h * scale_factor / div_size)), 
                      int(div_size * np.ceil(w * scale_factor / div_size)))

        imgs_resized = []
        for img in imgs:
            if input_size != org_size:
                imgs_resized.append(F.interpolate(img, size=input_size, mode='bilinear', align_corners=False))
            else:
                imgs_resized.append(img)

        # 提取全部帧特征
        feats = [self.encoder(img) for img in imgs_resized]
        ref_feats = feats[0]   # tuple: (f1, f2, f3, f4)
        oth_feats = feats[1:]  # list of tuples

        num_oth = self.num_frames - 1
        
        # 初始化光流矩阵 (不再需要掩码)
        flows_5 = [torch.zeros_like(ref_feats[3][:, 0:2, :, :]) for _ in range(num_oth)]

        # 金字塔多尺度解码
        def decode_level(f_idx_oth, f_idx_ref, prev_flows):
            cur_flows = []
            for i in range(num_oth):
                f = self.decoder(oth_feats[i][f_idx_oth], ref_feats[f_idx_ref], prev_flows[i])
                cur_flows.append(f)
            return cur_flows

        flows_4 = decode_level(3, 3, flows_5)
        flows_3 = decode_level(2, 2, flows_4)
        flows_2 = decode_level(1, 1, flows_3)
        flows_1 = decode_level(0, 0, flows_2)

        final_flows = []
        
        # 尺寸恢复
        for i in range(num_oth):
            f = flows_1[i]
            if input_size != org_size:
                scale_h = org_size[0] / input_size[0]
                scale_w = org_size[1] / input_size[1]
                f = F.interpolate(f, size=org_size, mode='bilinear', align_corners=False)
                f[:, 0, :, :] *= scale_w
                f[:, 1, :, :] *= scale_h
            final_flows.append(f)

        return final_flows
    
    def forward(self, imgs, scale_factor=0.5): 
        """
        Args:
            imgs: shape 为 (B, 9, 4, H, W) 或 (B, 36, H, W) 的张量
        """
        if isinstance(imgs, torch.Tensor):
            if imgs.dim() == 4:
                b, c, h, w = imgs.shape
                expected_channels = self.num_frames * 4
                if c != expected_channels:
                    raise ValueError(
                        f"RawNet expects {expected_channels} channels for 4D input, got {c}"
                    )
                imgs = imgs.view(b, self.num_frames, 4, h, w)
            elif imgs.dim() != 5:
                raise ValueError(
                    f"RawNet expects 4D or 5D tensor input, got shape {tuple(imgs.shape)}"
                )
            imgs = [imgs[:, i, :, :, :] for i in range(imgs.size(1))]
            
        ref_img = imgs[0]
        oth_imgs = imgs[1:]
        
        # 计算所有的配准 Flow (无 mask)
        flows = self.forward_flow(imgs, scale_factor=scale_factor)

        # 扭曲其余所有帧
        oth_imgs_warp = []
        for i in range(self.num_frames - 1):
            oth_imgs_warp.append(warp(oth_imgs[i], flows[i]))
            
        # 1. 将参考帧与对齐后的相邻帧直接在通道维度拼接
        aligned_frames = [ref_img] + oth_imgs_warp
        concat_features = torch.cat(aligned_frames, dim=1)
        
        # 2. 通过基于 StarNet 的融合网络进行卷积处理，输出 4 通道特征
        fused_4ch = self.fusion(concat_features)
        
        # 3. 残差学习：加上参考帧（加速收敛，防止颜色偏移）
        fused_4ch = fused_4ch + ref_img
        
        # 4. 通过转置卷积上采样恢复到原图尺寸，并转为 3 通道 RGB
        out_3ch = torch.sigmoid(self.upsample(fused_4ch))
        
        return out_3ch

# ----------------- FLOPs 测试执行模块 -----------------
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = RawNet(num_frames=9).to(device)
    
    # 构建测试张量
    dummy_input = torch.ones(1, 9, 4, 384, 768).to(device)
    flops = FlopCountAnalysis(model, dummy_input)
    print(flop_count_table(flops))

    num_params = sum(p.numel() for p in model.parameters())
    print("\n" + "="*20 +" Model params and FLOPs " + "="*20)
    print(f"\tTotal # of model parameters : {num_params / (1000**2) :.3f} M")
    
    # (1000**4) 在计算上等价于 10^12，即 TFLOPs ；GFLOPs 用的是 (1000**3) 或者 (1024**3)
    # 此处我们换算出 GFLOPs 输出，方便直观比对你的 "30G" 目标
    print(f"\tTotal FLOPs of the model : {flops.total() / (1000**3) :.3f} G (GigaMACs)")
    print(f"\t                         : {flops.total() / (1000**4) :.3f} T (TeraMACs)")
    print("=" * 64)
    print('\n------- Fusion started -------\n')

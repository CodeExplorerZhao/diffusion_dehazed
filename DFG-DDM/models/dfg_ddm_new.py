import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w)  # b,c,hw
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x+h_      


import math
import torch
import torch.nn as nn

# This script is from the following repositories
# https://github.com/ermongroup/ddim
# https://github.com/bahjat-kawar/ddrm


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels,
                                         out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w)  # b,c,hw
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x+h_

class ResBlock_fft_bench(nn.Module):
    def __init__(self, n_feat):
        super(ResBlock_fft_bench, self).__init__()
        self.main = nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1)
        self.mag = nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0)
        # self.mag_h = nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0)
        self.pha = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0),
        )

    def forward(self, x):
        _, _, H, W = x.shape
        fre = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(fre)
        pha = torch.angle(fre)
        # print(mag.max(), mag.min())
        # 设置高频阈值
        threshold = 0.005 * mag.max()
        # 筛选高频信息
        high_freq_mask = mag > threshold
        high_freq_info = mag * high_freq_mask.float()
        # print(high_freq_mask.sum().item(), (mag>0).sum().item())

        mag_out = self.mag(mag)
        mag_res = mag_out - mag + high_freq_info
        # print(mag_res.max())
        pooling = torch.nn.functional.adaptive_avg_pool2d(mag_res, (1, 1))
        pooling = torch.nn.functional.softmax(pooling, dim=1)
        pha1 = pha * pooling
        pha1 = self.pha(pha1)
        pha_out = pha1 + pha
        real = mag_out * torch.cos(pha_out)
        imag = mag_out * torch.sin(pha_out)
        fre_out = torch.complex(real, imag)
        y = torch.fft.irfft2(fre_out, s=(H, W), norm='backward')

        return self.main(x) + y            


class PhysModBlock(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        hidden = max(16, n_feat // 2)
        self.mod = nn.Sequential(
            nn.Conv2d(1, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, n_feat, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, tau=None):

        if tau is None:
            return x
        if tau.dim() == 3:
            tau = tau.unsqueeze(1)  # 插入通道维度，shape变为 (1, 1, 512, 512)
        tau = F.interpolate(tau, size=x.shape[2:], mode='bilinear', align_corners=False)
        haze = 1.0 - tau
        g = self.mod(haze)
        return x + x * g

# class ResBlock_dwt_bench(nn.Module):
#     """
#     Final recommended wavelet enhancement block.

#     Design:
#       1) Spatial branch: local residual enhancement
#       2) LL branch: low-frequency enhancement
#       3) HF branch: high-frequency residual correction
#       4) Adaptive fusion of spatial-domain and wavelet-domain residuals

#     Input:
#       x: [B, C, H, W]

#     Output:
#       y: [B, C, H, W]
#     """
#     def __init__(self, n_feat, hf_hidden=None, use_hf_gate=True, use_fusion_gate=True):
#         super().__init__()

#         # -------- 1) spatial branch --------
#         self.spatial = nn.Sequential(
#             nn.Conv2d(n_feat, n_feat, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(n_feat, n_feat, 3, padding=1)
#         )

#         # -------- 2) LL branch: low-frequency enhancement --------
#         self.ll = nn.Sequential(
#             nn.Conv2d(n_feat, n_feat, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(n_feat, n_feat, 3, padding=1)
#         )

#         # -------- 3) HF branch: high-frequency residual correction --------
#         hf_hidden = hf_hidden or n_feat
#         self.hf = nn.Sequential(
#             nn.Conv2d(n_feat * 3, hf_hidden, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(hf_hidden, n_feat * 3, 3, padding=1)
#         )

#         self.use_hf_gate = use_hf_gate
#         if self.use_hf_gate:
#             gate_hidden = max(16, n_feat // 2)
#             self.hf_gate = nn.Sequential(
#                 nn.Conv2d(n_feat * 3, gate_hidden, 3, padding=1),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(gate_hidden, n_feat * 3, 3, padding=1),
#                 nn.Sigmoid()
#             )

#         # -------- learnable scales --------
#         self.ll_scale = nn.Parameter(torch.tensor(0.1))
#         self.hf_scale = nn.Parameter(torch.tensor(0.1))
#         self.wavelet_scale = nn.Parameter(torch.tensor(0.1))

#         # -------- 4) adaptive fusion --------
#         self.use_fusion_gate = use_fusion_gate
#         fuse_hidden = max(32, n_feat)

#         if self.use_fusion_gate:
#             self.branch_gate = nn.Sequential(
#                 nn.Conv2d(n_feat * 2, 2, 1, padding=0),
#                 nn.Sigmoid()
#             )

#         self.fuse = nn.Sequential(
#             nn.Conv2d(n_feat * 2, fuse_hidden, 1, padding=0),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(fuse_hidden, n_feat, 3, padding=1)
#         )

#     @staticmethod
#     def _pad_even(x):
#         B, C, H, W = x.shape
#         pad_h = H % 2
#         pad_w = W % 2
#         if pad_h or pad_w:
#             x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
#         return x, H, W

#     @staticmethod
#     def dwt_haar(x):
#         x00 = x[:, :, 0::2, 0::2]
#         x01 = x[:, :, 0::2, 1::2]
#         x10 = x[:, :, 1::2, 0::2]
#         x11 = x[:, :, 1::2, 1::2]

#         LL = (x00 + x01 + x10 + x11) * 0.5
#         LH = (x00 - x01 + x10 - x11) * 0.5
#         HL = (x00 + x01 - x10 - x11) * 0.5
#         HH = (x00 - x01 - x10 + x11) * 0.5
#         return LL, LH, HL, HH

#     @staticmethod
#     def iwt_haar(LL, LH, HL, HH):
#         x00 = (LL + LH + HL + HH) * 0.5
#         x01 = (LL - LH + HL - HH) * 0.5
#         x10 = (LL + LH - HL - HH) * 0.5
#         x11 = (LL - LH - HL + HH) * 0.5

#         B, C, h, w = LL.shape
#         out = torch.zeros((B, C, h * 2, w * 2), device=LL.device, dtype=LL.dtype)
#         out[:, :, 0::2, 0::2] = x00
#         out[:, :, 0::2, 1::2] = x01
#         out[:, :, 1::2, 0::2] = x10
#         out[:, :, 1::2, 1::2] = x11
#         return out

#     def forward(self, x):
#         # -------- spatial residual --------
#         spatial_res = self.spatial(x)

#         # -------- wavelet decomposition --------
#         x_pad, H0, W0 = self._pad_even(x)
#         LL, LH, HL, HH = self.dwt_haar(x_pad)

#         # -------- LL branch: low-frequency enhancement --------
#         # Enhancement-style design: directly enhance low-frequency structure
#         LL_enh = self.ll(LL)
#         LL_out = LL + self.ll_scale * LL_enh

#         # -------- HF branch: high-frequency residual correction --------
#         # Residual-style design: preserve original high-frequency basis
#         HF = torch.cat([LH, HL, HH], dim=1)   # [B, 3C, h, w]
#         HF_res = self.hf(HF)

#         if self.use_hf_gate:
#             hf_gate = self.hf_gate(HF)
#             HF_out = HF + self.hf_scale * hf_gate * HF_res
#         else:
#             HF_out = HF + self.hf_scale * HF_res

#         LH_out, HL_out, HH_out = torch.chunk(HF_out, 3, dim=1)

#         # -------- inverse wavelet transform --------
#         wavelet_out = self.iwt_haar(LL_out, LH_out, HL_out, HH_out)
#         wavelet_out = wavelet_out[:, :, :H0, :W0]

#         # wavelet branch acts as a residual compensator
#         wavelet_res = self.wavelet_scale * (wavelet_out - x)

#         # -------- adaptive fusion --------
#         fusion_in = torch.cat([spatial_res, wavelet_res], dim=1)  # [B, 2C, H, W]

#         if self.use_fusion_gate:
#             branch_w = self.branch_gate(fusion_in)                # [B, 2, H, W]
#             spatial_w = branch_w[:, 0:1, :, :]
#             wavelet_w = branch_w[:, 1:2, :, :]
#             fusion_in = torch.cat(
#                 [spatial_w * spatial_res, wavelet_w * wavelet_res], dim=1
#             )

#         fused_res = self.fuse(fusion_in)

#         # -------- final output --------
#         return x + fused_res


# new
# class PhysModBlock(nn.Module):
#     def __init__(self, n_feat):
#         super().__init__()
#         hidden = max(16, n_feat // 2)
#         self.to_gamma = nn.Sequential(
#             nn.Conv2d(1, hidden, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(hidden, n_feat, 3, padding=1),
#             nn.Sigmoid()
#         )
#         self.to_beta = nn.Sequential(
#             nn.Conv2d(1, hidden, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(hidden, n_feat, 3, padding=1)
#         )

#     def forward(self, x, tau=None):
#         if tau is None:
#             return x
#         if tau.dim() == 3:
#             tau = tau.unsqueeze(1)
#         tau = F.interpolate(tau, size=x.shape[2:], mode='bilinear', align_corners=False)
#         haze = 1.0 - tau
#         gamma = self.to_gamma(haze)
#         beta = self.to_beta(haze)
#         return x * (1 + gamma) + torch.tanh(beta)


# class LLBlock(nn.Module):
#     """
#     Lightweight low-frequency enhancement block.
#     """
#     def __init__(self, n_feat):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(n_feat, n_feat, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(n_feat, n_feat, 3, padding=1),
#             ChannelAttention(n_feat)
#         )

#     def forward(self, x):
#         return self.block(x)


# class HFBlock(nn.Module):
#     """
#     High-frequency residual correction block with grouped spatial gating.
#     """
#     def __init__(self, n_feat, hidden_channels=None, use_group_gate=True):
#         super().__init__()
#         in_channels = n_feat * 3
#         hidden_channels = hidden_channels or n_feat

#         self.body = nn.Sequential(
#             nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(hidden_channels, in_channels, 3, padding=1)
#         )

#         self.use_group_gate = use_group_gate
#         if self.use_group_gate:
#             self.gate = GroupedSpatialAttention(kernel_size=7)

#     def forward(self, lh, hl, hh, scale):
#         hf = torch.cat([lh, hl, hh], dim=1)       # [B,3C,H,W]
#         hf_res = self.body(hf)                    # [B,3C,H,W]

#         lh_res, hl_res, hh_res = torch.chunk(hf_res, 3, dim=1)

#         if self.use_group_gate:
#             g = self.gate(hf)                     # [B,3,H,W]
#             g_lh = g[:, 0:1, :, :]
#             g_hl = g[:, 1:2, :, :]
#             g_hh = g[:, 2:3, :, :]

#             lh_out = lh + scale * g_lh * lh_res
#             hl_out = hl + scale * g_hl * hl_res
#             hh_out = hh + scale * g_hh * hh_res
#         else:
#             lh_out = lh + scale * lh_res
#             hl_out = hl + scale * hl_res
#             hh_out = hh + scale * hh_res

#         return lh_out, hl_out, hh_out


# class LLBlock(nn.Module):
#     """
#     带上下文感知的低频增强（替代原有的全局池化通道注意力）
#     """
#     def __init__(self, n_feat):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(n_feat, n_feat, 3, padding=1),
#             nn.ReLU(inplace=True),
#             # 使用 dilation=2 或 3 的空洞卷积，成倍扩大感受野，
#             # 让网络能感知更广的全局雾气分布，同时丝毫不损失空间分辨率！
#             nn.Conv2d(n_feat, n_feat, 3, padding=2, dilation=2), 
#             nn.ReLU(inplace=True),
#             nn.Conv2d(n_feat, n_feat, 3, padding=1)
#         )

#     def forward(self, x):
#         return self.block(x)

# new version with simplified design and better performance
# class ResBlock_dwt_bench(nn.Module):
#     """
#     Differentiated Wavelet Enhancement and Dual-Domain Fusion Block

#     Design:
#       1) Spatial branch: local residual enhancement
#       2) LL branch: lightweight low-frequency enhancement
#       3) HF branch: grouped high-frequency residual correction
#       4) Spatial-frequency fusion via convolutional fusion head
#     """
#     def __init__(self, n_feat, hf_hidden=None, use_hf_gate=True):
#         super().__init__()

#         # 1) spatial branch
#         self.spatial = nn.Sequential(
#             nn.Conv2d(n_feat, n_feat, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(n_feat, n_feat, 3, padding=1)
#         )

#         # 2) LL branch
#         self.ll = LLBlock(n_feat)

#         # 3) HF branch
#         self.hf = HFBlock(
#             n_feat=n_feat,
#             hidden_channels=hf_hidden or n_feat,
#             use_group_gate=use_hf_gate
#         )

#         # learnable scales
#         self.ll_scale = nn.Parameter(torch.tensor(0.1))
#         self.hf_scale = nn.Parameter(torch.tensor(0.1))
#         self.wavelet_scale = nn.Parameter(torch.tensor(0.1))

#         # 4) dual-domain fusion
#         fuse_hidden = max(32, n_feat)
#         self.fuse = nn.Sequential(
#             nn.Conv2d(n_feat * 2, fuse_hidden, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(fuse_hidden, n_feat, 3, padding=1)
#         )

#     @staticmethod
#     def _pad_even(x):
#         B, C, H, W = x.shape
#         pad_h = H % 2
#         pad_w = W % 2
#         if pad_h or pad_w:
#             x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
#         return x, H, W

#     @staticmethod
#     def dwt_haar(x):
#         x00 = x[:, :, 0::2, 0::2]
#         x01 = x[:, :, 0::2, 1::2]
#         x10 = x[:, :, 1::2, 0::2]
#         x11 = x[:, :, 1::2, 1::2]

#         ll = (x00 + x01 + x10 + x11) * 0.5
#         lh = (x00 - x01 + x10 - x11) * 0.5
#         hl = (x00 + x01 - x10 - x11) * 0.5
#         hh = (x00 - x01 - x10 + x11) * 0.5
#         return ll, lh, hl, hh

#     @staticmethod
#     def iwt_haar(ll, lh, hl, hh):
#         x00 = (ll + lh + hl + hh) * 0.5
#         x01 = (ll - lh + hl - hh) * 0.5
#         x10 = (ll + lh - hl - hh) * 0.5
#         x11 = (ll - lh - hl + hh) * 0.5

#         B, C, h, w = ll.shape
#         out = torch.zeros((B, C, h * 2, w * 2), device=ll.device, dtype=ll.dtype)
#         out[:, :, 0::2, 0::2] = x00
#         out[:, :, 0::2, 1::2] = x01
#         out[:, :, 1::2, 0::2] = x10
#         out[:, :, 1::2, 1::2] = x11
#         return out

#     def forward(self, x):
#         # spatial branch
#         spatial_res = self.spatial(x)

#         # wavelet decomposition
#         x_pad, H0, W0 = self._pad_even(x)
#         ll, lh, hl, hh = self.dwt_haar(x_pad)

#         # low-frequency enhancement
#         ll_enh = self.ll(ll)
#         ll_out = ll + self.ll_scale * ll_enh

#         # high-frequency residual correction
#         lh_out, hl_out, hh_out = self.hf(lh, hl, hh, scale=self.hf_scale)

#         # inverse wavelet transform
#         wavelet_out = self.iwt_haar(ll_out, lh_out, hl_out, hh_out)
#         wavelet_out = wavelet_out[:, :, :H0, :W0]

#         # wavelet residual compensation
#         wavelet_res = self.wavelet_scale * (wavelet_out - x)

#         # dual-domain fusion
#         fusion_in = torch.cat([spatial_res, wavelet_res], dim=1)
#         fused_res = self.fuse(fusion_in)

#         # final output
#         return x + fused_res   
class LLBlock(nn.Module):
    """
    低频增强块：先用稳健版，不先上 dilation
    """
    def __init__(self, n_feat):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat, n_feat, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat, n_feat, 3, padding=1)
        )

    def forward(self, x):
        # 预测低频残差
        return self.block(x)

class DirectionalGating(nn.Module):
    """
    修正后的高频方向感知门控：
    1. 不再混合 LH/HL/HH 生成权重，而是独立建模。
    2. 使用条形卷积 (Strip Conv) 匹配物理梯度方向。
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        # 为每个方向定制各向异性卷积核
        self.gate_lh = nn.Conv2d(2, 1, (1, kernel_size), padding=(0, kernel_size // 2))
        self.gate_hl = nn.Conv2d(2, 1, (kernel_size, 1), padding=(kernel_size // 2, 0))
        self.gate_hh = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, lh, hl, hh):
        def compute_mask(x, conv_op):
            # 基于子带自身统计量生成空间掩码，拒绝过早混合
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            return self.sigmoid(conv_op(torch.cat([avg_out, max_out], dim=1)))
        
        return compute_mask(lh, self.gate_lh), compute_mask(hl, self.gate_hl), compute_mask(hh, self.gate_hh)

class HFBlock(nn.Module):
    """
    高频残差校正块 (修正版)：
    1. 引入 DWConv 独立处理各子带，保持方向解耦。
    2. 引入可学习的 Bounded Scale，防止特征偏移。
    """
    def __init__(self, n_feat):
        super().__init__()
        # 各向异性处理分支
        self.lh_proc = nn.Conv2d(n_feat, n_feat, (1, 3), padding=(0, 1), groups=n_feat)
        self.hl_proc = nn.Conv2d(n_feat, n_feat, (3, 1), padding=(1, 0), groups=n_feat)
        self.hh_proc = nn.Conv2d(n_feat, n_feat, 3, padding=1, groups=n_feat)
        
        self.gating = DirectionalGating()
        self.learnable_alpha = nn.Parameter(torch.ones(1, 3, 1, 1) * 0.1)

    def forward(self, lh, hl, hh, scale):
        # 1. 独立方向特征提取
        lh_feat, hl_feat, hh_feat = self.lh_proc(lh), self.hl_proc(hl), self.hh_proc(hh)
        # 2. 获取方向感知的空间掩码
        m_lh, m_hl, m_hh = self.gating(lh, hl, hh)
        # 3. 物理约束下的残差补偿
        lh_out = lh + scale * self.learnable_alpha[:, 0:1] * m_lh * lh_feat
        hl_out = hl + scale * self.learnable_alpha[:, 1:2] * m_hl * hl_feat
        hh_out = hh + scale * self.learnable_alpha[:, 2:3] * m_hh * hh_feat
        return lh_out, hl_out, hh_out
    
class ResBlock_dwt_bench(nn.Module):
    """
    推荐版频域增强块：
    1) spatial residual
    2) LL residual enhancement
    3) HF directional correction
    4) spatial-frequency fusion
    """
    def __init__(self, n_feat):
        super().__init__()

        # spatial branch
        self.spatial = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat, n_feat, 3, padding=1)
        )

        # LL / HF
        self.ll = LLBlock(n_feat)
        self.hf = HFBlock(n_feat)

        # scales
        self.ll_scale = nn.Parameter(torch.tensor(0.1))
        self.hf_scale = nn.Parameter(torch.tensor(0.1))
        self.wavelet_scale = nn.Parameter(torch.tensor(0.1))

        # 可选：融合前做简单双分支门控
        self.fuse_gate = nn.Sequential(
            nn.Conv2d(n_feat * 2, 2, 1),
            nn.Sigmoid()
        )

        fuse_hidden = max(32, n_feat)
        self.fuse = nn.Sequential(
            nn.Conv2d(n_feat * 2, fuse_hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fuse_hidden, n_feat, 3, padding=1)
        )

    @staticmethod
    def _pad_even(x):
        B, C, H, W = x.shape
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, H, W

    @staticmethod
    def dwt_haar(x):
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (x00 - x01 + x10 - x11) * 0.5
        hl = (x00 + x01 - x10 - x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5
        return ll, lh, hl, hh

    @staticmethod
    def iwt_haar(ll, lh, hl, hh):
        x00 = (ll + lh + hl + hh) * 0.5
        x01 = (ll - lh + hl - hh) * 0.5
        x10 = (ll + lh - hl - hh) * 0.5
        x11 = (ll - lh - hl + hh) * 0.5

        B, C, h, w = ll.shape
        out = torch.zeros((B, C, h * 2, w * 2), device=ll.device, dtype=ll.dtype)
        out[:, :, 0::2, 0::2] = x00
        out[:, :, 0::2, 1::2] = x01
        out[:, :, 1::2, 0::2] = x10
        out[:, :, 1::2, 1::2] = x11
        return out

    def forward(self, x):
        # spatial branch
        spatial_res = self.spatial(x)

        # wavelet
        x_pad, H0, W0 = self._pad_even(x)
        ll, lh, hl, hh = self.dwt_haar(x_pad)

        # LL residual enhancement
        ll_res = self.ll(ll)
        ll_out = ll + self.ll_scale * ll_res

        # HF correction
        lh_out, hl_out, hh_out = self.hf(lh, hl, hh, scale=self.hf_scale)

        # inverse wavelet
        wavelet_out = self.iwt_haar(ll_out, lh_out, hl_out, hh_out)
        wavelet_out = wavelet_out[:, :, :H0, :W0]

        # wavelet residual
        wavelet_res = self.wavelet_scale * (wavelet_out - x)

        # gated fusion
        fusion_in = torch.cat([spatial_res, wavelet_res], dim=1)
        g = self.fuse_gate(fusion_in)
        spatial_res = spatial_res * g[:, 0:1]
        wavelet_res = wavelet_res * g[:, 1:2]

        fused_res = self.fuse(torch.cat([spatial_res, wavelet_res], dim=1))

        return x + fused_res

   

class DFG_UNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_phys = getattr(config.model, "use_phys", False)
        self.use_freq = getattr(config.model, "use_freq", False)
        self.use_freq_out = getattr(config.model, "use_freq_out", False)
        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        in_channels = config.model.in_channels * 2 if config.data.conditional else config.model.in_channels
        resolution = config.data.image_size
        resamp_with_conv = config.model.resamp_with_conv

        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch,
                            self.temb_ch),
            torch.nn.Linear(self.temb_ch,
                            self.temb_ch),
        ])

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        self.conv0 = torch.nn.Conv2d(256, self.ch, kernel_size=3, stride=1, padding=1)
        self.conv1 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(256, self.ch, kernel_size=3, stride=1, padding=1)
        curr_res = resolution
        in_ch_mult = (1,)+ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        if self.use_phys:
            self.phys_in = PhysModBlock(self.ch)
            self.tau_head = nn.Sequential(
                nn.Conv2d(block_in, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, kernel_size=1),
                nn.Sigmoid()
            )

        if self.use_freq:
            self.freq_in = ResBlock_dwt_bench(n_feat=self.ch)

        if self.use_freq and self.use_freq_out:
            self.freq_out = ResBlock_dwt_bench(n_feat=block_in)

    def forward(self, x, t, tau_hint=None):
        assert x.shape[2] == x.shape[3] == self.resolution

        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        tt = self.conv_in(x)

        if self.use_phys:
            tt = self.phys_in(tt, tau_hint)

        if self.use_freq:
            tt = self.freq_in(tt)

        hs = [tt]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)  # 512,8

        # swin2=self.E_block2(h)#32    
        # swin2=self.PPM2(swin2)   # (1,64,256,256) 256,8
        # swin2 = self.conv1(swin2)
        # h = self.fft2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        if self.use_freq and self.use_freq_out:
            h = self.freq_out(h)
        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        eps_hat = self.conv_out(h)

        if not self.use_phys:
            return eps_hat

        t_hat = self.tau_head(h)

        return eps_hat, t_hat

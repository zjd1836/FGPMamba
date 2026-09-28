import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F
from benchmarks.MambaCD.changedetection.models.Mamba_backbone import Backbone_VSSM



def check_nan(tensor, name):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"Fatal Error: NaN/Inf detected in [{name}]")
        print(f"Shape: {tensor.shape}")
        print(f"Min: {tensor.min()}, Max: {tensor.max()}")
        raise ValueError(f"NaN detected in {name}")
    return tensor


class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super(DSConv, self).__init__()
        padding = (kernel_size - 1) // 2
        self.depth_conv = nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding, groups=in_ch)
        self.point_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class FrequencyEnhanceBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.amp_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 1)
        )

    def forward(self, x):
        x = torch.clamp(x, min=-20, max=20)
        B, C, H, W = x.shape
        fft_x = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(fft_x)
        pha = torch.angle(fft_x)
        mag_enhanced = self.amp_conv(mag) + mag
        real = mag_enhanced * torch.cos(pha)
        imag = mag_enhanced * torch.sin(pha)
        fft_enhanced = torch.complex(real, imag)
        out = torch.fft.irfft2(fft_enhanced, s=(H, W), norm='backward')

        return x + out

class StructurePrompter(nn.Module):
    def __init__(self):
        super().__init__()
        dims = [96, 192, 384, 768]

        self.input_norm = nn.BatchNorm2d(3)

        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, dims[0], 3, stride=2, padding=1),
            nn.BatchNorm2d(dims[0]), nn.ReLU(inplace=True),
            FrequencyEnhanceBlock(dims[0])
        )
        self.stage2 = self._make_stage(dims[0], dims[1])
        self.stage3 = self._make_stage(dims[1], dims[2])
        self.stage4 = self._make_stage(dims[2], dims[3])

    def _make_stage(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            DSConv(out_c, out_c),
            FrequencyEnhanceBlock(out_c)
        )

    def forward(self, x):
        x = self.input_norm(x)
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        return [f1, f2, f3, f4]


class CrossModalGatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_gate = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1), nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim, 1), nn.Sigmoid()
        )

        self.norm = nn.GroupNorm(8, dim)

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x.repeat(dim, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.repeat(dim, 1, 1, 1))

        self.slope_gate = nn.Sequential(
            nn.Conv2d(dim, 1, 7, padding=3), nn.Sigmoid()
        )
        self.smooth = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

    def forward(self, global_feat, local_feat):
        chan_w = self.channel_gate(self.global_pool(global_feat))

        local_feat_norm = self.norm(local_feat)
        gx = F.conv2d(local_feat_norm, self.sobel_x, padding=1, groups=local_feat.shape[1])
        gy = F.conv2d(local_feat_norm, self.sobel_y, padding=1, groups=local_feat.shape[1])
        slope_sq = gx ** 2 + gy ** 2 + 1e-6
        slope = torch.sqrt(slope_sq)
        spatial_w = self.slope_gate(slope)

        return global_feat + self.smooth(local_feat * chan_w * spatial_w)


class SynergisticMambaEncoder(nn.Module):
    def __init__(self, pretrained=None):
        super().__init__()
        default_pretrained = 'vssm_tiny.pth'
        pt_path = pretrained if pretrained is not None else default_pretrained

        model_params = {
            "patch_size": 4, "in_chans": 3, "num_classes": 1000, "depths": [2, 2, 4, 2],
            "dims": 96, "ssm_d_state": 1, "ssm_ratio": 2.0, "ssm_rank_ratio": 2.0,
            "ssm_dt_rank": "auto", "ssm_act_layer": "silu", "ssm_conv": 3, "ssm_conv_bias": False,
            "ssm_drop_rate": 0.0, "ssm_init": "v0", "forward_type": "v3noz", "mlp_ratio": 4.0,
            "mlp_act_layer": "gelu", "mlp_drop_rate": 0.0, "drop_path_rate": 0.3, "patch_norm": True,
            "norm_layer": "ln", "downsample_version": "v3", "patchembed_version": "v2",
            "gmlp": False, "use_checkpoint": False,
        }
        self.global_stream = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pt_path, **model_params)
        self.local_stream = StructurePrompter()
        dims = [96, 192, 384, 768]
        self.fusion_modules = nn.ModuleList([CrossModalGatedFusion(dim) for dim in dims])

    def forward(self, t1, t2):
        g1 = self.global_stream(t1)
        l1 = self.local_stream(t1)
        g2 = self.global_stream(t2)
        l2 = self.local_stream(t2)

        f1, f2 = [], []
        for i in range(4):
            f1.append(self.fusion_modules[i](g1[i], l1[i]))
            f2.append(self.fusion_modules[i](g2[i], l2[i]))
        return f1, f2


class GradientRefinementBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(DSConv(in_channels, in_channels), nn.Conv2d(in_channels, in_channels, 1))
        self.norm = nn.GroupNorm(8, in_channels)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x.repeat(in_channels, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.repeat(in_channels, 1, 1, 1))
        self.gradient_gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1), nn.ReLU(),
            nn.Conv2d(in_channels // 2, 1, 1), nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.conv(x)

        x_norm = self.norm(x)
        gx = F.conv2d(x_norm, self.sobel_x, padding=1, groups=x.shape[1])
        gy = F.conv2d(x_norm, self.sobel_y, padding=1, groups=x.shape[1])
        grad_mag_sq = gx ** 2 + gy ** 2 + 1e-6
        grad_mag = torch.sqrt(grad_mag_sq)

        gate = self.gradient_gate(grad_mag)
        return x + feat * gate


class DifferenceEnhancedModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.sa = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.BatchNorm2d(in_channels // 4), nn.ReLU(),
            nn.Conv2d(in_channels // 4, 1, 7, padding=3), nn.Sigmoid()
        )
        self.fusion = DSConv(out_channels * 3, out_channels)

    def forward(self, t1, t2):
        raw_diff = torch.abs(t1 - t2)
        mask = self.sa(raw_diff)
        t1_w = self.proj(t1) * mask
        t2_w = self.proj(t2) * mask
        diff_proj = self.proj(raw_diff)
        return self.fusion(torch.cat([t1_w, t2_w, diff_proj], dim=1))


class GradientGuidedCascadedDecoder(nn.Module):
    def __init__(self, encoder_dims=[96, 192, 384, 768], base_dim=128):
        super().__init__()
        self.diff_stages = nn.ModuleList([DifferenceEnhancedModule(dim, base_dim) for dim in encoder_dims])
        self.fusion_4_3 = nn.Sequential(DSConv(base_dim * 2, base_dim), GradientRefinementBlock(base_dim))
        self.fusion_3_2 = nn.Sequential(DSConv(base_dim * 2, base_dim), GradientRefinementBlock(base_dim))
        self.fusion_2_1 = nn.Sequential(DSConv(base_dim * 2, base_dim), GradientRefinementBlock(base_dim))
        self.main_head = nn.Conv2d(base_dim, 1, 1)

    def forward(self, t1_feats, t2_feats):
        diffs = [stage(t1, t2) for stage, t1, t2 in zip(self.diff_stages, t1_feats, t2_feats)]

        f4_up = F.interpolate(diffs[3], scale_factor=2, mode='bilinear', align_corners=False)
        f3 = self.fusion_4_3(torch.cat([f4_up, diffs[2]], dim=1))

        f3_up = F.interpolate(f3, scale_factor=2, mode='bilinear', align_corners=False)
        f2 = self.fusion_3_2(torch.cat([f3_up, diffs[1]], dim=1))

        f2_up = F.interpolate(f2, scale_factor=2, mode='bilinear', align_corners=False)
        f1 = self.fusion_2_1(torch.cat([f2_up, diffs[0]], dim=1))

        return self.main_head(f1)


class FGPMmaba(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SynergisticMambaEncoder()
        self.decoder = GradientGuidedCascadedDecoder(base_dim=128)

    def forward(self, t1, t2):
        if torch.isnan(t1).any() or torch.isnan(t2).any():
            print("Input Images contain NaN!")

        t1_feats, t2_feats = self.encoder(t1, t2)
        change_logit = self.decoder(t1_feats, t2_feats)

        return F.interpolate(change_logit, size=t1.shape[2:], mode='bilinear', align_corners=False)

if __name__ == '__main__':

    model = FGPMmaba().cuda()

    x1 = torch.randn(1, 3, 256, 256).cuda()
    x2 = torch.randn(1, 3, 256, 256).cuda()
    out = model(x1, x2)
    print(f"Output shape: {out.shape}")

"""Model definitions from FDR26_benchmark.ipynb, Zenodo 20605487.
Extracted without functional changes using scripts/vendor_reference.py.
Upstream README declares MIT for code; see reference/PROVENANCE.md.
Only AttentionClassifier(convnext_tiny) is used; RETFound is not supported here.
"""
import torch
from torch import nn
import timm

NUM_CLASSES = 5
IMG_SIZE = 384
DROPOUT = 0.5

class SEBlock(nn.Module):

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channels, channels // reduction, bias=False), nn.ReLU(), nn.Linear(channels // reduction, channels, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        return x * self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)

class ChannelAttention(nn.Module):

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channels, channels // reduction, bias=False), nn.ReLU(), nn.Linear(channels // reduction, channels, bias=False))
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        a = self.fc(self.avg_pool(x).view(b, c))
        m = self.fc(self.max_pool(x).view(b, c))
        return x * self.sig(a + m).view(b, c, 1, 1)

class SpatialAttention(nn.Module):

    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sig(self.conv(torch.cat([avg, mx], dim=1)))

class CBAM(nn.Module):

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))

class AttentionClassifier(nn.Module):
    """Backbone timm + CAB (SE puis CBAM) pour les CNN + tête MLP commune."""

    def __init__(self, backbone_name, num_classes=NUM_CLASSES, pretrained=True, img_size=IMG_SIZE, dropout=DROPOUT):
        super().__init__()
        kw = dict(pretrained=pretrained, num_classes=0, global_pool='')
        if backbone_name.startswith('RETFOUND'):
            backbone_name_timm, pretrained_bb = ('vit_large_patch16_224', False)
        else:
            backbone_name_timm, pretrained_bb = (backbone_name, pretrained)
        kw['pretrained'] = pretrained_bb
        self.backbone = None
        for essai in (dict(img_size=img_size), dict(dynamic_img_size=True), dict()):
            try:
                self.backbone = timm.create_model(backbone_name_timm, **essai, **kw)
                break
            except (TypeError, RuntimeError, AssertionError, KeyError):
                continue
        if self.backbone is None:
            raise RuntimeError(f'Backbone {backbone_name} impossible a creer @ {img_size}px')
        self.est_retfound = backbone_name.startswith('RETFOUND')
        if self.est_retfound and pretrained:
            if not charger_poids_retfound(self.backbone, img_size):
                raise RuntimeError('RETFound indisponible (poids non telecharges)')
        nf = self.backbone.num_features
        self.is_conv = self._detect_conv(img_size)
        if self.is_conv:
            self.se = SEBlock(nf)
            self.cbam = CBAM(nf)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(nf, 512), nn.ReLU(), nn.Dropout(dropout * 0.6), nn.Linear(512, num_classes))

    def _detect_conv(self, res):
        self.backbone.eval()
        with torch.no_grad():
            try:
                out = self.backbone.forward_features(torch.zeros(1, 3, res, res))
            except Exception:
                out = self.backbone.forward_features(torch.zeros(1, 3, 224, 224))
        nf = self.backbone.num_features
        return out.ndim == 4 and out.shape[1] == nf and (out.shape[-1] != nf)

    def _to_vector(self, x):
        nf = self.backbone.num_features
        if x.ndim == 4:
            if x.shape[1] == nf and x.shape[-1] != nf:
                return x.mean(dim=(2, 3))
            return x.mean(dim=(1, 2))
        if x.ndim == 3:
            return x.mean(dim=1)
        return x

    def forward(self, x):
        x = self.backbone.forward_features(x)
        if self.is_conv:
            x = self.cbam(self.se(x))
        return self.head(self.drop(self._to_vector(x)))

"""The trainable part of smp_model.youtube_lstm3, fed by cached features.

Layer names and shapes match youtube_lstm3 so its weights load directly (see ext.parity_check).
Differences from the original, all deliberate:
  * frozen ResNet/BERT are gone; their (eval-mode) outputs come from the feature cache
  * any input branch can be switched off, and the image branch accepts 2048/512/2560-d input
  * batch size is taken from the input, so single-video prediction works (needed for the demo)
"""
import torch
from torch import nn

from ext.common import N_CATE, N_LANG

V = 128  # vector_len in the original


def mlp(*dims):
    """Linear -> BatchNorm(affine=False) -> ReLU blocks, last Linear bare (the original's MLP pattern)."""
    layers = []
    for i in range(len(dims) - 2):
        layers += [nn.Linear(dims[i], dims[i + 1]), nn.BatchNorm1d(dims[i + 1], affine=False), nn.ReLU()]
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


class PopularityHead(nn.Module):
    def __init__(self, img_dim=2048, text=True, catlang=True, meta_dim=5, n_targets=30):
        super().__init__()
        self.img_dim, self.use_text, self.use_catlang, self.meta_dim = img_dim, text, catlang, meta_dim
        self.n_targets = n_targets
        branches = 0
        if img_dim:
            self.img_MLP = mlp(img_dim, 1024, 512, V); branches += 1
        if catlang:
            self.cate_embedding = nn.Sequential(nn.Embedding(N_CATE, V))
            self.lang_embedding = nn.Sequential(nn.Embedding(N_LANG, V))
            self.cate_MLP, self.lang_MLP = mlp(V, 256, V), mlp(V, 256, V)
            self.emb_MLP = mlp(V, 256, V); branches += 1
        if text:
            self.conv = nn.Conv2d(5, 1, 1)
            self.conv.weight.data.normal_(1 / 5, 0.01)
            self.text_MLP = mlp(768, 512, V); branches += 1
        if meta_dim:
            self.meta_MLP = mlp(meta_dim, V, V); branches += 1
        if branches == 0:
            raise ValueError("at least one input branch is required")
        f = branches * V
        self.hc_MLP = mlp(f, 512, V)
        self.x_MLP_list = nn.ModuleList([mlp(f, 768, V) for _ in range(n_targets)])
        self.LSTMCell_list = nn.ModuleList([nn.LSTMCell(V, V) for _ in range(n_targets)])
        self.MLPo_list = nn.ModuleList([mlp(2 * V, V, V // 2, 1) for _ in range(n_targets)])

    def fuse(self, img=None, text=None, cate=None, lang=None, meta=None):
        parts = []
        if self.img_dim:
            parts.append(self.img_MLP(img))
        if self.use_catlang:
            c = self.cate_MLP(self.cate_embedding(cate))
            l = self.lang_MLP(self.lang_embedding(lang))
            parts.append(self.emb_MLP(c * l))
        if self.use_text:
            t = self.conv(text.unsqueeze(3)).flatten(1)  # [B,5,768,1] -> [B,768]
            parts.append(self.text_MLP(t))
        if self.meta_dim:
            parts.append(self.meta_MLP(meta))
        return torch.cat(parts, 1)

    def forward(self, img=None, text=None, cate=None, lang=None, meta=None):
        fv = self.fuse(img, text, cate, lang, meta)
        h = self.hc_MLP(fv)
        c = self.hc_MLP(fv)  # the original initialises h and c with the same MLP
        out = []
        for i in range(self.n_targets):
            h, c = self.LSTMCell_list[i](self.x_MLP_list[i](fv), (h, c))
            out.append(self.MLPo_list[i](torch.cat([h, c], 1)))
        return torch.cat(out, 1)


class CustomLoss(nn.Module):
    """main.CustomLoss without the dead code.

    Removed: the l2_reg loop (computed every step over all 244M parameters, never added to the loss).
    Kept as-is: the argmax peak term (it adds to the loss value but passes no gradient), so loss
    values stay comparable with the original.
    """

    def __init__(self, lambda1=1.0, lambda2=1.0, peak_weight=1.0, epsilon=1e-6):
        super().__init__()
        self.base = nn.SmoothL1Loss(beta=0.1)
        self.l1, self.l2, self.pw, self.eps = lambda1, lambda2, peak_weight, epsilon

    def forward(self, out, y, step, total_steps):
        w = 0.5 * (1 + torch.cos(torch.tensor(step / total_steps * torch.pi)))
        base = self.base(out, y)
        n = out.size(1)
        peak = nn.functional.l1_loss(nn.functional.one_hot(out.argmax(1), n).float(),
                                     nn.functional.one_hot(y.argmax(1), n).float())
        d1, td1 = out[:, 1:] - out[:, :-1], y[:, 1:] - y[:, :-1]
        d2, td2 = d1[:, 1:] - d1[:, :-1], td1[:, 1:] - td1[:, :-1]
        smooth = self.eps * (d1.abs().sum() + d2.abs().sum())
        return base + w * self.pw * peak + w * self.l1 * self.base(d1, td1) + w * self.l2 * self.base(d2, td2) + smooth

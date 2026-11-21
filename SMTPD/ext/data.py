"""Cached-feature dataset, reproducible splits, and a fast batch iterator."""
import os

import numpy as np
import torch

IMAGE_DIMS = {"resnet": 2048, "clip": 512, "both": 2560, "none": 0}


class FeatureStore:
    """Serves model inputs straight from the arrays written by ext.extract_features.

    image:   'resnet' | 'clip' | 'both' | 'none'
    ep_days: k early-popularity days given as input (Day 1..k); targets are Day k+1..30.
             k=0 predicts all 30 days (the paper's no-EP setting); k=1 is the paper's EP setting.
    """

    def __init__(self, feat_dir, image="resnet", text=True, catlang=True, numeric=True,
                 clip_sim=False, ep_days=0, in_memory=False):
        self.feat_dir = feat_dir
        tab = np.load(os.path.join(feat_dir, "tabular.npz"))
        self.ids, self.days, self.user = tab["ids"], tab["days"], tab["user"]
        self.cate, self.lang = tab["cate"].astype(np.int64), tab["lang"].astype(np.int64)
        self.numeric_arr = tab["numeric"]
        self.n = len(self.ids)
        self.has_img = self._load("has_img")
        self.image, self.text, self.catlang, self.numeric = image, text, catlang, numeric
        self.clip_sim, self.ep_days = clip_sim, ep_days
        load = (lambda name: self._load(name, in_memory))
        self.resnet = load("resnet") if image in ("resnet", "both") else None
        self.clip_img = load("clip_img") if image in ("clip", "both") else None
        self.bert = load("bert") if text else None
        self.sim = self._load("clip_sim") if clip_sim else None

    def _load(self, name, in_memory=True):
        path = os.path.join(self.feat_dir, name + ".npy")
        if not os.path.exists(path):
            raise SystemExit(f"Missing {path}: run ext.extract_features first.")
        return np.load(path, mmap_mode=None if in_memory else "r")

    # --- shapes the model needs ---------------------------------------------------------
    @property
    def img_dim(self):
        return IMAGE_DIMS[self.image]

    @property
    def meta_dim(self):
        return (5 if self.numeric else 0) + self.ep_days + (1 if self.clip_sim else 0)

    @property
    def n_targets(self):
        return 30 - self.ep_days

    def describe(self):
        return dict(image=self.image, text=self.text, catlang=self.catlang, numeric=self.numeric,
                    clip_sim=self.clip_sim, ep_days=self.ep_days, img_dim=self.img_dim,
                    meta_dim=self.meta_dim, n_targets=self.n_targets)

    # --- batches -------------------------------------------------------------------------
    def get(self, idx, device):
        """idx must be sorted (faster memmap reads)."""
        f = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(device)
        out = {"img": None, "text": None, "cate": None, "lang": None, "meta": None}
        if self.image == "resnet":
            out["img"] = f(self.resnet[idx])
        elif self.image == "clip":
            out["img"] = f(self.clip_img[idx])
        elif self.image == "both":
            out["img"] = f(np.concatenate([self.resnet[idx], self.clip_img[idx]], 1))
        if self.text:
            out["text"] = f(self.bert[idx])
        if self.catlang:
            out["cate"] = torch.from_numpy(self.cate[idx]).to(device)
            out["lang"] = torch.from_numpy(self.lang[idx]).to(device)
        meta = []
        if self.numeric:
            meta.append(self.numeric_arr[idx])
        if self.ep_days:
            meta.append(self.days[idx, :self.ep_days])
        if self.clip_sim:
            meta.append(self.sim[idx, None])
        if meta:
            out["meta"] = f(np.concatenate(meta, 1))
        y = f(self.days[idx, self.ep_days:])
        return out, y

    def batches(self, idx, batch_size, device, shuffle=False, rng=None, drop_last=False):
        order = rng.permutation(idx) if shuffle else idx
        for s in range(0, len(order), batch_size):
            b = order[s:s + batch_size]
            if (drop_last and len(b) < batch_size) or len(b) < 2:  # BatchNorm needs >1 sample
                continue
            b = np.sort(b)
            x, y = self.get(b, device)
            yield b, x, y


def make_split(feat_dir, kind="random", seed=23, fractions=(0.7, 0.1, 0.2)):
    """Train/val/test indices, cached in <feat_dir>/splits so every run and baseline uses the same split.

    random:  videos shuffled independently (the original code's setting, but with a separate val set).
    channel: all videos of a channel land in the same part, so test channels are unseen in training.
    """
    path = os.path.join(feat_dir, "splits", f"{kind}_seed{seed}.npz")
    if os.path.exists(path):
        s = np.load(path)
        return {k: s[k] for k in ("train", "val", "test")}
    tab = np.load(os.path.join(feat_dir, "tabular.npz"))
    n = len(tab["ids"])
    rng = np.random.default_rng(seed)
    if kind == "random":
        perm = rng.permutation(n)
        a, b = int(fractions[0] * n), int((fractions[0] + fractions[1]) * n)
        parts = perm[:a], perm[a:b], perm[b:]
    elif kind == "channel":
        user = tab["user"]
        channels = rng.permutation(user.max() + 1)
        counts = np.bincount(user, minlength=user.max() + 1)[channels]
        cum = np.cumsum(counts) / n
        tr_ch = channels[cum <= fractions[0]]
        va_ch = channels[(cum > fractions[0]) & (cum <= fractions[0] + fractions[1])]
        part = np.full(user.max() + 1, 2)
        part[tr_ch], part[va_ch] = 0, 1
        p = part[user]
        parts = np.where(p == 0)[0], np.where(p == 1)[0], np.where(p == 2)[0]
    else:
        raise ValueError(kind)
    split = {k: np.sort(v) for k, v in zip(("train", "val", "test"), parts)}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **split)
    return split

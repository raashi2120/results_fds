"""Checks that ext.model.PopularityHead + cached features == the original youtube_lstm3.

Both models get the same weights, the same rows and eval mode; the original runs ResNet/BERT live
(on ImageNet-normalised images, as the cache does), the head reads the cache. Differences should be
at float16 rounding level only.

  python -m ext.parity_check --features ../features_10k
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

from ext.common import DEFAULT_CSV, DEFAULT_FEATURES, DEFAULT_IMAGES, TEXT_COLS
from ext.data import FeatureStore
from ext.extract_features import CoverImages
from ext.model import PopularityHead

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import smp_model  # noqa: E402  (original code)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default=DEFAULT_FEATURES)
    p.add_argument("--n", type=int, default=32)
    args = p.parse_args()
    torch.manual_seed(0)
    dev = torch.device("cpu")
    smp_model.device = dev
    store = FeatureStore(args.features, image="resnet", ep_days=0)
    df = pd.read_csv(DEFAULT_CSV, nrows=store.n)
    # rows with a real cover, and a category the original vocab knows
    rows = np.where(store.has_img & (df["category"] != "Travel  Events").values)[0][:args.n]
    B = len(rows)

    orig = smp_model.youtube_lstm3(29, B).eval()  # meta_num = 6: 5 numeric + zero EP column
    head = PopularityHead(img_dim=2048, text=True, catlang=True, meta_dim=6, n_targets=29).eval()
    missing, unexpected = head.load_state_dict(orig.state_dict(), strict=False)
    assert not missing, missing
    assert all(k.startswith(("img_feature", "bert_model", "proj_", "dropout")) for k in unexpected), unexpected

    imgs = CoverImages(store.ids[rows].tolist(), DEFAULT_IMAGES)
    img = torch.stack([imgs[i][0] for i in range(B)])
    sub = df.iloc[rows]
    texts = [sub[c].astype(str).tolist() for c in TEXT_COLS]
    meta = torch.cat([torch.tensor(sub.iloc[:, 10:15].values, dtype=torch.float32), torch.zeros(B, 1)], 1)
    cat = [sub["category"].astype(str).tolist(), sub["title"].astype(str).tolist()]
    with torch.no_grad():
        ref, _ = orig(img, texts, meta, cat)
        x, _ = store.get(rows, dev)
        x["meta"] = torch.cat([x["meta"], torch.zeros(B, 1)], 1)
        out = head(**x)
    diff = (ref[:, :29] - out).abs()
    print(f"{B} videos, 29 days: max |diff| {diff.max():.2e}, mean |diff| {diff.mean():.2e}, "
          f"output scale {ref.abs().mean():.2e}")
    print("PARITY OK" if diff.max() < 1e-2 else "PARITY FAILED")


if __name__ == "__main__":
    main()

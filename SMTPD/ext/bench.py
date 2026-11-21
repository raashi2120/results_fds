"""Two-minute speed check, for deciding where to run things (e.g. on the lab computer).

  python -m ext.bench            # head training speed + per-run estimate
  python -m ext.bench --backbones   # also time ResNet-101 + BERT x5 (feature extraction speed)
"""
import argparse
import time

import torch

from ext.common import pick_device
from ext.model import CustomLoss, PopularityHead

N_TRAIN, N_ALL = int(282481 * 0.7), 282481


def sync(d):
    if d.type == "cuda":
        torch.cuda.synchronize()
    elif d.type == "mps":
        torch.mps.synchronize()


def timeit(fn, d, n=10):
    for _ in range(3):
        fn()
    sync(d); t = time.time()
    for _ in range(n):
        fn()
    sync(d)
    return (time.time() - t) / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto")
    p.add_argument("--backbones", action="store_true")
    args = p.parse_args()
    d = pick_device(args.device)
    name = torch.cuda.get_device_name(0) if d.type == "cuda" else d.type
    print(f"device: {name}, torch {torch.__version__}")
    m = PopularityHead(2048, True, True, 5, 30).to(d).train()
    opt, loss_fn = torch.optim.Adam(m.parameters(), 1e-3), CustomLoss()
    for B in (64, 256):
        x = dict(img=torch.randn(B, 2048, device=d), text=torch.randn(B, 5, 768, device=d),
                 cate=torch.randint(0, 16, (B,), device=d), lang=torch.randint(0, 7, (B,), device=d),
                 meta=torch.randn(B, 5, device=d))
        y = torch.randn(B, 30, device=d)

        def step():
            opt.zero_grad(); loss_fn(m(**x), y, 1, 100).backward(); opt.step()
        s = timeit(step, d)
        ep_min = s * N_TRAIN / B / 60 * 1.15  # +15% for validation
        print(f"head, batch {B:3d}: {s * 1000:5.0f} ms/step -> {ep_min:5.1f} min/epoch -> "
              f"~{ep_min * 20 / 60:.1f} h per run (about 20 epochs with early stopping)")
    if args.backbones:
        from torchvision import models
        from transformers import BertConfig, BertModel
        r = torch.nn.Sequential(*list(models.resnet101().children())[:-1]).to(d).eval()
        b = BertModel(BertConfig(vocab_size=119547)).to(d).eval()
        img, ids = torch.randn(64, 3, 224, 224, device=d), torch.randint(1000, 5000, (64, 64), device=d)
        with torch.inference_mode():
            tr = timeit(lambda: r(img), d, 5)
            tb = timeit(lambda: [b(ids) for _ in range(5)], d, 3)
        print(f"ResNet-101 {tr:.2f}s + BERT x5 {tb:.2f}s per 64 videos -> "
              f"~{(tr + tb) * N_ALL / 64 / 3600:.1f} h to extract all features (excluding image decoding)")


if __name__ == "__main__":
    main()

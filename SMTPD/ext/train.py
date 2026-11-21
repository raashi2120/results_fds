"""Train the popularity head on cached features.

Model selection uses the validation set only; the test set is scored once, at the end, with the
best-validation checkpoint (the original code selected on the test set).

Examples:
  python -m ext.train --name paper_noEP_s0                       # ResNet, no early popularity
  python -m ext.train --ep_days 1 --name paper_EP_s0             # the paper's EP setting
  python -m ext.train --image clip --name clip_noEP_s0
  python -m ext.train --split channel --name chan_paper_noEP_s0
  python -m ext.train --no_text --name abl_notext_s0             # sensitivity: drop the text branch
"""
import argparse
import copy
import json
import os
import random
import time

import numpy as np
import torch

from ext.common import DEFAULT_FEATURES, DEFAULT_RUNS, pick_device, seq_metrics
from ext.data import FeatureStore, make_split
from ext.model import CustomLoss, PopularityHead


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="run folder name under --runs_dir")
    p.add_argument("--features", default=DEFAULT_FEATURES)
    p.add_argument("--runs_dir", default=DEFAULT_RUNS)
    p.add_argument("--split", default="random", choices=["random", "channel"])
    p.add_argument("--split_seed", type=int, default=23, help="fixed across runs; only --seed varies")
    p.add_argument("--seed", type=int, default=0, help="model init / shuffling seed")
    p.add_argument("--image", default="resnet", choices=["resnet", "clip", "both", "none"])
    p.add_argument("--no_text", action="store_true")
    p.add_argument("--no_catlang", action="store_true")
    p.add_argument("--no_numeric", action="store_true")
    p.add_argument("--clip_sim", action="store_true", help="add cover-title CLIP similarity to the numeric branch")
    p.add_argument("--ep_days", type=int, default=0, help="early-popularity days given as input (0 = none)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=5, help="early stopping on validation AMAE")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3, help="paper: 1e-3 (the original code used 1e-2)")
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--in_memory", action="store_true", help="load features into RAM instead of memory-mapping")
    p.add_argument("--device", default="auto")
    p.add_argument("--overwrite", action="store_true")
    return p


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


@torch.no_grad()
def predict(model, store, idx, batch_size, device):
    model.eval()
    preds, labels, rows = [], [], []
    for b, x, y in store.batches(idx, batch_size * 4, device):
        preds.append(model(**x).float().cpu().numpy()); labels.append(y.cpu().numpy()); rows.append(b)
    return np.concatenate(rows), np.concatenate(preds), np.concatenate(labels)


def run(args):
    out_dir = os.path.join(args.runs_dir, args.name)
    if os.path.exists(os.path.join(out_dir, "metrics.json")) and not args.overwrite:
        print(f"{args.name}: already finished, skipping (use --overwrite to redo)")
        return
    os.makedirs(out_dir, exist_ok=True)
    set_seed(args.seed)
    device = pick_device(args.device)
    store = FeatureStore(args.features, image=args.image, text=not args.no_text, catlang=not args.no_catlang,
                         numeric=not args.no_numeric, clip_sim=args.clip_sim, ep_days=args.ep_days,
                         in_memory=args.in_memory)
    split = make_split(args.features, args.split, args.split_seed)
    model = PopularityHead(img_dim=store.img_dim, text=store.text, catlang=store.catlang,
                           meta_dim=store.meta_dim, n_targets=store.n_targets).to(device)
    config = dict(vars(args), device=str(device), inputs=store.describe(),
                  sizes={k: int(len(v)) for k, v in split.items()},
                  trainable_params=sum(p.numel() for p in model.parameters()))
    json.dump(config, open(os.path.join(out_dir, "config.json"), "w"), indent=2)
    print(f"{args.name} on {device}: {config['sizes']}, inputs {store.describe()}")

    loss_fn = CustomLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.2, patience=1, min_lr=0)
    rng = np.random.default_rng(args.seed)
    steps_per_epoch = len(split["train"]) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    best, best_state, best_epoch, bad, history, step = np.inf, None, -1, 0, [], 0
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        losses = []
        for _, x, y in store.batches(split["train"], args.batch_size, device, shuffle=True, rng=rng, drop_last=True):
            opt.zero_grad()
            loss = loss_fn(model(**x), y, step, total_steps)
            loss.backward()
            opt.step()
            losses.append(loss.item()); step += 1
        _, vp, vy = predict(model, store, split["val"], args.batch_size, device)
        val_loss = float(loss_fn(torch.from_numpy(vp), torch.from_numpy(vy), step, total_steps))
        sched.step(val_loss)
        vm = seq_metrics(vy, vp)
        history.append(dict(epoch=epoch + 1, train_loss=float(np.mean(losses)), val_loss=val_loss,
                            val_AMAE=vm["AMAE"], val_ASRC=vm["ASRC"], lr=opt.param_groups[0]["lr"],
                            seconds=round(time.time() - t0, 1)))
        print(f"epoch {epoch + 1:2d}  train {np.mean(losses):.4f}  val loss {val_loss:.4f}  "
              f"AMAE {vm['AMAE']:.4f}  ASRC {vm['ASRC']:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if vm["AMAE"] < best:
            best, best_state, best_epoch, bad = vm["AMAE"], copy.deepcopy(model.state_dict()), epoch + 1, 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop: no val improvement for {args.patience} epochs")
                break

    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(out_dir, "model.pt"))
    _, vp, vy = predict(model, store, split["val"], args.batch_size, device)
    rows, tp, ty = predict(model, store, split["test"], args.batch_size, device)
    has = store.has_img[rows]
    metrics = dict(best_epoch=best_epoch, val=seq_metrics(vy, vp), test=seq_metrics(ty, tp),
                   test_with_cover=seq_metrics(ty[has], tp[has]), history=history)
    np.savez_compressed(os.path.join(out_dir, "preds_test.npz"), rows=rows, ids=store.ids[rows], preds=tp,
                        labels=ty, has_img=has)
    json.dump(metrics, open(os.path.join(out_dir, "metrics.json"), "w"), indent=2)
    print(f"{args.name}: best epoch {best_epoch}, test AMAE {metrics['test']['AMAE']:.4f} "
          f"ASRC {metrics['test']['ASRC']:.4f}")


if __name__ == "__main__":
    run(build_parser().parse_args())

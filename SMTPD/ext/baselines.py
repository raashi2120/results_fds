"""Simple baselines on the same splits as ext.train (extension #1).

Inputs: early-popularity days (if --ep_days > 0), the 5 numeric features, category and title language.
Models:
  mean         train-set mean popularity for each day
  persistence  every future day = last observed day (only with --ep_days > 0)
  ridge        linear regression (one-hot category/language)
  catboost     gradient-boosted trees, one multi-output model, early-stopped on validation

Examples:
  python -m ext.baselines --ep_days 1
  python -m ext.baselines --ep_days 0 --split channel --models ridge,catboost
"""
import argparse
import json
import os
import time

import numpy as np

from ext.common import DEFAULT_FEATURES, DEFAULT_RUNS, N_CATE, N_LANG, seq_metrics
from ext.data import make_split


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", default=DEFAULT_FEATURES)
    p.add_argument("--runs_dir", default=DEFAULT_RUNS)
    p.add_argument("--split", default="random", choices=["random", "channel"])
    p.add_argument("--split_seed", type=int, default=23)
    p.add_argument("--ep_days", type=int, default=0)
    p.add_argument("--models", default="mean,persistence,ridge,catboost")
    p.add_argument("--cb_iterations", type=int, default=3000)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    tab = np.load(os.path.join(args.features, "tabular.npz"))
    has_img = np.load(os.path.join(args.features, "has_img.npy"))
    split = make_split(args.features, args.split, args.split_seed)
    k = args.ep_days
    Y = tab["days"][:, k:]
    num = np.concatenate([tab["numeric"], tab["days"][:, :k]], 1)
    cate, lang = tab["cate"].astype(int), tab["lang"].astype(int)
    tr, va, te = split["train"], split["val"], split["test"]

    for name in args.models.split(","):
        out_dir = os.path.join(args.runs_dir, "baselines", f"{args.split}_ep{k}_{name}")
        if os.path.exists(os.path.join(out_dir, "metrics.json")) and not args.overwrite:
            print(f"{out_dir}: done, skipping"); continue
        if name == "persistence" and k == 0:
            continue
        t0 = time.time()
        if name == "mean":
            pred = np.tile(Y[tr].mean(0), (len(te), 1)); vpred = np.tile(Y[tr].mean(0), (len(va), 1))
        elif name == "persistence":
            pred = np.repeat(tab["days"][te, k - 1:k], Y.shape[1], 1)
            vpred = np.repeat(tab["days"][va, k - 1:k], Y.shape[1], 1)
        elif name == "ridge":
            from sklearn.linear_model import Ridge
            X = np.concatenate([num, np.eye(N_CATE)[cate], np.eye(N_LANG)[lang]], 1)
            m = Ridge(alpha=1.0).fit(X[tr], Y[tr]); pred, vpred = m.predict(X[te]), m.predict(X[va])
        elif name == "catboost":
            import pandas as pd
            from catboost import CatBoostRegressor, Pool
            cols = [f"num{i}" for i in range(num.shape[1])]
            X = pd.DataFrame(num, columns=cols); X["cate"] = cate; X["lang"] = lang
            pool = lambda i: Pool(X.iloc[i], Y[i], cat_features=["cate", "lang"])
            m = CatBoostRegressor(loss_function="MultiRMSE", iterations=args.cb_iterations, learning_rate=0.1,
                                  depth=6, od_type="Iter", od_wait=100, random_seed=0, verbose=250,
                                  thread_count=-1)
            m.fit(pool(tr), eval_set=pool(va), use_best_model=True)
            pred, vpred = m.predict(X.iloc[te]), m.predict(X.iloc[va])
        else:
            raise SystemExit(f"unknown model {name}")
        os.makedirs(out_dir, exist_ok=True)
        has = has_img[te]
        metrics = dict(val=seq_metrics(Y[va], vpred), test=seq_metrics(Y[te], pred),
                       test_with_cover=seq_metrics(Y[te][has], pred[has]), seconds=round(time.time() - t0, 1))
        json.dump(metrics, open(os.path.join(out_dir, "metrics.json"), "w"), indent=2)
        np.savez_compressed(os.path.join(out_dir, "preds_test.npz"), rows=te, ids=tab["ids"][te], preds=pred,
                            labels=Y[te], has_img=has)
        print(f"{args.split} ep{k} {name:12s} test AMAE {metrics['test']['AMAE']:.4f} "
              f"ASRC {metrics['test']['ASRC']:.4f}  ({metrics['seconds']:.0f}s)")


if __name__ == "__main__":
    main()

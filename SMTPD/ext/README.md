# Extension pipeline

ResNet-101 and BERT are frozen in the original model. They make up 243M of its 244M parameters,
and recomputing them every epoch is the whole training cost (about 20 s per batch on a laptop CPU).
This pipeline computes their outputs once and trains only the trainable part (`ext/model.py`,
which has the same layers as `smp_model.youtube_lstm3`).

All commands run from the `SMTPD/` folder with the virtual environment active:

```bash
source .venv/bin/activate
```

Default paths, relative to the project folder (`fds proj/`):

| | Path |
|---|---|
| Dataset CSV and covers | `data_source/basic_view_pn.csv`, `data_source/img_yt/` |
| Cached features | `features/` |
| Training runs | `runs/<name>/` (`config.json`, `metrics.json`, `preds_test.npz`, `model.pt`) |

## Steps

| Step | Command | Where it runs | Time |
|---|---|---|---|
| 1. Cache features (resumable) | `python -m ext.extract_features` | Mac, overnight | about 4–5 h |
| 2. Simple baselines | `python -m ext.baselines --ep_days 1` (and `--ep_days 0`, and `--split channel`) | Mac | minutes |
| 2b. Channel-split CatBoost | `python -m ext.baselines --split channel --ep_days 1 --models catboost` (and `--ep_days 0`) | Lab CPU, alongside the grid | about 5–6 h each on the Mac |
| 3. Training grid | `python -m ext.grid --dry_run`, then `python -m ext.grid --phase 3,4,5 --parallel 3` | Lab GPU | about 5–8 h |
| 4. Single run | `python -m ext.train --name test_run --epochs 2` | Either | |
| Speed check | `python -m ext.bench` (add `--backbones` for extraction speed) | Lab GPU first | 2 min |
| Fidelity check | `python -m ext.parity_check` | Mac | 1 min |

**Copying to the lab computer:** you need `SMTPD/` (excluding `.venv`), `features/`, and
`pip install torch numpy scipy scikit-learn pandas catboost`. The lab computer doesn't need the
images or the BERT model.

## Deliberate differences from the original code (state these in the write-up)

1. **Feature extraction.** ResNet and BERT run in eval mode on ImageNet-normalised images.
   In the original, `model.train()` left ResNet's batch-norm statistics drifting, kept BERT's
   dropout active, and fed ResNet un-normalised pixels.
2. **Train/validation/test split, 70/10/20.** It is fixed by `--split_seed 23` and saved in
   `features/splits/`, and only the model seed varies. The original chose its checkpoint on the
   test set (`test_indices = val_indices`).
3. **Learning rate 1e-3,** following the paper. The original code used 1e-2.
4. **Early popularity.** `--ep_days 0` predicts Days 1–30, matching the paper's no-EP setting.
   `--ep_days 1` predicts Days 2–30 from Day 1, matching the paper's EP setting. The original
   code's default (no EP, predicting Days 2–30) matches neither.
5. **Fixes:**
   - Early stopping works (the original reset its counter every epoch).
   - "Travel  Events" maps to the Travel category instead of unknown.
   - The unused `l2_reg` loop is removed.
   - Any batch size works.

## Dataset note

16,904 of the cover files are HTML pages saved as `.jpg`, and 13 covers are missing. So **6.0% of
videos have no usable cover.** As in the paper, they get an all-zero image. `has_img` marks them,
and every run also reports `test_with_cover` metrics, computed on videos with a real cover only.

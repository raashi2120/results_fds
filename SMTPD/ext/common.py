"""Shared constants and helpers for the extension pipeline."""
import os

import numpy as np
import torch
from scipy import stats

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.dirname(REPO_DIR)
DEFAULT_CSV = os.path.join(PROJECT_DIR, "data_source", "basic_view_pn.csv")
DEFAULT_IMAGES = os.path.join(PROJECT_DIR, "data_source", "img_yt")
DEFAULT_FEATURES = os.path.join(PROJECT_DIR, "features")
DEFAULT_RUNS = os.path.join(PROJECT_DIR, "runs")
BERT_PATH = os.environ.get("BERT_PATH", os.path.join(PROJECT_DIR, "bert_multilingual"))
CLIP_IMAGE_MODEL = "openai/clip-vit-base-patch32"
CLIP_TEXT_MODEL = "sentence-transformers/clip-ViT-B-32-multilingual-v1"

# CSV column layout (matches smp_data.py).
TEXT_COLS = ["user_id", "category", "title", "tags", "description"]
NUMERIC_COLS = ["user_subscribers", "video_duration", "title_length", "tag_num", "desc_length"]
DAY_COLS = [f"Day {i}" for i in range(1, 31)]

# Same vocabularies as smp_model.py. "Travel  Events" is what the data actually
# contains; the original vocab only had "Travel", so those videos fell to OOA.
CATE_VOCAB = {"People  Blogs": 1, "Gaming": 2, "News  Politics": 3, "Entertainment": 4,
              "Music": 5, "Education": 6, "Sports": 7, "Howto  Style": 8,
              "Film  Animation": 9, "Nonprofits  Activism": 10, "Travel": 11,
              "Travel  Events": 11, "Comedy": 12, "Science  Technology": 13,
              "Autos  Vehicles": 14, "Pets  Animals": 15}
LANG_VOCAB = {"en": 1, "zh": 2, "ko": 3, "ja": 4, "hi": 5, "ru": 6}
N_CATE, N_LANG = 16, 7


def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seq_metrics(labels, preds):
    """Per-day MAE / MSE / Spearman, plus their averages (AMAE, AMSE, ASRC), as in tools.print_output_seq."""
    labels, preds = np.asarray(labels, dtype=np.float64), np.asarray(preds, dtype=np.float64)
    mae = np.abs(labels - preds).mean(0)
    mse = ((labels - preds) ** 2).mean(0)
    src = np.array([stats.spearmanr(labels[:, i], preds[:, i])[0] for i in range(labels.shape[1])])
    return {"AMAE": float(mae.mean()), "AMSE": float(mse.mean()), "ASRC": float(np.nanmean(src)),
            "MAE": mae.tolist(), "MSE": mse.tolist(), "SRC": src.tolist()}

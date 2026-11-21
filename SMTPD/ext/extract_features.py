"""Compute every frozen-backbone feature once and save it to disk.

Outputs in --out_dir (row i == CSV row i):
  tabular.npz    ids, numeric [N,5], days [N,30], cate [N], lang [N], user [N] (channel code), has_img [N]
  resnet.npy     [N,2048] float16  ResNet-101 pooled features (eval mode, ImageNet-normalised input)
  clip_img.npy   [N,512]  float16  CLIP ViT-B/32 image embedding
  bert.npy       [N,5,768] float16 BERT-Multilingual pooled output for the 5 text fields
  clip_txt.npy   [N,512]  float16  multilingual CLIP text embedding of the title
  clip_sim.npy   [N]      float32  cosine(cover, title) in CLIP space (0 where the cover is missing)

The run is resumable: progress.json records how many rows of each group are done.

Usage:
  python -m ext.extract_features                      # everything, default paths
  python -m ext.extract_features --limit 10000 --out_dir ../features_10k
"""
import argparse
import json
import os
import time

import langid
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

from ext.common import (BERT_PATH, CATE_VOCAB, CLIP_IMAGE_MODEL, CLIP_TEXT_MODEL, DAY_COLS,
                        DEFAULT_CSV, DEFAULT_FEATURES, DEFAULT_IMAGES, LANG_VOCAB, NUMERIC_COLS,
                        TEXT_COLS, pick_device)

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
CLIP_MEAN, CLIP_STD = [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]


class CoverImages(Dataset):
    """Decodes each cover once and returns the ResNet view and the CLIP view."""

    def __init__(self, ids, images_dir):
        self.ids, self.images_dir = ids, images_dir
        # The original pipeline resized straight to 224x224 (no crop); kept for ResNet.
        self.resnet_tf = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor(),
                                             transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        self.clip_tf = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(CLIP_MEAN, CLIP_STD)])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        try:
            img = Image.open(os.path.join(self.images_dir, self.ids[i] + ".jpg")).convert("RGB")
            return self.resnet_tf(img), self.clip_tf(img), True
        except Exception:
            # Paper: missing covers are replaced with an all-zero image.
            z = torch.zeros(3, 224, 224)
            return z, z.clone(), False


class Progress:
    def __init__(self, out_dir, n):
        self.path = os.path.join(out_dir, "progress.json")
        self.state = json.load(open(self.path)) if os.path.exists(self.path) else {"n": n}
        if self.state["n"] != n:
            raise SystemExit(f"{out_dir} was started with {self.state['n']} rows, not {n}. "
                             "Use a different --out_dir or delete progress.json.")

    def done(self, group):
        return self.state.get(group, 0)

    def set(self, group, rows):
        self.state[group] = rows
        tmp = self.path + ".tmp"
        json.dump(self.state, open(tmp, "w"))
        os.replace(tmp, self.path)


def open_array(out_dir, name, shape, dtype, fresh):
    path = os.path.join(out_dir, name)
    if fresh or not os.path.exists(path):
        return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    arr = np.load(path, mmap_mode="r+")
    assert arr.shape == shape, f"{path} has shape {arr.shape}, expected {shape}"
    return arr


def write_tabular(df, out_dir):
    path = os.path.join(out_dir, "tabular.npz")
    if os.path.exists(path):
        return
    t = time.time()
    titles = df["title"].astype(str).tolist()
    lang = np.array([LANG_VOCAB.get(langid.classify(x)[0], 0) for x in tqdm(titles, desc="langid")], np.int8)
    np.savez(path,
             ids=df["video_id"].astype(str).to_numpy(dtype=str),  # fixed-width unicode, loads without pickle
             numeric=df[NUMERIC_COLS].values.astype(np.float32),
             days=df[DAY_COLS].values.astype(np.float32),
             cate=df["category"].map(CATE_VOCAB).fillna(0).astype(np.int8).values,
             lang=lang,
             user=pd.factorize(df["user_id"].astype(str))[0].astype(np.int32))
    print(f"tabular.npz written ({time.time() - t:.0f}s)")


def extract_images(df, args, prog, device):
    n, start = len(df), prog.done("images")
    if start >= n:
        return
    from transformers import CLIPModel
    resnet = torch.nn.Sequential(*list(models.resnet101(weights="IMAGENET1K_V1").children())[:-1]).to(device).eval()
    clip = CLIPModel.from_pretrained(CLIP_IMAGE_MODEL).to(device).eval()
    fresh = start == 0
    r_arr = open_array(args.out_dir, "resnet.npy", (n, 2048), np.float16, fresh)
    c_arr = open_array(args.out_dir, "clip_img.npy", (n, 512), np.float16, fresh)
    h_arr = open_array(args.out_dir, "has_img.npy", (n,), np.bool_, fresh)
    ids = df["video_id"].astype(str).tolist()[start:]
    loader = DataLoader(CoverImages(ids, args.images_dir), batch_size=args.batch_size,
                        num_workers=args.num_workers, shuffle=False)
    pos = start
    with torch.inference_mode():
        for k, (x_r, x_c, ok) in enumerate(tqdm(loader, desc="images (ResNet+CLIP)", initial=start // args.batch_size,
                                                total=(n + args.batch_size - 1) // args.batch_size)):
            r = resnet(x_r.to(device)).flatten(1)
            c = clip.get_image_features(pixel_values=x_c.to(device))
            c = c if torch.is_tensor(c) else c.pooler_output
            b = len(ok)
            r_arr[pos:pos + b] = r.float().cpu().numpy().astype(np.float16)
            c_arr[pos:pos + b] = c.float().cpu().numpy().astype(np.float16)
            h_arr[pos:pos + b] = ok.numpy()
            pos += b
            if k % 50 == 49:
                r_arr.flush(); c_arr.flush(); h_arr.flush(); prog.set("images", pos)
    r_arr.flush(); c_arr.flush(); h_arr.flush(); prog.set("images", pos)
    del resnet, clip


def extract_bert(df, args, prog, device):
    n, start = len(df), prog.done("bert")
    if start >= n:
        return
    from transformers import BertModel, BertTokenizer
    tok = BertTokenizer.from_pretrained(BERT_PATH)
    bert = BertModel.from_pretrained(BERT_PATH).to(device).eval()
    arr = open_array(args.out_dir, "bert.npy", (n, 5, 768), np.float16, start == 0)
    # str() matches smp_data.py, so missing values become the string "nan" as in the original.
    fields = [df[c].astype(str).tolist() for c in TEXT_COLS]
    B = args.batch_size
    with torch.inference_mode():
        for k, s in enumerate(tqdm(range(start, n, B), desc="BERT x5 fields")):
            e = min(s + B, n)
            outs = []
            for f in fields:
                enc = tok(f[s:e], max_length=64, truncation=True, padding="max_length", return_tensors="pt")
                o = bert(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device))
                outs.append(o.pooler_output)
            arr[s:e] = torch.stack(outs, 1).float().cpu().numpy().astype(np.float16)
            if k % 50 == 49:
                arr.flush(); prog.set("bert", e)
    arr.flush(); prog.set("bert", n)
    del bert


def extract_clip_text(df, args, prog, device):
    n, start = len(df), prog.done("clip_txt")
    if start >= n:
        return
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(CLIP_TEXT_MODEL, device=str(device))
    arr = open_array(args.out_dir, "clip_txt.npy", (n, 512), np.float16, start == 0)
    titles = df["title"].astype(str).tolist()
    chunk = args.batch_size * 50
    for s in tqdm(range(start, n, chunk), desc="CLIP title text"):
        e = min(s + chunk, n)
        arr[s:e] = st.encode(titles[s:e], batch_size=args.batch_size, convert_to_numpy=True).astype(np.float16)
        arr.flush(); prog.set("clip_txt", e)
    del st


def write_clip_sim(args):
    path = os.path.join(args.out_dir, "clip_sim.npy")
    img = np.load(os.path.join(args.out_dir, "clip_img.npy"), mmap_mode="r")
    txt = np.load(os.path.join(args.out_dir, "clip_txt.npy"), mmap_mode="r")
    has = np.load(os.path.join(args.out_dir, "has_img.npy"))
    sim = np.zeros(len(has), np.float32)
    for s in range(0, len(has), 50000):
        a = img[s:s + 50000].astype(np.float32); b = txt[s:s + 50000].astype(np.float32)
        a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-8
        b /= np.linalg.norm(b, axis=1, keepdims=True) + 1e-8
        sim[s:s + 50000] = (a * b).sum(1)
    sim[~has] = 0.0
    np.save(path, sim)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default=DEFAULT_CSV)
    p.add_argument("--images_dir", default=DEFAULT_IMAGES)
    p.add_argument("--out_dir", default=DEFAULT_FEATURES)
    p.add_argument("--limit", type=int, default=0, help="only the first N rows (for testing)")
    p.add_argument("--groups", default="images,bert,clip_txt", help="comma list: images,bert,clip_txt")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.csv, nrows=args.limit or None)
    device = pick_device(args.device)
    print(f"{len(df)} rows -> {args.out_dir} on {device}")
    prog = Progress(args.out_dir, len(df))
    t0 = time.time()
    write_tabular(df, args.out_dir)
    groups = args.groups.split(",")
    if "images" in groups:
        extract_images(df, args, prog, device)
    if "bert" in groups:
        extract_bert(df, args, prog, device)
    if "clip_txt" in groups:
        extract_clip_text(df, args, prog, device)
    if prog.done("images") >= len(df) and prog.done("clip_txt") >= len(df):
        write_clip_sim(args)
    print(f"done in {(time.time() - t0) / 60:.1f} min; progress: {prog.state}")


if __name__ == "__main__":
    main()

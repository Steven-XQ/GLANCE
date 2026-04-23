"""
Extract per-frame BNInception features for MECCANO and store in LMDB.

Produces LMDB files compatible with FeaturesHOLoader in holoaders.py.
Each entry: key = '{video_id}/{frame:05d}.jpg', value = dict with 'GLOBAL_FEAT' (1024-D).

Usage:
    python scripts/extract_meccano_features.py --split train   # train + val combined
    python scripts/extract_meccano_features.py --split test
"""

import os
import sys
import argparse
import pickle
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='train', choices=['train', 'test'],
                        help='train feats_train LMDB (covers train+val); test feats_test LMDB')
    parser.add_argument('--frames_dir',
                        default='/scratch/u6cu/sx2022.u6cu/datasets/MECCANO/extracted_frames')
    parser.add_argument('--model_path',
                        default='./common/rulstm/FEATEXT/models/ek100/TSN-rgb-ek100.pth.tar')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def build_model(model_path, device):
    from pretrainedmodels import bninception
    model = bninception(pretrained=None)
    state_dict = torch.load(model_path, map_location='cpu')['state_dict']
    state_dict = {k.replace('module.base_model.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.last_linear = nn.Identity()
    model.global_pool = nn.AdaptiveAvgPool2d(1)
    model.to(device)
    model.eval()
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize([256, 454]),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x[[2, 1, 0], ...] * 255),  # RGB -> BGR
        transforms.Normalize(mean=[104, 117, 128], std=[1, 1, 1]),
    ])


def collect_frames(split):
    """Return dict video_id -> set of frame indices for the given split.

    split='train' combines MECCANO_train_split.csv + MECCANO_val_split.csv (used for feats_train).
    split='test'  uses MECCANO_test_split.csv (used for feats_test).
    """
    import pandas as pd

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    annot_dir = os.path.join(base_dir, 'common', 'meccano-annotations')
    uid2future_path = os.path.join(base_dir, 'data', 'uid2future_file_name_meccano.pickle')

    # feats_train.lmdb: train split only. feats_test.lmdb: val + test
    # (val frames are needed by --evaluate --traj_only which uses mode!='train' -> feats_test).
    if split == 'train':
        csvs = ['MECCANO_train_split.csv']
    else:
        csvs = ['MECCANO_val_split.csv', 'MECCANO_test_split.csv']
    dfs = [pd.read_csv(os.path.join(annot_dir, c)) for c in csvs]
    df = pd.concat(dfs, ignore_index=True)

    with open(uid2future_path, 'rb') as f:
        uid2future = pickle.load(f)

    # Sampling params for MECCANO (match datasetopts.py meccano branch)
    fps = 6.0
    ori_fps = 12.0
    t_buffer = 10.0 / 6.0
    t_ant = 0.5
    num_obs = int(np.floor(t_buffer * fps))

    frames_needed = {}  # video_id (4-digit str) -> set of int frame indices

    for _, row in df.iterrows():
        video_id = str(row['video_id']).zfill(4)
        start_frame = int(str(row['start_frame']).replace('.jpg', ''))

        frames_needed.setdefault(video_id, set())

        # Observation frames (same math as input_loaders.sample_history_frames)
        time_start = (start_frame - 1) / ori_fps
        time_ant = time_start - t_ant
        times = (np.arange(1, num_obs + 1) - num_obs) / fps + time_ant
        times = np.clip(times, 0, np.inf).astype(np.float32)
        frame_idxs = np.floor(times * ori_fps).astype(np.int32) + 1
        for fidx in frame_idxs:
            frames_needed[video_id].add(int(fidx))

        uid = int(row['uid'])
        for fpath in uid2future.get(uid, []):
            # "{video_id}/{:05d}.jpg"
            _, fname = fpath.split('/', 1)
            fidx = int(fname.replace('.jpg', ''))
            frames_needed[video_id].add(fidx)

    total = sum(len(v) for v in frames_needed.values())
    print(f"  {split}: {len(frames_needed)} videos, {total} frames to extract")
    return frames_needed


def extract_features(args):
    from lmdbdict import lmdbdict

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_sub = 'feats_train' if args.split == 'train' else 'feats_test'
    lmdb_dir = os.path.join(base_dir, 'data', 'meccano', out_sub)
    os.makedirs(lmdb_dir, exist_ok=True)
    lmdb_path = os.path.join(lmdb_dir, 'data.lmdb')

    print(f"Building model from {args.model_path}...")
    model = build_model(args.model_path, args.device)
    transform = get_transform()

    print(f"Collecting frames for {args.split} split...")
    frames_needed = collect_frames(args.split)

    print(f"Extracting features to {lmdb_path}...")
    env = lmdbdict(lmdb_path, mode='w')

    for video_id in tqdm(sorted(frames_needed.keys()), desc='Videos'):
        frame_indices = sorted(frames_needed[video_id])
        video_dir = os.path.join(args.frames_dir, video_id)
        if not os.path.isdir(video_dir):
            print(f"  WARNING: {video_dir} not found, skipping")
            continue

        for batch_start in range(0, len(frame_indices), args.batch_size):
            batch_indices = frame_indices[batch_start:batch_start + args.batch_size]
            batch_imgs, batch_keys = [], []
            for fidx in batch_indices:
                fpath = os.path.join(video_dir, f"{fidx:05d}.jpg")
                key = f"{video_id}/{fidx:05d}.jpg"
                if not os.path.exists(fpath):
                    continue
                try:
                    img = Image.open(fpath).convert('RGB')
                    batch_imgs.append(transform(img))
                    batch_keys.append(key)
                except Exception as e:
                    print(f"  WARNING: Failed to load {fpath}: {e}")

            if not batch_imgs:
                continue

            batch_tensor = torch.stack(batch_imgs).to(args.device)
            with torch.no_grad():
                feats = model(batch_tensor)
            feats = feats.squeeze().detach().cpu().numpy()
            if feats.ndim == 1:
                feats = feats.reshape(1, -1)

            for i, key in enumerate(batch_keys):
                env[key.encode()] = {'GLOBAL_FEAT': feats[i].astype(np.float32)}
        env.flush()

    del env
    print(f"Done! Features saved to {lmdb_path}")


if __name__ == '__main__':
    args = get_args()
    extract_features(args)

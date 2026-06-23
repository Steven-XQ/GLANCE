import os
import sys
import argparse
import pickle
import json
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
                        help='Which split to extract features for')
    parser.add_argument('--frames_dir', default='/scratch/u6cu/sx2022.u6cu/datasets/EGTEA_Gaze_Plus/EGTEA/extracted_frames',
                        help='Path to extracted EGTEA frames')
    parser.add_argument('--model_path', default='./common/rulstm/FEATEXT/models/ek100/TSN-rgb-ek100.pth.tar',
                        help='Path to pretrained TSN BNInception weights')
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


def collect_frames_to_extract(split, frames_dir):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load uid2future for future frames
    uid2future_path = os.path.join(base_dir, 'data', 'uid2future_file_name_egtea.pickle')
    with open(uid2future_path, 'rb') as f:
        uid2future = pickle.load(f)

    # Load annotation CSVs
    import pandas as pd
    if split == 'train':
        csv_path = os.path.join(base_dir, 'common', 'egtea-annotations', 'EGTEA_train_split1.csv')
    else:
        csv_path = os.path.join(base_dir, 'common', 'egtea-annotations', 'EGTEA_validation_split1.csv')
    df = pd.read_csv(csv_path)

    # Collect all frames we need
    # For each action: observation frames + future frames
    fps = 6.0
    ori_fps = 30.0
    t_buffer = 10.0 / 6.0
    t_ant = 0.5
    num_obs = int(np.floor(t_buffer * fps))

    frames_needed = {}  # video_id -> set of frame indices

    for _, row in df.iterrows():
        video_id = row['video_id']
        start_frame = int(row['start_frame'])

        if video_id not in frames_needed:
            frames_needed[video_id] = set()

        # Compute observation frame indices (same logic as input_loaders.py)
        time_start = (start_frame - 1) / ori_fps
        time_ant = time_start - t_ant
        times = (np.arange(1, num_obs + 1) - num_obs) / fps + time_ant
        times = np.clip(times, 0, np.inf).astype(np.float32)
        frame_idxs = np.floor(times * ori_fps).astype(np.int32) + 1

        for fidx in frame_idxs:
            frames_needed[video_id].add(int(fidx))

        # Future frames
        uid = int(row['uid'])
        if uid in uid2future:
            for fpath in uid2future[uid]:
                # fpath format: "{video_id}/frame_{:010d}.jpg"
                fname = fpath.split('/')[-1]
                fidx = int(fname.replace('frame_', '').replace('.jpg', ''))
                frames_needed[video_id].add(fidx)

    # Verify frames exist
    total_frames = sum(len(v) for v in frames_needed.values())
    print(f"  {split}: {len(frames_needed)} videos, {total_frames} total frames to extract")

    return frames_needed


def extract_features(args):
    from lmdbdict import lmdbdict

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if args.split == 'train':
        lmdb_dir = os.path.join(base_dir, 'data', 'egtea', 'feats_train')
    else:
        lmdb_dir = os.path.join(base_dir, 'data', 'egtea', 'feats_test')

    os.makedirs(lmdb_dir, exist_ok=True)
    lmdb_path = os.path.join(lmdb_dir, 'data.lmdb')

    print(f"Building model from {args.model_path}...")
    model = build_model(args.model_path, args.device)
    transform = get_transform()

    print(f"Collecting frames for {args.split} split...")
    frames_needed = collect_frames_to_extract(args.split, args.frames_dir)

    print(f"Extracting features to {lmdb_path}...")
    env = lmdbdict(lmdb_path, mode='w')

    for video_id in tqdm(sorted(frames_needed.keys()), desc='Videos'):
        frame_indices = sorted(frames_needed[video_id])
        video_dir = os.path.join(args.frames_dir, video_id)

        if not os.path.exists(video_dir):
            print(f"  WARNING: {video_dir} not found, skipping")
            continue

        # Process in batches
        for batch_start in range(0, len(frame_indices), args.batch_size):
            batch_indices = frame_indices[batch_start:batch_start + args.batch_size]
            batch_imgs = []
            batch_keys = []
            batch_valid = []

            for fidx in batch_indices:
                fpath = os.path.join(video_dir, f"frame_{fidx:010d}.jpg")
                key = f"{video_id}/frame_{fidx:010d}.jpg"

                if not os.path.exists(fpath):
                    continue

                try:
                    img = Image.open(fpath).convert('RGB')
                    img_tensor = transform(img)
                    batch_imgs.append(img_tensor)
                    batch_keys.append(key)
                    batch_valid.append(True)
                except Exception as e:
                    print(f"  WARNING: Failed to load {fpath}: {e}")
                    continue

            if len(batch_imgs) == 0:
                continue

            batch_tensor = torch.stack(batch_imgs).to(args.device)
            with torch.no_grad():
                feats = model(batch_tensor)  # (B, 1024)
            feats = feats.squeeze().detach().cpu().numpy()
            if feats.ndim == 1:
                feats = feats.reshape(1, -1)

            for i, key in enumerate(batch_keys):
                result_dict = {
                    'GLOBAL_FEAT': feats[i].astype(np.float32),
                }
                env[key.encode()] = result_dict
        env.flush()

    del env
    print(f"Done! Features saved to {lmdb_path}")


if __name__ == '__main__':
    args = get_args()
    extract_features(args)

import os
import sys
import json
import pickle
import shutil
import zipfile
import numpy as np
import pandas as pd

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OCT_DIR = os.path.join(BASE_DIR, "CVPR2022-hoi-forecast-training-data")
DATA_DIR = os.path.join(BASE_DIR, "data", "egtea")
ANNOT_DIR = os.path.join(BASE_DIR, "common", "egtea-annotations")
RULSTM_DIR = os.path.join(BASE_DIR, "common", "rulstm", "RULSTM", "data", "egtea")
EGTEA_FRAMES_DIR = "/scratch/u6cu/sx2022.u6cu/datasets/EGTEA_Gaze_Plus/EGTEA/extracted_frames"


def build_uid_mapping():
    egtea_csv = os.path.join(OCT_DIR, "evaluation_labels", "egtea.csv")
    df = pd.read_csv(egtea_csv, header=None,
                     names=['uid', 'video_id', 'start_frame', 'stop_frame',
                            'verb_class', 'noun_class', 'action_class'])

    # oct_index -> (rulstm_uid, video_id)
    oct_to_rulstm = {}
    for row_idx, row in df.iterrows():
        oct_to_rulstm[row_idx] = {
            'rulstm_uid': int(row['uid']),
            'video_id': row['video_id'],
            'start_frame': int(row['start_frame']),
        }
    return oct_to_rulstm


def extract_and_remap_labels(oct_to_rulstm):
    label_dir = os.path.join(DATA_DIR, "labels")
    os.makedirs(label_dir, exist_ok=True)

    zip_path = os.path.join(OCT_DIR, "raw_egtea", "raw_egtea.zip")
    if not os.path.exists(zip_path):
        print(f"ERROR: {zip_path} not found")
        sys.exit(1)

    print(f"  Extracting and remapping labels from {zip_path}...")
    remapped = 0
    skipped = 0
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for member in zf.namelist():
            if not (member.startswith("labels/") and member.endswith(".pkl")):
                continue
            fname = os.path.basename(member)
            oct_uid = int(fname.replace("label_", "").replace(".pkl", ""))

            if oct_uid not in oct_to_rulstm:
                skipped += 1
                continue

            rulstm_uid = oct_to_rulstm[oct_uid]['rulstm_uid']
            target = os.path.join(label_dir, f"label_{rulstm_uid}.pkl")

            if not os.path.exists(target):
                with zf.open(member) as src:
                    data = src.read()
                with open(target, 'wb') as dst:
                    dst.write(data)
            remapped += 1

    print(f"  Remapped {remapped} labels (skipped {skipped} with no RULSTM mapping)")
    return remapped


def remap_eval_labels(oct_to_rulstm):
    src = os.path.join(OCT_DIR, "evaluation_labels", "egtea_eval_labels.pkl")
    dst = os.path.join(DATA_DIR, "egtea_eval_labels.pkl")

    with open(src, 'rb') as f:
        eval_labels_oct = pickle.load(f)

    eval_labels_rulstm = {}
    remapped = 0
    skipped = 0
    for oct_uid, label_data in eval_labels_oct.items():
        if oct_uid in oct_to_rulstm:
            rulstm_uid = oct_to_rulstm[oct_uid]['rulstm_uid']
            eval_labels_rulstm[rulstm_uid] = label_data
            remapped += 1
        else:
            skipped += 1

    with open(dst, 'wb') as f:
        pickle.dump(eval_labels_rulstm, f)
    print(f"  Remapped {remapped} eval labels (skipped {skipped}), saved to {dst}")


def generate_video_info():
    label_dir = os.path.join(DATA_DIR, "labels")
    uids = []
    for f in os.listdir(label_dir):
        if f.startswith("label_") and f.endswith(".pkl"):
            uid = int(f.replace("label_", "").replace(".pkl", ""))
            uids.append(uid)
    uids.sort()

    info_path = os.path.join(DATA_DIR, "video_info.json")
    with open(info_path, 'w') as f:
        json.dump(uids, f)
    print(f"  Generated video_info.json with {len(uids)} UIDs")


def generate_uid2future(oct_to_rulstm):
    label_dir = os.path.join(DATA_DIR, "labels")

    # Build rulstm_uid -> video_id mapping
    uid2video = {}
    for oct_uid, info in oct_to_rulstm.items():
        uid2video[info['rulstm_uid']] = info['video_id']

    uid2future = {}
    for f in sorted(os.listdir(label_dir)):
        if not f.endswith('.pkl'):
            continue
        rulstm_uid = int(f.replace("label_", "").replace(".pkl", ""))
        if rulstm_uid not in uid2video:
            continue

        video_id = uid2video[rulstm_uid]

        with open(os.path.join(label_dir, f), 'rb') as fh:
            label_info = pickle.load(fh)

        frame_indices = label_info['frame_indices']
        # frame_indices[0] is the last observation frame
        # frame_indices[1:] are future frames
        future_frames = frame_indices[1:]
        future_paths = []
        for fidx in future_frames:
            fname = f"{video_id}/frame_{int(fidx):010d}.jpg"
            future_paths.append(fname)
        uid2future[rulstm_uid] = future_paths

    out_path = os.path.join(BASE_DIR, "data", "uid2future_file_name_egtea.pickle")
    with open(out_path, 'wb') as f:
        pickle.dump(uid2future, f)
    print(f"  Generated uid2future with {len(uid2future)} entries")


def create_annotations():
    os.makedirs(ANNOT_DIR, exist_ok=True)

    # Load actions.csv for action label lookup
    actions_df = pd.read_csv(os.path.join(RULSTM_DIR, "actions.csv"), header=None,
                             names=['action_class', 'verb_noun', 'action_label'])
    actions_df['action_label'] = actions_df['action_label'].str.strip()
    actions_df['verb_noun'] = actions_df['verb_noun'].str.strip()

    action_info = {}
    for _, row in actions_df.iterrows():
        parts = row['verb_noun'].split('_')
        verb_class = int(parts[0])
        noun_class = int(parts[1])
        label_parts = row['action_label'].split('_', 1)
        verb = label_parts[0] if len(label_parts) > 0 else ""
        noun = label_parts[1] if len(label_parts) > 1 else ""
        action_info[row['action_class']] = {
            'verb': verb, 'noun': noun, 'action': row['action_label'],
            'verb_class': verb_class, 'noun_class': noun_class
        }

    # Copy actions.csv
    shutil.copy2(os.path.join(RULSTM_DIR, "actions.csv"),
                 os.path.join(ANNOT_DIR, "actions.csv"))

    for split_name, src_name in [('train', 'training1.csv'), ('validation', 'validation1.csv')]:
        src_path = os.path.join(RULSTM_DIR, src_name)
        df = pd.read_csv(src_path, header=None,
                         names=['uid', 'video_id', 'start_frame', 'stop_frame',
                                'verb_class', 'noun_class', 'action_class'])

        df['participant_id'] = df['video_id'].map(lambda x: x.split('-')[0])
        df['start_time'] = df['start_frame'] / 30.0
        df['stop_time'] = df['stop_frame'] / 30.0
        df['verb'] = df['action_class'].map(lambda ac: action_info.get(ac, {}).get('verb', ''))
        df['noun'] = df['action_class'].map(lambda ac: action_info.get(ac, {}).get('noun', ''))
        df['action'] = df['action_class'].map(lambda ac: action_info.get(ac, {}).get('action', ''))
        df['all_nouns'] = df['noun'].map(lambda x: [x])
        df['all_noun_classes'] = df.apply(lambda row: [row['noun_class']], axis=1)

        out_path = os.path.join(ANNOT_DIR, f"EGTEA_{split_name}_split1.csv")
        df.to_csv(out_path, index=False)
        print(f"  Created {out_path} with {len(df)} entries")


def verify_labels(oct_to_rulstm):
    label_dir = os.path.join(DATA_DIR, "labels")
    verified = 0
    for oct_uid in [0, 6, 10]:
        if oct_uid not in oct_to_rulstm:
            continue
        info = oct_to_rulstm[oct_uid]
        rulstm_uid = info['rulstm_uid']
        expected_sf = info['start_frame']

        fpath = os.path.join(label_dir, f"label_{rulstm_uid}.pkl")
        if os.path.exists(fpath):
            with open(fpath, 'rb') as f:
                label = pickle.load(f)
            actual_sf = int(label['frame_indices'][-1])
            match = actual_sf == expected_sf
            print(f"  Verify: OCT_{oct_uid} -> RULSTM_{rulstm_uid}: "
                  f"fi[-1]={actual_sf}, expected={expected_sf}, {'OK' if match else 'MISMATCH!'}")
            if match:
                verified += 1
    print(f"  Verified {verified} labels")


def main():
    print("=== Preparing EGTEA-Gaze+ data for Diff-IP2D ===\n")

    os.makedirs(DATA_DIR, exist_ok=True)

    print("Step 1: Building OCT-to-RULSTM UID mapping...")
    oct_to_rulstm = build_uid_mapping()
    print(f"  Mapping covers {len(oct_to_rulstm)} entries")

    print("\nStep 2: Extracting and remapping labels...")
    # Clean old labels first (they used wrong UIDs)
    label_dir = os.path.join(DATA_DIR, "labels")
    if os.path.exists(label_dir):
        shutil.rmtree(label_dir)
    extract_and_remap_labels(oct_to_rulstm)

    print("\nStep 3: Verifying remapped labels...")
    verify_labels(oct_to_rulstm)

    print("\nStep 4: Remapping eval labels...")
    remap_eval_labels(oct_to_rulstm)

    print("\nStep 5: Generating video_info.json...")
    generate_video_info()

    print("\nStep 6: Generating uid2future_file_name_egtea.pickle...")
    generate_uid2future(oct_to_rulstm)

    print("\nStep 7: Creating annotation CSVs...")
    create_annotations()

    print("\n=== Done! ===")
    print(f"Data directory: {DATA_DIR}")
    print(f"Annotations: {ANNOT_DIR}")


if __name__ == "__main__":
    main()

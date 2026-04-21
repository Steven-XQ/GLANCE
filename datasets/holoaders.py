# Redesigned by Junyi Ma
# Diff-IP2D: Diffusion-Based Hand-Object Interaction Prediction on Egocentric Videos
# https://github.com/IRMVLab/Diff-IP2D
# We thank OCT (Liu et al.), Diffuseq (Gong et al.), and USST (Bao et al.) for providing the codebases.

import os
import numpy as np
import pickle
from lmdbdict import lmdbdict
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets.dataloaders import EpicDataset, get_datasets
from datasets.ho_utils import load_video_info, process_video_info, process_eval_video_info
import cv2


def match_keypoints(kpsA, kpsB, featuresA, featuresB, ratio=0.7, reprojThresh=4.0):
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
    matches = []

    for m in rawMatches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0]))

    if len(matches) > 4:
        ptsA = np.float32([kpsA[m.queryIdx].pt for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx].pt for m in matches])
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
        matchesMask = status.ravel().tolist()
        return matches, H, matchesMask
    return None

def get_pair_homography(frame_1, frame_2):
    flag = True
    descriptor = cv2.SIFT_create()
    (kpsA, featuresA) = descriptor.detectAndCompute(frame_1, mask=None)
    (kpsB, featuresB) = descriptor.detectAndCompute(frame_2, mask=None)
    matches, matchesMask = None, None
    try:
        (matches, H_BA, matchesMask) = match_keypoints(kpsB, kpsA, featuresB, featuresA)
    except Exception:
        H_BA = np.array([1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0]).reshape(3, 3)
        flag = False

    NoneType = type(None)
    if type(H_BA) == NoneType:
        H_BA = np.array([1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0]).reshape(3, 3)
        flag = False
    try:
        np.linalg.inv(H_BA)
    except Exception:
        H_BA = np.array([1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0]).reshape(3, 3)
        flag = False
    return matches, H_BA, matchesMask, flag

class FeaturesHOLoader(object):
    def __init__(self, sampler, feature_base_path, fps=4.0, input_name='rgb',
                 frame_tmpl='frame_{:010d}.jpg',
                 transform_feat=None, transform_video=None,
                 t_observe=2.5, mode="train", ek_version="ek100",
                 use_gaze=False, gaze_data_base=None,
                 gaze_heatmap_size=32, gaze_sigma=2.0):

        self.feature_base_path = feature_base_path
        self.ek_version = ek_version

        if ek_version == 'egtea':
            if mode == 'train':
                self.lmdb_path = os.path.join("./data/egtea/feats_train", "data.lmdb")
            else:
                self.lmdb_path = os.path.join("./data/egtea/feats_test", "data.lmdb")
            self.env = None
            self.raw_images_base = "/scratch/u6cu/sx2022.u6cu/datasets/EGTEA_Gaze_Plus/EGTEA/extracted_frames"
        else:
            if mode == 'train':
                self.lmdb_path1 = os.path.join("./data/ek100/feats_train", "full_data_with_future_train_part1.lmdb")
                self.lmdb_path2 = os.path.join("./data/ek100/feats_train", "full_data_with_future_train_part2.lmdb")
                self.env1 = None
                self.env2 = None
            else:
                self.lmdb_path = os.path.join("./data/ek100/feats_test", "data.lmdb")
                self.env = None
            self.raw_images_base = "./data/raw_images/EPIC-KITCHENS"

        self.fps = fps
        self.input_name = input_name
        self.frame_tmpl = frame_tmpl
        self.transform_feat = transform_feat
        self.transform_video = transform_video
        self.sampler = sampler
        self.t_observe = t_observe
        self.num_observe = int(t_observe * self.fps)
        self.mode = mode

        if ek_version == 'egtea':
            uid2future_path = "./data/uid2future_file_name_egtea.pickle"
        else:
            uid2future_path = "./data/uid2future_file_name.pickle"
        with open(uid2future_path, 'rb') as f:
            self.uid2future_file_name = pickle.load(f)

        self.use_gaze = use_gaze and (ek_version == 'egtea')
        self.gaze_heatmap_size = gaze_heatmap_size
        self.gaze_sigma = gaze_sigma
        if self.use_gaze:
            self.gaze_data_base = gaze_data_base or "/scratch/u6cu/sx2022.u6cu/datasets/EGTEA_Gaze_Plus/EGTEA/Gaze_Data/gaze_data"
            self._gaze_cache = {}

    def _load_gaze_for_video(self, video_id):
        """Load and cache per-frame gaze data for a video.
        Returns dict: frame_number -> (x_norm, y_norm) averaged over samples.
        """
        if video_id in self._gaze_cache:
            return self._gaze_cache[video_id]

        gaze_file = os.path.join(self.gaze_data_base, f"{video_id}.txt")
        if not os.path.exists(gaze_file):
            self._gaze_cache[video_id] = None
            return None

        frame_gaze = {}
        with open(gaze_file, 'r') as f:
            for line in f:
                if line.startswith('#') or line.startswith('Time'):
                    continue
                parts = line.strip().split('\t')
                if len(parts) < 8 or parts[1] != 'SMP':
                    continue
                try:
                    x_px = float(parts[3])
                    y_px = float(parts[4])
                    frame_num = int(parts[5])
                    event_type = parts[7].strip()
                except (ValueError, IndexError):
                    continue
                if event_type == 'Blink':
                    continue
                x_norm = max(0.0, min(1.0, x_px / 1280.0))
                y_norm = max(0.0, min(1.0, y_px / 960.0))
                if frame_num not in frame_gaze:
                    frame_gaze[frame_num] = []
                frame_gaze[frame_num].append((x_norm, y_norm))

        # Average multiple gaze points per frame
        averaged = {}
        for fnum, points in frame_gaze.items():
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            averaged[fnum] = (sum(xs) / len(xs), sum(ys) / len(ys))

        self._gaze_cache[video_id] = averaged
        return averaged

    def _open_lmdb(self):
        """Lazy LMDB open for multiprocessing safety."""
        from lmdbdict import lmdbdict
        if self.ek_version == 'egtea':
            if self.env is None:
                self.env = lmdbdict(self.lmdb_path, 'r')
        else:
            if self.mode == 'train':
                if self.env1 is None:
                    self.env1 = lmdbdict(self.lmdb_path1, 'r')
                    self.env2 = lmdbdict(self.lmdb_path2, 'r')
            else:
                if self.env is None:
                    self.env = lmdbdict(self.lmdb_path, 'r')

    def _db_get(self, env, key):
        """Get value from lmdbdict using raw txn with pickle protocol 3 (for EK LMDBs)."""
        raw_env = env.env
        with raw_env.begin() as txn:
            raw_val = txn.get(pickle.dumps(key, protocol=3))
            if raw_val is not None:
                return pickle.loads(raw_val)

    def _db_get_native(self, env, key):
        """Get value from lmdbdict using its native __getitem__ (protocol-safe)."""
        try:
            return env[key]
        except KeyError:
            return None

    def _get_feat_dict(self, key_enc):
        if self.ek_version == 'egtea':
            f_dict = self._db_get_native(self.env, key_enc)
            if f_dict is None:
                raise KeyError(f"Key {key_enc} missing from EGTEA LMDB at {self.lmdb_path}")
            return f_dict
        else:
            if self.mode == 'train':
                f_dict = self._db_get(self.env1, key_enc)
                if f_dict is None:
                    f_dict = self._db_get(self.env2, key_enc)
                    if f_dict is None:
                        raise KeyError(f"CRITICAL: Key {key_enc} is genuinely missing from both parts of the database!")
                return f_dict
            else:
                f_dict = self._db_get(self.env, key_enc)
                if f_dict is None:
                    raise KeyError(f"invalid key {key_enc}, check lmdb file in {self.lmdb_path}")
                return f_dict

    def _make_full_name(self, action, f_name):
        """Build the LMDB key / full path name for a frame."""
        if self.ek_version == 'egtea':
            return os.path.join(action.video_id, f_name)
        else:
            return os.path.join(action.participant_id, "rgb_frames", action.video_id, f_name)

    def _get_raw_image_path(self, full_name):
        """Get path to raw image for homography computation."""
        return os.path.join(self.raw_images_base, full_name)

    def __call__(self, action):
        self._open_lmdb()

        times, frames_idxs = self.sampler(action)
        assert self.num_observe <= len(frames_idxs), \
            "num of observation exceed the limit of {}, set smaller t_observe, current is {}".format(len(frames_idxs), self.t_observe)
        frames_names = [self.frame_tmpl.format(i) for i in frames_idxs]
        start_frame_idx = len(frames_idxs) - self.num_observe
        frames_names = frames_names[start_frame_idx:]

        future_file_name_list = []
        if self.mode=='train':
            if action.uid in self.uid2future_file_name:
                future_file_name_list = self.uid2future_file_name[action.uid]
                for f_name in future_file_name_list:
                    f_splt = f_name.split('/')
                    f_name_new = os.path.join(f_splt[-1])
                    frames_names.append(f_name_new)

        full_names = []
        global_feats, global_masks = [], []
        rhand_feats, rhand_masks, rhand_bboxs = [], [], []
        lhand_feats, lhand_masks, lhand_bboxs = [], [], []
        robj_feats, robj_masks, robj_bboxs = [], [], []
        lobj_feats, lobj_masks, lobj_bboxs = [], [], []

        image_list = []
        homography_stack = [np.eye(3)]
        exist_flag=False
        homo_path_train = "./data/homos_train"
        homo_path_test = "./data/homos_test"
        for f_idx, f_name in enumerate(frames_names):
            full_name = self._make_full_name(action, f_name)
            full_names.append(full_name)

            if self.mode=='train':
                file_path = os.path.join(homo_path_train, action.participant_id, "homo", action.video_id, f_name[:-3]+"npy")
                if f_idx == 0 and os.path.exists(file_path):
                    assert os.path.exists(file_path)
                    exist_flag = True
                    homography_stack = np.load(file_path)
                elif not exist_flag:
                    image_cur = cv2.imread(self._get_raw_image_path(full_name))
                    if f_idx > 0 and f_idx < 10 and (not exist_flag):
                        matches, H_BA, matchesMask, flag = get_pair_homography(image_list[f_idx - 1], image_cur)
                        homography_stack.append(np.dot(homography_stack[-1], H_BA))
                    image_list.append(image_cur)
            else:
                file_path = os.path.join(homo_path_test, action.participant_id, "homo", action.video_id, f_name[:-3]+"npy")
                if f_idx == 0 and os.path.exists(file_path):
                    assert os.path.exists(file_path)
                    exist_flag = True
                    homography_stack = np.load(file_path)
                elif not exist_flag:
                    image_cur = cv2.imread(self._get_raw_image_path(full_name))
                    if f_idx > 0 and f_idx < 15 and exist_flag==False and f_idx != 10:
                        matches, H_BA, matchesMask, flag = get_pair_homography(image_list[f_idx - 1], image_cur)
                        homography_stack.append(np.dot(homography_stack[-1], H_BA))
                    image_list.append(image_cur)

            key_enc = full_name.strip().encode('utf-8')
            f_dict = self._get_feat_dict(key_enc)

            global_feat = f_dict['GLOBAL_FEAT']
            global_masks.append(1)
            global_feats.append(global_feat)

            if 'HAND_RIGHT_FEAT' in f_dict:
                rhand_feat = f_dict['HAND_RIGHT_FEAT']
            else:
                rhand_feat = np.zeros_like(global_feat, dtype=np.float32)
            rhand_feats.append(rhand_feat)

            if 'HAND_LEFT_FEAT' in f_dict:
                lhand_feat = f_dict['HAND_LEFT_FEAT']
            else:
                lhand_feat = np.zeros_like(global_feat, dtype=np.float32)
            lhand_feats.append(lhand_feat)

            if 'OBJECT_RIGHT_FEAT' in f_dict:
                robj_feat = f_dict['OBJECT_RIGHT_FEAT']
            else:
                robj_feat = np.zeros_like(global_feat, dtype=np.float32)
            robj_feats.append(robj_feat)

            if 'OBJECT_LEFT_FEAT' in f_dict:
                lobj_feat = f_dict['OBJECT_LEFT_FEAT']
            else:
                lobj_feat = np.zeros_like(global_feat, dtype=np.float32)
            lobj_feats.append(lobj_feat)

            if 'HAND_RIGHT_BBOX' in f_dict:
                rhand_bbox = f_dict['HAND_RIGHT_BBOX']
                rhand_masks.append(1)
            else:
                cx, cy = (0.75, 1.5)
                sx, sy = (0.1, 0.1)
                rhand_bbox = np.array([cx - sx / 2, cy - sy / 2, cx + sx / 2, cy + sy / 2])
                rhand_masks.append(0)
            rhand_bboxs.append(rhand_bbox)

            if 'HAND_LEFT_BBOX' in f_dict:
                lhand_bbox = f_dict['HAND_LEFT_BBOX']
                lhand_masks.append(1)
            else:
                cx, cy = (0.25, 1.5)
                sx, sy = (0.1, 0.1)
                lhand_bbox = np.array([cx - sx / 2, cy - sy / 2, cx + sx / 2, cy + sy / 2])
                lhand_masks.append(0)
            lhand_bboxs.append(lhand_bbox)

            if 'OBJECT_RIGHT_BBOX' in f_dict:
                robj_bbox = f_dict['OBJECT_RIGHT_BBOX']
                robj_masks.append(1)
            else:
                robj_bbox = np.array([0.0, 0.0, 1.0, 1.0])
                robj_masks.append(0)
            robj_bboxs.append(robj_bbox)

            if 'OBJECT_LEFT_BBOX' in f_dict:
                lobj_bbox = f_dict['OBJECT_LEFT_BBOX']
                lobj_masks.append(1)
            else:
                lobj_bbox = np.array([0.0, 0.0, 1.0, 1.0])
                lobj_masks.append(0)
            lobj_bboxs.append(lobj_bbox)


        global_feats = np.stack(global_feats, axis=0)
        rhand_feats = np.stack(rhand_feats, axis=0)
        lhand_feats = np.stack(lhand_feats, axis=0)
        robj_feats = np.stack(robj_feats, axis=0)
        lobj_feats = np.stack(lobj_feats, axis=0)
        feats = np.stack((global_feats, rhand_feats, lhand_feats, robj_feats, lobj_feats), axis=0)

        rhand_bboxs = np.stack(rhand_bboxs, axis=0)
        lhand_bboxs = np.stack(lhand_bboxs, axis=0)
        robj_bboxs = np.stack(robj_bboxs, axis=0)
        lobj_bboxs = np.stack(lobj_bboxs, axis=0)

        bbox_feats = np.stack((rhand_bboxs, lhand_bboxs, robj_bboxs, lobj_bboxs), axis=0)

        global_masks = np.stack(global_masks, axis=0)
        rhand_masks = np.stack(rhand_masks, axis=0)
        lhand_masks = np.stack(lhand_masks, axis=0)
        robj_masks = np.stack(robj_masks, axis=0)
        lobj_masks = np.stack(lobj_masks, axis=0)

        valid_masks = np.stack((global_masks, rhand_masks, lhand_masks, robj_masks, lobj_masks), axis=0)

        if not exist_flag:
            if self.mode=='train':
                homo_folder_path = os.path.join(homo_path_train, action.participant_id, "homo", action.video_id)
            else:
                homo_folder_path = os.path.join(homo_path_test, action.participant_id, "homo", action.video_id)
            if not os.path.exists(homo_folder_path):
                os.makedirs(homo_folder_path)
            homography_stack = np.stack(homography_stack, axis=0)
            homo_file_path = os.path.join(homo_folder_path, frames_names[0][:-4])
            np.save(homo_file_path, homography_stack)
            print("saving homo to ", homo_file_path)

        out = {"name": full_names, "feat": feats, "bbox_feat": bbox_feats, "valid_mask": valid_masks, 'times': times, 'start_time': action.start_time, 'frames_idxs': frames_idxs, 'future_file_name_list': future_file_name_list, 'homography_stack': homography_stack, }

        if self.use_gaze:
            from diffip2d.gaze_modules import generate_gaze_heatmap
            gaze_data = self._load_gaze_for_video(action.video_id)
            gaze_heatmaps = []
            gaze_coords = []  # (T, 3): x, y, valid_flag
            # Generate heatmap for each observation frame
            obs_frame_idxs = frames_idxs[len(frames_idxs) - self.num_observe:]
            for fidx in obs_frame_idxs:
                fidx = int(fidx)
                if gaze_data is not None and fidx in gaze_data:
                    xy = np.array(gaze_data[fidx], dtype=np.float32)
                    heatmap = generate_gaze_heatmap(xy, self.gaze_heatmap_size, self.gaze_sigma)
                    coord = np.array([float(xy[0]), float(xy[1]), 1.0], dtype=np.float32)
                else:
                    heatmap = np.zeros((1, self.gaze_heatmap_size, self.gaze_heatmap_size), dtype=np.float32)
                    coord = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                gaze_heatmaps.append(heatmap)
                gaze_coords.append(coord)
            out['gaze_heatmap'] = np.stack(gaze_heatmaps, axis=0)  # (T, 1, H, W)
            out['gaze_coord'] = np.stack(gaze_coords, axis=0)       # (T, 3)

        return out


class EpicHODataset(EpicDataset):
    def __init__(self, df, partition, ori_fps=60.0, fps=4.0, loader=None, t_ant=None, transform=None,
                 num_actions_prev=None, label_path=None, eval_label_path=None,
                 annot_path=None, rulstm_annot_path=None, ek_version=None, mode='train'):
        super().__init__(df=df, partition=partition, ori_fps=ori_fps, fps=fps,
                         loader=loader, t_ant=t_ant, transform=transform,
                         num_actions_prev=num_actions_prev)
        self.ek_version = ek_version
        self.rulstm_annot_path = rulstm_annot_path
        self.annot_path = annot_path
        self.shape = (456, 256)
        self.discarded_labels, self.discarded_ids = self._get_discarded()
        self.mode = mode

        if 'eval' not in self.partition:
            self.label_dir = os.path.join(label_path, "labels")
        else:
            with open(eval_label_path, 'rb') as f:
                self.eval_labels = pickle.load(f)

    def _load_eval_labels(self, uid):
        video_info = self.eval_labels[uid]
        future_hands, future_valid = process_eval_video_info(video_info, fps=self.fps, t_ant=self.t_ant)
        return future_hands, future_valid

    def _load_labels(self, uid):
        if os.path.exists(os.path.join(self.label_dir, "label_{}.pkl".format(uid))):
            label_valid = True
            video_info = load_video_info(self.label_dir, uid)
            future_hands, contact_point, future_valid, last_frame_index = process_video_info(video_info, fps=self.fps,
                                                                                             t_ant=self.t_ant,
                                                                                             shape=self.shape)
        else:
            label_valid = False
            length = int(self.fps * self.t_ant + 1)
            future_hands = np.zeros((2, length, 2), dtype=np.float32)
            contact_point = np.zeros((2,), dtype=np.float32)
            future_valid = np.array([0, 0], dtype=int)
            last_frame_index = None

        return future_hands, contact_point, future_valid, last_frame_index, label_valid

    def _get_discarded(self):
        discarded_ids = []
        discarded_labels = []
        if 'train' not in self.partition:
            label_type = ['verb', 'noun', 'action']
        else:
            label_type = 'action'
        if 'test' in self.partition:
            challenge = True
        else:
            challenge = False

        for action in self.actions_invalid:
            discarded_ids.append(action.uid)
            if isinstance(label_type, list):
                if challenge:
                    discarded_labels.append(-1)
                else:
                    verb, noun, action_class = action.verb_class, action.noun_class, action.action_class
                    label = np.array([verb, noun, action_class], dtype=int)
                    discarded_labels.append(label)
            else:
                if challenge:
                    discarded_labels.append(-1)
                else:
                    action_class = action.action_class
                    discarded_labels.append(action_class)
        return discarded_labels, discarded_ids

    def __getitem__(self, idx):
        a = self.actions[idx]
        sample = {'uid': a.uid}

        inputs = self.loader(a)
        sample.update(inputs)

        sample['verb'] = a.verb

        if 'eval' not in self.partition:
            future_hands, contact_point, future_valid, last_frame_index, label_valid = self._load_labels(a.uid)

            sample['future_hands'] = future_hands
            sample['contact_point'] = contact_point
            sample['future_valid'] = future_valid
            sample['label_valid'] = label_valid

            if "frames_idxs" in sample and last_frame_index is not None:
                assert last_frame_index == sample["frames_idxs"][-1], \
                    "dataloader video clip {} last observation frame mismatch, " \
                    "index load from history s {} while load from future is{}!".format(a.uid, sample["frames_idxs"][-1], last_frame_index)

        else:
            future_hands, future_valid = self._load_eval_labels(a.uid)
            sample['future_hands'] = future_hands
            sample['future_valid'] = future_valid
            sample['label_valid'] = True

        if 'test' not in self.partition:
            sample['verb_class'] = a.verb_class
            sample['noun_class'] = a.noun_class
            sample['action_class'] = a.action_class
            sample['label'] = np.array([a.verb_class, a.noun_class, a.action_class], dtype=int)
        return sample


def get_dataloaders(args, epic_ds=None, featuresloader=None):
    dss = get_datasets(args, epic_ds=epic_ds, featuresloader=featuresloader)

    def _worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)

    dl_args = {
        'batch_size': args.batch_size,
        'pin_memory': True,
        'num_workers': args.num_workers,
        'drop_last': False,
        'worker_init_fn': _worker_init_fn,
    }
    if args.mode in ['train', 'training']:
        sampler_train = DistributedSampler(dss['train'])
        sampler_validation = DistributedSampler(dss['validation'])
        sampler_eval = DistributedSampler(dss['eval'])
        dls = {
            'train': DataLoader(dss['train'], sampler=sampler_train, **dl_args),
            'validation': DataLoader(dss['validation'], sampler=sampler_validation, shuffle=False, **dl_args),
            'eval': DataLoader(dss['eval'], shuffle=False, sampler=sampler_eval, **dl_args)
        }
    elif args.mode in ['validate', 'validation', 'validating']:
        sampler_validation = DistributedSampler(dss['validation'])
        sampler_eval = DistributedSampler(dss['eval'])
        dls = {
            'validation': DataLoader(dss['validation'], sampler=sampler_validation, shuffle=False, **dl_args),
            'eval': DataLoader(dss['eval'], sampler=sampler_eval, shuffle=False, **dl_args)
        }
    elif args.mode == 'test':
        if args.ek_version == "ek55":
            dls = {
                'test_s1': DataLoader(dss['test_s1'], shuffle=False, **dl_args),
                'test_s2': DataLoader(dss['test_s2'], shuffle=False, **dl_args),
            }
        elif args.ek_version == "ek100":
            dls = {
                'test': DataLoader(dss['test'], shuffle=False, **dl_args),
            }
    else:
        raise Exception(f'Error. Mode "{args.mode}" not supported.')
    return dls

import os
import json


class DatasetArgs(object):
    def __init__(self, ek_version='ek55', mode="train", use_label_only=True,
                 base_path="./", batch_size=32, num_workers=0, modalities=['feat'],
                 fps=None, t_buffer=None):

        self.features_paths = {
            'ek55': os.path.join(base_path, 'data/ek55/feats'),
            'ek100': os.path.join(base_path, 'data/ek100/feats'),
            'egtea': os.path.join(base_path, 'data/egtea/feats')}
        # generated data labels
        self.label_path = {
            'ek55': os.path.join(base_path, 'data/ek55'),
            'ek100': os.path.join(base_path, 'data/ek100'),
            'egtea': os.path.join(base_path, 'data/egtea')}

        # eval labels
        self.eval_label_path = {
            'ek55': os.path.join(base_path, 'data/ek55/ek55_eval_labels.pkl'),
            'ek100': os.path.join(base_path, 'data/ek100/ek100_eval_labels.pkl'),
            'egtea': os.path.join(base_path, 'data/egtea/egtea_eval_labels.pkl')
        }

        self.annot_path = {
            'ek55': os.path.join(base_path, 'common/epic-kitchens-55-annotations'),
            'ek100': os.path.join(base_path, 'common/epic-kitchens-100-annotations'),
            'egtea': os.path.join(base_path, 'common/egtea-annotations')}

        self.rulstm_annot_path = {
            'ek55': os.path.join(base_path, 'common/rulstm/RULSTM/data/ek55'),
            'ek100': os.path.join(base_path, 'common/rulstm/RULSTM/data/ek100'),
            'egtea': os.path.join(base_path, 'common/rulstm/RULSTM/data/egtea')}

        self.pretrained_backbone_path = {
            'ek55': os.path.join(base_path, 'common/rulstm/FEATEXT/models/ek55', 'TSN-rgb.pth.tar'),
            'ek100': os.path.join(base_path, 'common/rulstm/FEATEXT/models/ek100', 'TSN-rgb-ek100.pth.tar'),
        }

        # Dataset-specific parameters
        if ek_version == 'egtea':
            self.fps = fps if fps is not None else 6
            self.t_buffer = t_buffer if t_buffer is not None else 10.0 / 6.0
            self.ori_fps = 30.0
            self.t_ant = 0.5
        else:
            self.fps = fps if fps is not None else 4
            self.t_buffer = t_buffer if t_buffer is not None else 2.5
            self.ori_fps = 60.0
            self.t_ant = 1.0

        self.validation_ratio = 0.2
        self.use_rulstm_splits = True

        # only preprocess uids that have corresponding labels, in "video_info.json"
        self.use_label_only = use_label_only

        self.task = 'anticipation'
        self.num_actions_prev = 1

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.modalities = modalities
        self.ek_version = ek_version
        self.mode = mode

        # Gaze defaults (can be overridden by add_attr)
        self.use_gaze = False
        self.gaze_data_base = None
        self.gaze_heatmap_size = 32
        self.gaze_sigma = 2.0

    def add_attr(self, attr_name, attr_value):
        setattr(self, attr_name, attr_value)

    def has_attr(self, attr_name):
        return hasattr(self, attr_name)

    def __repr__(self):
        return 'Input Args: ' + json.dumps(self.__dict__, indent=4)

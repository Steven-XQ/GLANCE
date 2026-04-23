import os
import json
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import numpy as np


def timestr2sec(t_str):
    hh, mm, ss = [float(x) for x in t_str.split(':')]
    t_sec = hh * 3600.0 + mm * 60.0 + ss
    return t_sec


def read_rulstm_splits(rulstm_annotation_path):
    header = ['uid', 'video_id', 'start_frame', 'stop_frame', 'verb_class', 'noun_class', 'action_class']
    df_train = pd.read_csv(os.path.join(rulstm_annotation_path, 'training.csv'), names=header)
    df_validation = pd.read_csv(os.path.join(rulstm_annotation_path, 'validation.csv'), names=header)
    return df_train, df_validation


def str2list(s, out_type=None):
    """
    Convert a string "[i1, i2, ...]" of items into a list [i1, i2, ...] of items.
    """
    s = s.replace('[', '').replace(']', '')
    s = s.replace('\'', '')
    s = s.split(', ')
    if out_type is not None:
        s = [out_type(ss) for ss in s]
    return s


def split_train_val(df, validation_ratio=0.2, use_rulstm_splits=False,
                    rulstm_annotation_path=None, label_info_path=None,
                    use_label_only=True):
    if label_info_path is not None and use_label_only:
        with open(label_info_path, 'r') as f:
            uids_label = json.load(f)
        df = df.loc[df['uid'].isin(uids_label)]
    if use_rulstm_splits:
        assert rulstm_annotation_path is not None
        df_train_rulstm, df_validation_rulstm = read_rulstm_splits(rulstm_annotation_path)
        uids_train = df_train_rulstm['uid'].values.tolist()
        uids_validation = df_validation_rulstm['uid'].values.tolist()
        df_train = df.loc[df['uid'].isin(uids_train)]
        df_validation = df.loc[df['uid'].isin(uids_validation)]
    else:
        if validation_ratio == 0.0:
            df_train = df
            df_validation = pd.DataFrame(columns=df.columns)
        elif validation_ratio == 1.0:
            df_train = pd.DataFrame(columns=df.columns)
            df_validation = df
        elif 0.0 < validation_ratio < 1.0:
            df_train, df_validation = train_test_split(df, test_size=validation_ratio,
                                                       random_state=3577,
                                                       shuffle=True, stratify=df['participant_id'])
        else:
            raise Exception(f'Error. Validation "{validation_ratio}" not supported.')
    return df_train, df_validation


def create_actions_df(annot_path, rulstm_annot_path, label_path, eval_label_path, ek_version, out_path='actions.csv', use_rulstm_splits=True):
    if use_rulstm_splits:
        if ek_version == 'ek55':
            df_actions = pd.read_csv(os.path.join(rulstm_annot_path['ek55'], 'actions.csv'))
        elif ek_version == 'ek100':
            df_actions = pd.read_csv(os.path.join(rulstm_annot_path['ek100'], 'actions.csv'))
            df_actions['action'] = df_actions.action.map(lambda x: x.replace(' ', '_'))

        df_actions['verb_class'] = df_actions.verb
        df_actions['noun_class'] = df_actions.noun
        df_actions['verb'] = df_actions.action.map(lambda x: x.split('_')[0])
        df_actions['noun'] = df_actions.action.map(lambda x: x.split('_')[1])
        df_actions['action'] = df_actions.action
        df_actions['action_class'] = df_actions.id
        del df_actions['id']

    else:
        if ek_version == 'ek55':
            df_train = get_ek55_annotation(annot_path, rulstm_annot_path, label_path, eval_label_path=None, partition='train',
                                           use_label_only=False, raw=True)
            df_validation = get_ek55_annotation(annot_path, rulstm_annot_path, label_path, eval_label_path, partition='validation',
                                                use_label_only=False, raw=True)
            df = pd.concat([df_train, df_validation])
            df.sort_values(by=['uid'], inplace=True)

        elif ek_version == 'ek100':
            df_train = get_ek100_annotation(annot_path, rulstm_annot_path, label_path, eval_label_path=None, partition='train',
                                            use_label_only=False, raw=True)
            df_validation = get_ek100_annotation(annot_path, rulstm_annot_path, label_path, eval_label_path, partition='validation',
                                                 use_label_only=False, raw=True)
            df = pd.concat([df_train, df_validation])
            df.sort_values(by=['narration_id'], inplace=True)

        noun_classes = df.noun_class.values
        nouns = df.noun.values
        verb_classes = df.verb_class.values
        verbs = df.verb.values

        actions_combinations = [f'{v}_{n}' for v, n in zip(verb_classes, noun_classes)]
        actions = [f'{v}_{n}' for v, n in zip(verbs, nouns)]

        df_actions = {'verb_class': [], 'noun_class': [], 'verb': [], 'noun': [], 'action': []}
        vn_combinations = []
        for i, a in enumerate(actions_combinations):
            if a in vn_combinations:
                continue

            v, n = a.split('_')
            v = int(v)
            n = int(n)
            df_actions['verb_class'] += [v]
            df_actions['noun_class'] += [n]
            df_actions['action'] += [actions[i]]
            df_actions['verb'] += [verbs[i]]
            df_actions['noun'] += [nouns[i]]
            vn_combinations += [a]
        df_actions = pd.DataFrame(df_actions)
        df_actions.sort_values(by=['verb_class', 'noun_class'], inplace=True)
        df_actions['action_class'] = range(len(df_actions))

    df_actions.to_csv(out_path, index=False)
    print(f'Saved file at "{out_path}".')


def get_ek55_annotation(annot_path, rulstm_annot_path, label_path, eval_label_path, partition, validation_ratio=0.2,
                        use_rulstm_splits=False, use_label_only=True, raw=False):
    if partition in ['train', 'validation']:
        csv_path = os.path.join(annot_path['ek55'], 'EPIC_train_action_labels.csv')
        label_info_path = os.path.join(label_path['ek55'], "video_info.json")
        df = pd.read_csv(csv_path)
        df_train, df_validation = split_train_val(df, validation_ratio=validation_ratio,
                                                  use_rulstm_splits=use_rulstm_splits,
                                                  rulstm_annotation_path=rulstm_annot_path['ek55'],
                                                  label_info_path=label_info_path,
                                                  use_label_only=use_label_only)

        df = df_train if partition == 'train' else df_validation
        if not use_rulstm_splits:
            df.sort_values(by=['uid'], inplace=True)

    elif partition in ['eval', 'evaluation']:
        csv_path = os.path.join(annot_path['ek55'], 'EPIC_train_action_labels.csv')
        df = pd.read_csv(csv_path)
        with open(eval_label_path['ek55'], 'rb') as f:
            eval_labels = pickle.load(f)
        eval_uids = eval_labels.keys()
        df = df.loc[df['uid'].isin(eval_uids)]

    elif partition == 'test_s1':
        csv_path = os.path.join(annot_path['ek55'], 'EPIC_test_s1_timestamps.csv')
        df = pd.read_csv(csv_path)

    elif partition == 'test_s2':
        csv_path = os.path.join(annot_path['ek55'], 'EPIC_test_s2_timestamps.csv')
        df = pd.read_csv(csv_path)
    else:
        raise Exception(f'Error. Partition "{partition}" not supported.')

    if raw:
        return df

    actions_df_path = os.path.join(annot_path['ek55'], 'actions.csv')
    if not os.path.exists(actions_df_path):
        create_actions_df(annot_path, rulstm_annot_path, label_path, eval_label_path, 'ek55', out_path=actions_df_path, use_rulstm_splits=True)
    df_actions = pd.read_csv(actions_df_path)

    df['start_time'] = df['start_timestamp'].map(lambda t: timestr2sec(t))
    df['stop_time'] = df['stop_timestamp'].map(lambda t: timestr2sec(t))
    if 'test' not in partition:
        action_classes = []
        actions = []
        for _, row in df.iterrows():
            v, n = row.verb_class, row.noun_class
            df_a_sub = df_actions[(df_actions['verb_class'] == v) & (df_actions['noun_class'] == n)]
            a_cl = df_a_sub['action_class'].values
            a = df_a_sub['action'].values
            if len(a_cl) > 1:
                print(a_cl)
            action_classes += [a_cl[0]]
            actions += [a[0]]
        df['action_class'] = action_classes
        df['action'] = actions
        df['all_nouns'] = df['all_nouns'].map(lambda x: str2list(x))
        df['all_noun_classes'] = df['all_noun_classes'].map(lambda x: str2list(x, out_type=int))

    return df


def get_ek100_annotation(annot_path, rulstm_annot_path, label_path, eval_label_path, partition, validation_ratio=0.2,
                         use_rulstm_splits=False, use_label_only=True, raw=False):
    
    if partition in 'train':
        df = pd.read_csv(os.path.join(annot_path['ek100'], 'EPIC_100_train.csv'))
        uids = np.arange(len(df))

    elif partition in 'validation':
        df_train = pd.read_csv(os.path.join(annot_path['ek100'], 'EPIC_100_train.csv'))
        df = pd.read_csv(os.path.join(annot_path['ek100'], 'EPIC_100_validation.csv'))
        uids = np.arange(len(df)) + len(df_train)
 
    elif partition in 'evaluation':
        df_train = pd.read_csv(os.path.join(annot_path['ek100'], 'EPIC_100_train.csv'))
        df = pd.read_csv(os.path.join(annot_path['ek100'], 'EPIC_100_validation.csv'))
        uids = np.arange(len(df)) + len(df_train)
        df['uid'] = uids
        with open(eval_label_path['ek100'], 'rb') as f:
            eval_labels = pickle.load(f)
        eval_uids = eval_labels.keys()
        df = df.loc[df['uid'].isin(eval_uids)]

    elif partition == 'test':
        df_train = pd.read_csv(os.path.join(annot_path['ek100'], 'EPIC_100_train.csv'))
        df_validation = pd.read_csv(os.path.join(annot_path['ek100'], 'EPIC_100_validation.csv'))
        df = pd.read_csv(os.path.join(annot_path['ek100'], 'EPIC_100_test_timestamps.csv'))
        uids = np.arange(len(df)) + len(df_train) + len(df_validation) 

    else:
        raise Exception(f'Error. Partition "{partition}" not supported.')
    if raw:
        return df

    actions_df_path = os.path.join(annot_path['ek100'], 'actions.csv')
    if not os.path.exists(actions_df_path):
        create_actions_df(annot_path, rulstm_annot_path, label_path, eval_label_path, 'ek100', actions_df_path)
    df_actions = pd.read_csv(actions_df_path)

    df['start_time'] = df['start_timestamp'].map(lambda t: timestr2sec(t))
    df['stop_time'] = df['stop_timestamp'].map(lambda t: timestr2sec(t))
    if not 'uid' in df:
        df['uid'] = uids

    if use_label_only:
        label_info_path = os.path.join(label_path['ek100'], "video_info.json")
        with open(label_info_path, 'r') as f:
            uids_label = json.load(f)
            df = df.loc[df['uid'].isin(uids_label)]

    if 'test' not in partition:
        action_classes = []
        actions = []
        for _, row in df.iterrows():
            v, n = row.verb_class, row.noun_class
            df_a_sub = df_actions[(df_actions['verb_class'] == v) & (df_actions['noun_class'] == n)]
            a_cl = df_a_sub['action_class'].values
            a = df_a_sub['action'].values
            if len(a_cl) > 1:
                print(a_cl)
            action_classes += [a_cl[0]]
            actions += [a[0]]
        df['action_class'] = action_classes
        df['action'] = actions
        df['all_nouns'] = df['all_nouns'].map(lambda x: str2list(x))
        df['all_noun_classes'] = df['all_noun_classes'].map(lambda x: str2list(x, out_type=int))
    return df


def get_meccano_annotation(annot_path, rulstm_annot_path, label_path, eval_label_path, partition, validation_ratio=0.2,
                            use_rulstm_splits=False, use_label_only=True, raw=False):
    """Load MECCANO annotations for the specified partition.

    partition options: 'train', 'validation', 'eval' (test split filtered by eval_labels), 'test'.
    """
    if partition == 'train':
        df = pd.read_csv(os.path.join(annot_path['meccano'], 'MECCANO_train_split.csv'))
    elif partition in ('validation', 'val'):
        df = pd.read_csv(os.path.join(annot_path['meccano'], 'MECCANO_val_split.csv'))
    elif partition in ('eval', 'evaluation', 'test'):
        df = pd.read_csv(os.path.join(annot_path['meccano'], 'MECCANO_test_split.csv'))
    else:
        raise Exception(f'Error. Partition "{partition}" not supported for MECCANO.')

    if raw:
        return df

    if partition in ('eval', 'evaluation'):
        with open(eval_label_path['meccano'], 'rb') as f:
            eval_labels = pickle.load(f)
        eval_uids = eval_labels.keys()
        df = df.loc[df['uid'].isin(eval_uids)]

    if use_label_only:
        label_info_path = os.path.join(label_path['meccano'], "video_info.json")
        with open(label_info_path, 'r') as f:
            uids_label = json.load(f)
        df = df.loc[df['uid'].isin(uids_label)]

    # Force video_id / participant_id to 4-digit strings (pandas parses "0001" -> int 1)
    df['video_id'] = df['video_id'].map(lambda v: f"{int(v):04d}")
    df['participant_id'] = df['participant_id'].map(lambda v: f"{int(v):04d}")

    # all_nouns / all_noun_classes stored as string repr of lists in CSV
    if 'all_nouns' in df.columns:
        df['all_nouns'] = df['all_nouns'].map(lambda x: str2list(x) if isinstance(x, str) else x)
        df['all_noun_classes'] = df['all_noun_classes'].map(lambda x: str2list(x, out_type=int) if isinstance(x, str) else x)

    return df


def get_egtea_annotation(annot_path, rulstm_annot_path, label_path, eval_label_path, partition, validation_ratio=0.2,
                          use_rulstm_splits=False, use_label_only=True, raw=False):
    """Load EGTEA-Gaze+ annotations for the specified partition."""

    if partition == 'train':
        df = pd.read_csv(os.path.join(annot_path['egtea'], 'EGTEA_train_split1.csv'))
    elif partition in ('validation', 'eval', 'evaluation'):
        df = pd.read_csv(os.path.join(annot_path['egtea'], 'EGTEA_validation_split1.csv'))
    else:
        raise Exception(f'Error. Partition "{partition}" not supported for EGTEA.')

    if raw:
        return df

    # For eval partition, filter to UIDs that have eval labels
    if partition in ('eval', 'evaluation'):
        with open(eval_label_path['egtea'], 'rb') as f:
            eval_labels = pickle.load(f)
        eval_uids = eval_labels.keys()
        df = df.loc[df['uid'].isin(eval_uids)]

    # Filter to UIDs with labels if requested
    if use_label_only:
        label_info_path = os.path.join(label_path['egtea'], "video_info.json")
        with open(label_info_path, 'r') as f:
            uids_label = json.load(f)
        df = df.loc[df['uid'].isin(uids_label)]

    # all_nouns and all_noun_classes are stored as string repr of lists in CSV
    if 'all_nouns' in df.columns:
        df['all_nouns'] = df['all_nouns'].map(lambda x: str2list(x) if isinstance(x, str) else x)
        df['all_noun_classes'] = df['all_noun_classes'].map(lambda x: str2list(x, out_type=int) if isinstance(x, str) else x)

    return df
# Evaluate affordance metrics (SIM, AUC-J, NSS) for GLANCE (v8) on EGTEA-Gaze+.
# Expects ./diffip_weights/checkpoint_35.pth.tar from run_train.py.

import sys
import os
sys.path.append('.')

if __name__ == '__main__':

    COMMANDLINE = (
        "python -m torch.distributed.launch "
        "--nproc_per_node=8 --master_port=12327 --use_env "
        "traineval.py "
        "--evaluate "
        "--ek_version=egtea --num_classes=106 "
        "--seq_len_obs=10 --seq_len_unobs=3 "
        "--resume=./diffip_weights/checkpoint_35.pth.tar "
        "--manual_seed=42 "
        "--use_gaze --gaze_bias_init_delta=3 --gaze_bias_init_amp=0.5"
    )

    print(COMMANDLINE)
    os.system(COMMANDLINE)

# Train GLANCE (v8) on EGTEA-Gaze+ at E = 35 epochs with seed = 42.
# GLANCE = full gaze pathway with Gaussian-initialised eye-hand-latency bias
# (delta = 3, A = 0.5).

import sys
import os
sys.path.append('.')

if __name__ == '__main__':

    COMMANDLINE = (
        "python -m torch.distributed.launch "
        "--nproc_per_node=8 --master_port=12325 --use_env "
        "traineval.py "
        "--ek_version=egtea --epochs=35 --batch_size=8 "
        "--num_classes=106 --seq_len_obs=10 --seq_len_unobs=3 "
        "--learnable_weight=True --manual_seed=42 "
        "--use_gaze --gaze_bias_init_delta=3 --gaze_bias_init_amp=0.5"
    )

    print(COMMANDLINE)
    os.system(COMMANDLINE)

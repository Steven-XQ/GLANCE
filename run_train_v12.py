import sys
import os
sys.path.append('.')

if __name__ == '__main__':
    COMMANDLINE = f"python traineval.py --ek_version=egtea --epochs=30 --batch_size=32 --num_classes=106 --seq_len_obs=10 --seq_len_unobs=3 --learnable_weight=True --use_gaze --gaze_bias_init_delta=3 --gaze_bias_init_amp=0.5"
    print(COMMANDLINE)
    os.system(COMMANDLINE)

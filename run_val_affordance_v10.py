import sys
import os
sys.path.append('.')

if __name__ == '__main__':
    COMMANDLINE = f"python traineval.py --evaluate --ek_version=egtea --num_classes=106 --seq_len_obs=10 --seq_len_unobs=3 --resume=./diffip_weights/checkpoint_30.pth.tar --use_gaze --gaze_fixed_delta=2"
    print(COMMANDLINE)
    os.system(COMMANDLINE)

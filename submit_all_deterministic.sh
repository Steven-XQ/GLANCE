#!/bin/bash
#SBATCH --job-name=det_all
#SBATCH --output=logs/det_all_%j.out
#SBATCH --error=logs/det_all_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

source $SCRATCH/miniconda3/etc/profile.d/conda.sh
conda activate diffip

cd $HOME/Diff-IP2D
mkdir -p logs

TRAIN_BASE="--ek_version=egtea --epochs=30 --batch_size=32 --num_classes=106 --seq_len_obs=10 --seq_len_unobs=3 --learnable_weight=True"
EVAL_TRAJ="--evaluate --ek_version=egtea --num_classes=106 --seq_len_obs=10 --seq_len_unobs=3 --resume=./diffip_weights/checkpoint_30.pth.tar --traj_only"
EVAL_AFF="--evaluate --ek_version=egtea --num_classes=106 --seq_len_obs=10 --seq_len_unobs=3 --resume=./diffip_weights/checkpoint_30.pth.tar"

run_exp() {
    local name="$1"
    local flags="$2"
    echo "=========================================="
    echo "=== $name ==="
    echo "=========================================="
    echo "=== $name TRAINING ==="
    TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=1 --master_port=12225 --use_env run_experiment.py $TRAIN_BASE $flags
    echo "=== $name EVAL TRAJ ==="
    TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=1 --master_port=12226 --use_env run_experiment.py $EVAL_TRAJ $flags
    echo "=== $name EVAL AFF ==="
    TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=1 --master_port=12227 --use_env run_experiment.py $EVAL_AFF $flags
    echo "=== $name DONE ==="
}

run_exp "BASELINE" ""
run_exp "v2" "--use_gaze"
run_exp "v4" "--use_gaze --gaze_coord_only"
run_exp "v6" "--use_gaze --gaze_fusion_only"
run_exp "v7" "--use_gaze --gaze_alpha_clamp=0.1"
run_exp "v8" "--use_gaze --gaze_last_n_blocks=2"
run_exp "v9" "--use_gaze --gaze_detach_diffusion"
run_exp "v10" "--use_gaze --gaze_fixed_delta=2"
run_exp "v11" "--use_gaze --gaze_bias_init_delta=2 --gaze_bias_init_amp=2.0"
run_exp "v12" "--use_gaze --gaze_bias_init_delta=3 --gaze_bias_init_amp=0.5"

echo "=========================================="
echo "=== ALL EXPERIMENTS COMPLETE           ==="
echo "=========================================="

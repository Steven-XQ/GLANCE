#!/bin/bash
#SBATCH --job-name=v13_base_30
#SBATCH --output=logs/v13_base_30_%j.out
#SBATCH --error=logs/v13_base_30_%j.err
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

run_exp() {
    local name="$1"
    local epochs="$2"
    local flags="$3"
    local TRAIN_BASE="--ek_version=egtea --epochs=$epochs --batch_size=32 --num_classes=106 --seq_len_obs=10 --seq_len_unobs=3 --learnable_weight=True"
    local EVAL_RESUME="--evaluate --ek_version=egtea --num_classes=106 --seq_len_obs=10 --seq_len_unobs=3 --resume=./diffip_weights/checkpoint_${epochs}.pth.tar"
    echo "=========================================="
    echo "=== $name (${epochs}ep) ==="
    echo "=========================================="
    echo "=== $name TRAINING ==="
    TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=1 --master_port=12225 --use_env run_experiment.py $TRAIN_BASE $flags
    echo "=== $name EVAL TRAJ ==="
    TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=1 --master_port=12226 --use_env run_experiment.py $EVAL_RESUME --traj_only $flags
    echo "=== $name EVAL AFF ==="
    TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=1 --master_port=12227 --use_env run_experiment.py $EVAL_RESUME $flags
    echo "=== $name DONE ==="
}

run_exp "v13" 30 "--use_gaze --gaze_bias_init_delta=3 --gaze_bias_init_amp=0.5 --gaze_before_motion --gaze_detach_diffusion"
run_exp "BASELINE" 30 ""

echo "=========================================="
echo "=== ALL EXPERIMENTS COMPLETE           ==="
echo "=========================================="

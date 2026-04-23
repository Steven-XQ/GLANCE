#!/bin/bash
#SBATCH --job-name=meccano_gaze
#SBATCH --output=logs/meccano_gaze_%j.out
#SBATCH --error=logs/meccano_gaze_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

source $SCRATCH/miniconda3/etc/profile.d/conda.sh
conda activate diffip

cd $HOME/Diff-IP2D
mkdir -p logs

run_exp() {
    local name="$1"
    local epochs="$2"
    local flags="$3"
    local TRAIN_BASE="--ek_version=meccano --epochs=$epochs --batch_size=32 --num_classes=61 --seq_len_obs=10 --seq_len_unobs=3 --learnable_weight=True"
    local EVAL_RESUME="--evaluate --ek_version=meccano --num_classes=61 --seq_len_obs=10 --seq_len_unobs=3 --resume=./diffip_weights/checkpoint_${epochs}.pth.tar"
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

V12="--use_gaze --gaze_bias_init_delta=3 --gaze_bias_init_amp=0.5"

EP=${EPOCHS:-30}

run_exp "v2"   $EP "--use_gaze"
run_exp "v4"   $EP "--use_gaze --gaze_coord_only"
run_exp "v6"   $EP "--use_gaze --gaze_fusion_only"
run_exp "v8"   $EP "--use_gaze --gaze_last_n_blocks=2"
run_exp "v9"   $EP "--use_gaze --gaze_detach_diffusion"
run_exp "v10"  $EP "--use_gaze --gaze_fixed_delta=2"
run_exp "v11"  $EP "--use_gaze --gaze_bias_init_delta=2 --gaze_bias_init_amp=2.0"
run_exp "v12"  $EP "$V12"
run_exp "v12a" $EP "$V12 --gaze_heatmap_only"
run_exp "v12b" $EP "$V12 --gaze_cfg_dropout=0.1"
run_exp "v12c" $EP "$V12 --gaze_before_motion"
run_exp "v13"  $EP "$V12 --gaze_before_motion --gaze_detach_diffusion"

echo "=========================================="
echo "=== ALL MECCANO GAZE VARIANTS COMPLETE  ==="
echo "=========================================="

#!/bin/bash
#SBATCH--job-name=sdpo_qwen_trial
#SBATCH--partition=fengl2 #如需使⽤gpuB或gpu_v100,直接替换即可
#SBATCH--nodes=1
#SBATCH--ntasks-per-node=1
#SBATCH--cpus-per-task=24
#SBATCH --time=06:00:00
#SBATCH--gres=gpu:1 #使⽤显卡的数量
#SBATCH-o./logs/%J.out #输出⽇志
#SBATCH-e./logs/%J.err #报错⽇志
#SBATCH--mail-type=BEGIN,END,FAIL
#SBATCH--mail-user=213231600@seu.edu.cn

#================================= 环境配置  ===========================
set -euo pipefail

PROJECT_ROOT="/seu_share2/home/fenglei/213231600/SDPO"
cd "$PROJECT_ROOT"

mkdir -p logs
mkdir -p checkpoints

module load anaconda3
source activate sdpo


# HuggingFace / Transformers
export HF_HOME="/seu_share2/home/fenglei/213231600/hf"
export HF_DATASETS_CACHE="/seu_share2/home/fenglei/213231600/hf_datasets"
export HF_TOKEN="hf_token_here"
export HF_HUB_ENABLE_HF_TRANSFER=1

# XDG cache（tokenizers / rust / sentencepiece 等）
export XDG_CACHE_HOME=/seu_share2/home/fenglei/213231600/xdg_cache

# Ray 临时目录
export RAY_TMPDIR=/seu_share2/home/fenglei/213231600/ray_tmp

# wandb
export WANDB_DIR=/seu_share2/home/fenglei/213231600/wandb
export WANDB_MODE=online

# triton
export TRITON_CACHE_DIR="/seu_share2/home/fenglei/213231600/triton_cache"

mkdir -p "$RAY_TMPDIR" "$HF_HOME" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR"

export USER="${USER:-$(whoami)}"


#================================= 训练配置 ===========================
DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    shift
fi

SUFFIX="${1:-math_sdpo}"

CONFIG_NAME="sdpo"
DATA_PATH="datasets/math"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-4B}"

TRAIN_FILE="${TRAIN_FILE:-${PROJECT_ROOT}/datasets/math/train.parquet}"
VAL_FILE="${VAL_FILE:-${PROJECT_ROOT}/datasets/math/test.parquet}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
LR="${LR:-1e-6}"
DONTS_REPROMPT_ON_SELF_SUCCESS="${DONTS_REPROMPT_ON_SELF_SUCCESS:-True}"
ALPHA="${ALPHA:-0.5}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
VAL_SAMPLES="${VAL_SAMPLES:-16}"
SAVE_FREQ="${SAVE_FREQ:-234}"
TEST_FREQ="${TEST_FREQ:--1}"

TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-3072}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.6}"

export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

# Slurm 下不要手动覆盖 CUDA_VISIBLE_DEVICES，直接用调度器分到的卡
if [[ -n "${NUM_GPUS:-}" ]]; then
    N_GPUS_PER_NODE="${NUM_GPUS}"
elif [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
    N_GPUS_PER_NODE="${SLURM_GPUS_ON_NODE}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    N_GPUS_PER_NODE="$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")"
else
    N_GPUS_PER_NODE=1
fi
export N_GPUS_PER_NODE

MODEL_NAME="$(echo "${MODEL_PATH}" | tr '/' '-')"
EXP_NAME="SDPO-math-E${TRAIN_BATCH_SIZE}-alpha${ALPHA}-lr${LR}-dross${DONTS_REPROMPT_ON_SELF_SUCCESS}-${MODEL_NAME}-${SUFFIX}"
DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-${PROJECT_ROOT}/checkpoints/${EXP_NAME}}"

HYDRA_ARGS=(
    "data.train_files=['${TRAIN_FILE}']"
    "data.val_files=['${VAL_FILE}']"
    "data.train_batch_size=${TRAIN_BATCH_SIZE}"
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
    "data.max_response_length=${MAX_RESPONSE_LENGTH}"
    "trainer.group_name=SDPO-slurm-math"
    "trainer.nnodes=1"
    "trainer.n_gpus_per_node=${N_GPUS_PER_NODE}"
    "trainer.total_epochs=${TOTAL_EPOCHS}"
    "trainer.save_freq=${SAVE_FREQ}"
    "trainer.test_freq=${TEST_FREQ}"
    "trainer.default_local_dir=${DEFAULT_LOCAL_DIR}"
    "actor_rollout_ref.rollout.n=${ROLLOUT_BATCH_SIZE}"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
    "actor_rollout_ref.model.path=${MODEL_PATH}"
    "actor_rollout_ref.actor.optim.lr=${LR}"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
    "actor_rollout_ref.actor.self_distillation.distillation_topk=100"
    "algorithm.rollout_correction.rollout_is=token"
    "actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=${DONTS_REPROMPT_ON_SELF_SUCCESS}"
    "actor_rollout_ref.actor.self_distillation.alpha=${ALPHA}"
    "actor_rollout_ref.actor.self_distillation.include_environment_feedback=False"
    "actor_rollout_ref.actor.optim.lr_warmup_steps=10"
    "actor_rollout_ref.rollout.val_kwargs.n=${VAL_SAMPLES}"
    "custom_reward_function.path=${PROJECT_ROOT}/verl/utils/reward_score/feedback/__init__.py"
)

CMD=(
    bash
    "${PROJECT_ROOT}/training/verl_training.sh"
    "${EXP_NAME}"
    "${CONFIG_NAME}"
    "${DATA_PATH}"
    "${HYDRA_ARGS[@]}"
)

echo "----------------------------------------------------------------"
echo "Experiment: ${EXP_NAME}"
echo "Model: ${MODEL_PATH}"
echo "Train file: ${TRAIN_FILE}"
echo "Val file: ${VAL_FILE}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<slurm-managed>}"
echo "trainer.n_gpus_per_node: ${N_GPUS_PER_NODE}"
echo "trainer.total_epochs: ${TOTAL_EPOCHS}"
echo "trainer.save_freq: ${SAVE_FREQ}"
echo "trainer.test_freq: ${TEST_FREQ}"
echo "trainer.default_local_dir: ${DEFAULT_LOCAL_DIR}"
echo "WANDB_MODE: ${WANDB_MODE}"
echo "----------------------------------------------------------------"

if [[ "${DRY_RUN}" == "true" ]]; then
    printf '%q ' "${CMD[@]}"
    printf '\n'
else
    "${CMD[@]}"
fi

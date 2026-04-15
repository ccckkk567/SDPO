#!/bin/bash
set -euo pipefail

# 用法：
#   bash run_local_sdpo_math.sh [--dry-run] [experiment_name_suffix]
#
# 示例：
#   1) 远端 8 卡都可用时：
#      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_local_sdpo_math_qwen35_4b.sh
#
#   2) 如果当前只想先用 2 号卡做单卡试跑：
#      CUDA_VISIBLE_DEVICES=2 NUM_GPUS=1 bash run_local_sdpo_math_qwen35_4b.sh debug
#
# 说明：
#   - 这个脚本不依赖 Slurm，直接在当前机器启动训练。
#   - 不自动探测远端 GPU 占用情况，使用哪几张卡由 CUDA_VISIBLE_DEVICES 决定。
#   - 如果你的实际模型名不是 Qwen/Qwen3.5-4B，可在运行时覆盖 MODEL_PATH。

#================================= 训练配置 ===========================
DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    shift
fi

SUFFIX="${1:-math_sdpo}"

# =============================================================================
# 固定实验配置
# =============================================================================

CONFIG_NAME="sdpo"
DATA_PATH="datasets/math"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-4B}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
LR="${LR:-1e-6}"
DONTS_REPROMPT_ON_SELF_SUCCESS="${DONTS_REPROMPT_ON_SELF_SUCCESS:-True}"
ALPHA="${ALPHA:-0.5}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
VAL_SAMPLES="${VAL_SAMPLES:-16}"
SAVE_FREQ="${SAVE_FREQ:-234}"

TOTAL_EPOCHS="${TOTAL_EPOCHS:-10}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-3072}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.6}"

# 可选，先别默认打开；如果后面显存/吞吐不理想再试
# USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-False}"

# =============================================================================
# 环境设置
# =============================================================================

export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export USER="${USER:-$(whoami)}"

# 让 Ray 保留我们手动设置的 CUDA_VISIBLE_DEVICES。
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

# 默认按整机 8 卡来跑；如果你只想用一部分卡，请在命令前自行覆盖。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# 优先使用外部传入的 NUM_GPUS；否则根据 CUDA_VISIBLE_DEVICES 自动计数。
if [[ -n "${NUM_GPUS:-}" ]]; then
    N_GPUS_PER_NODE="${NUM_GPUS}"
else
    N_GPUS_PER_NODE="$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")"
fi
export N_GPUS_PER_NODE

if [[ "${N_GPUS_PER_NODE}" -lt 1 ]]; then
    echo "Error: N_GPUS_PER_NODE must be >= 1"
    exit 1
fi

# =============================================================================
# 构造实验名与 Hydra 覆盖参数
# =============================================================================

MODEL_NAME="$(echo "${MODEL_PATH}" | tr '/' '-')"
EXP_NAME="SDPO-math-E${TRAIN_BATCH_SIZE}-alpha${ALPHA}-lr${LR}-dross${DONTS_REPROMPT_ON_SELF_SUCCESS}-${MODEL_NAME}-${SUFFIX}"
DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-${PROJECT_ROOT}/checkpoints/${EXP_NAME}}"

HYDRA_ARGS=(
    "data.train_batch_size=${TRAIN_BATCH_SIZE}"
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
    "data.max_response_length=${MAX_RESPONSE_LENGTH}"
    "trainer.group_name=SDPO-local-math"
    "trainer.nnodes=1"
    "trainer.n_gpus_per_node=${N_GPUS_PER_NODE}"
    "trainer.total_epochs=${TOTAL_EPOCHS}"
    "trainer.save_freq=${SAVE_FREQ}"
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
echo "Starting Local SDPO Training"
echo "Experiment: ${EXP_NAME}"
echo "Data: ${DATA_PATH}"
echo "Model: ${MODEL_PATH}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "trainer.n_gpus_per_node: ${N_GPUS_PER_NODE}"
echo "trainer.total_epochs: ${TOTAL_EPOCHS}"
echo "trainer.save_freq: ${SAVE_FREQ}"
echo "trainer.default_local_dir: ${DEFAULT_LOCAL_DIR}"
echo "data.max_prompt_length: ${MAX_PROMPT_LENGTH}"
echo "data.max_response_length: ${MAX_RESPONSE_LENGTH}"
echo "actor_rollout_ref.rollout.gpu_memory_utilization: ${GPU_MEMORY_UTILIZATION}"
echo "----------------------------------------------------------------"

if [[ "${DRY_RUN}" == "true" ]]; then
    printf '%q ' "${CMD[@]}"
    printf '\n'
else
    "${CMD[@]}"
fi

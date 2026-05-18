#!/bin/bash
#SBATCH --account=g34
#SBATCH --time=03:30:00
#SBATCH --job-name=gipfel-final_13b_8n_tp1_pp8_dp4_fp8_mbs1_gbs512_vpp1
#SBATCH --output=logs/train/%x-%j.out
#SBATCH --error=logs/train/%x-%j.err
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=850000
#SBATCH --no-requeue

echo "START TIME: $(date)"

################ Configs ################
WORKDIR=/users/$USER/gipfelsturm
MEGATRON_LM_DIR=$WORKDIR/Megatron-LM
DATA_PREFIX=/iopsstor/scratch/cscs/$USER/dataset/climbmix_small/climbmix_small
DATASET_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/cache

# Training config
MBS=1
GBS=512
SEQ_LEN=4096
TRAINING_STEPS=300

# Logging
PROJECT_NAME=gipfelsturm
EXP_NAME=final_13b_8n_tp1_pp8_dp4_fp8_mbs1_gbs512_vpp1-${SLURM_NNODES}n
LOG_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/$PROJECT_NAME/$EXP_NAME
TENSORBOARD_DIR=$LOG_DIR/tensorboard
#########################################

mkdir -p $WORKDIR/logs/train $LOG_DIR $TENSORBOARD_DIR

# Required for TP; must be unset with FSDP (conflicts with FSDP's async comms)
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / SLURM_GPUS_PER_NODE ))
# Try to reduce our memory footprint a bit: https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-guide.html#techniques-for-reducing-memory-to-avoid-memory-overflow-and-enhance-training-efficiency
# export TORCH_NCCL_AVOID_RECORD_STREAMS=1
# export NCCL_NVLS_ENABLE=0
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))

TRANSFORMER_ENGINE_ARGS=(
    --transformer-impl transformer_engine
)

NETWORK_SIZE_ARGS=(
    --num-layers 40
    --hidden-size 5120
    --ffn-hidden-size 17920
    --num-attention-heads 40
    --group-query-attention
    --num-query-groups 8
    --max-position-embeddings $SEQ_LEN
    --position-embedding-type rope
    --normalization RMSNorm
    --swiglu
    --untie-embeddings-and-output-weights
    --seq-length $SEQ_LEN
)

TRAINING_ARGS=(
    --micro-batch-size $MBS
    --global-batch-size $GBS
    --train-iters $TRAINING_STEPS
    --log-interval 1
    --eval-interval 25
    --eval-iters 5
    --cross-entropy-loss-fusion
    --optimizer adam
    --dataloader-type single
    --manual-gc
)

REGULARIZATION_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --weight-decay 0.1
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-beta2 0.95
)

LEARNING_RATE_ARGS=(
    --lr 3e-4
    --min-lr 3e-5
    --lr-decay-style cosine
    --lr-warmup-iters 50
)

INITIALIZATION_ARGS=(
    --seed 42
    --init-method-std 0.02
)

MIXED_PRECISION_ARGS=(
    --bf16
    --fp8-format hybrid
    --fp8-recipe delayed
    --fp8-amax-history-len 1024
    --fp8-amax-compute-algo max
    --fp8-margin 0
    --fp8-param-gather
)

DISTRIBUTED_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 8
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
    --num-layers-per-virtual-pipeline-stage 1
    --data-parallel-sharding-strategy no_shard
)

EXTRA_ARGS=(
    --cuda-graph-impl transformer_engine
    --te-rng-tracker
    --cuda-graph-scope attn mlp
)

LOGGING_ARGS=(
    --log-throughput
    --log-progress
    --tensorboard-dir $TENSORBOARD_DIR
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard
    --timing-log-level 1
    --timing-log-option minmax
)

TOKENIZER_ARGS=(
    --tokenizer-type GPT2BPETokenizer
    --vocab-file $WORKDIR/data/gpt2-vocab.json
    --merge-file $WORKDIR/data/gpt2-merges.txt
)

DATA_ARGS=(
    --data-path $DATA_PREFIX
    --data-cache-path $DATASET_CACHE_DIR
    --split 99,1,0
    --num-workers 2
)

SCRIPT_ARGS="$MEGATRON_LM_DIR/pretrain_gpt.py \
    ${TRANSFORMER_ENGINE_ARGS[@]} \
    ${NETWORK_SIZE_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${REGULARIZATION_ARGS[@]} \
    ${LEARNING_RATE_ARGS[@]} \
    ${INITIALIZATION_ARGS[@]} \
    ${MIXED_PRECISION_ARGS[@]} \
    ${DISTRIBUTED_ARGS[@]} \
    ${EXTRA_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    ${DATA_ARGS[@]}"

# WANDB
if [ -n "$WANDB_API_KEY" ]; then
    echo "[$(date)] WANDB enabled."
    SCRIPT_ARGS="$SCRIPT_ARGS \
        --wandb-save-dir $LOG_DIR \
        --wandb-project $PROJECT_NAME \
        --wandb-exp-name $EXP_NAME-$SLURM_JOB_ID"
else
    export WANDB_MODE=disabled
    echo "[$(date)] WANDB disabled."
fi

# Build TRAINING_CMD as a plain string so it survives bash -c expansion safely.
# Bash arrays cannot be exported into srun's bash -c subshell.
TRAINING_CMD="python -m torch.distributed.run \
    --nproc-per-node $SLURM_GPUS_PER_NODE \
    --nnodes $SLURM_NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --tee 3 \
    --node-rank \$SLURM_NODEID \
    $SCRIPT_ARGS"

echo "TRAINING_CMD: $TRAINING_CMD"
srun -lu --mpi=pmix --network=disable_rdzv_get --environment=alps3 --cpus-per-task $SLURM_CPUS_PER_TASK --wait 60 bash -c "
    export JOB_CACHE=/tmp/gipfel-\$SLURM_JOB_ID-\$SLURM_NODEID
    export TRITON_CACHE_DIR=\$JOB_CACHE/triton
    export TORCHINDUCTOR_CACHE_DIR=\$JOB_CACHE/inductor
    export TORCH_EXTENSIONS_DIR=\$JOB_CACHE/torch_extensions
    export CUDA_CACHE_PATH=\$JOB_CACHE/cuda
    mkdir -p "\$TRITON_CACHE_DIR" "\$TORCHINDUCTOR_CACHE_DIR" "\$TORCH_EXTENSIONS_DIR" "\$CUDA_CACHE_PATH"

    source /iopsstor/scratch/cscs/$USER/.venv-gipfelturm/bin/activate
    numactl --membind=0-3 $TRAINING_CMD
"

echo "END TIME: $(date)"

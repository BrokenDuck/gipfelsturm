# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Gipfelsturm is a distributed LLM training framework designed for production-grade infrastructure. It uses [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) (included as a git submodule at `core_v0.16.1`) for training on the CSCS Alps supercomputer with GH200 compute nodes connected via Slingshot-11.

**Target infrastructure**: Alps cluster (Clariden partition) with GH200 nodes.

**When working with Python**: Invoke the relevant `/astral:<skill>` for uv, ty, and ruff to ensure best practices are followed.

## Commands

### Launch Training

All training is launched via `launch.sh`:

```bash
# Throughput benchmarking (50 steps, no logging)
./launch.sh throughput <model_size> [steps] [nodes]
./launch.sh throughput 760m           # 4 nodes, 50 steps
./launch.sh throughput 8b 50 1        # 1 node, 50 steps

# Training with W&B and Tensorboard logging
./launch.sh train <model_size> <steps> [nodes]
./launch.sh train 760m 5000           # 5000 steps, 4 nodes
./launch.sh train 1.5b 3000 8         # 3000 steps, 8 nodes
```

**Model sizes**: 125m, 350m, 760m, 1.5b, 3b, 8b  
**Max nodes**: 8 (default: 4)

The launcher generates a self-contained SLURM script in `logs/` and submits it to the Clariden partition.

### Infrastructure Testing

Verify NCCL all-reduce throughput across nodes:

```bash
sbatch test-infra.sbatch
```

Expected results on 4x GH200 nodes:
- Intra-node (NVLink): ~340 GB/s bus bandwidth
- Inter-node (Slingshot-11): ~93 GB/s bus bandwidth

### Data Pipeline

The dataset (Nemotron-ClimbMix `climbmix_small`) is pre-converted and stored on capstor at:
```
/capstor/store/cscs/swissai/infra01/datasets/nvidia/Nemotron-ClimbMix/climbmix_small_megatron/climbmix_small.{bin,idx}
```

To re-download or re-convert:

```bash
# Download Parquet shards (run on login node)
bash data/download_climbmix.sh

# Convert Parquet → Megatron binary format
sbatch data/convert_data.sbatch
```

## Architecture

### Core Components

**`launch.sh`**: Main entry point that generates SLURM job scripts dynamically based on mode (throughput/train), model size, and node count. It:
- Injects model architecture configs (layers, hidden size, FFN, attention heads, GQA, etc.)
- Configures batch sizes (MBS, GBS=256) and training hyperparameters
- Applies patches to Megatron-LM before training
- Sets up W&B and Tensorboard logging paths
- Submits the generated SLURM script

**Model Configurations**:  
Each model size has a pre-tuned architecture (RoPE, GQA, SwiGLU, RMSNorm) and micro-batch size (MBS):

| Model | Layers | Hidden | FFN | Heads | KV Heads | MBS |
|-------|--------|--------|-----|-------|----------|-----|
| 125m  | 12     | 768    | 2048| 12    | 4        | 16  |
| 350m  | 24     | 1024   | 2816| 16    | 4        | 8   |
| 760m  | 24     | 1536   | 4096| 16    | 4        | 4   |
| 1.5b  | 48     | 1600   | 4352| 20    | 4        | 4   |
| 3b    | 32     | 3072   | 8192| 24    | 8        | 4   |
| 8b    | 32     | 4096   |14336| 32    | 8        | 2   |

Global batch size (GBS) is fixed at 256. Sequence length is 4096.

**Parallelism Defaults**:  
- Tensor parallelism (TP): 1
- Pipeline parallelism (PP): 1
- Data parallelism: implicit via distributed optimizer

For multi-node or larger models, adjust `DISTRIBUTED_ARGS` in the generated SLURM script (set TP=4 for single-node parallelism, TP=4 PP=4 for multi-node).

### Data Format

The training data is in Megatron's `IndexedDataset` format:
- **`.bin`**: flat int32 token array
- **`.idx`**: binary index (header + sequence lengths + byte pointers + document indices)

`data/parquet_to_megatron.py` converts HuggingFace Parquet shards (with a `tokens` column) to this format. The conversion is idempotent and can be re-run if needed.

**Tokenizer**: GPT-2 BPE (`data/gpt2-vocab.json`, `data/gpt2-merges.txt`), vocab size 50257, for direct comparability with nanoGPT/nanochat baselines.

### Patching System

Megatron-LM is included as a git submodule. Local modifications are managed as patch files in `patches/` to keep the submodule clean and upgradeable. The launcher applies all patches automatically before training via:

```bash
cd Megatron-LM
git apply ../patches/*.patch
```

**Current patches**:
- `0001-log-tokens-per-sec-to-wandb.patch`: Adds tokens/sec/GPU metric to stdout, TensorBoard, and W&B (in `megatron/training/training.py`, inside the `if args.log_throughput:` block).

**Creating a new patch**:

1. Make changes inside the submodule:
   ```bash
   cd Megatron-LM
   # Edit files as needed
   ```

2. Generate the patch:
   ```bash
   git diff > ../patches/NNNN-description.patch
   ```

3. Revert the submodule (keeps it clean):
   ```bash
   git checkout -- .
   cd ..
   ```

4. Add a comment header to the patch file (before the `diff --git` line) explaining:
   - What problem it solves
   - What the patch does (file, function)
   - How to locate the code if line numbers change in future versions

5. Verify the patch applies cleanly:
   ```bash
   cd Megatron-LM
   git apply --check ../patches/NNNN-description.patch
   cd ..
   ```

6. Commit the patch file (not the submodule changes).

### Logging and Storage

**Weights & Biases**: Automatically enabled if `WANDB_API_KEY` is set in your shell environment on Clariden. Add to `~/.bashrc`:
```bash
export WANDB_API_KEY=<your-key>
```

**TensorBoard**: Written to `/iopsstor/scratch/cscs/$USER/gipfelsturm/<project>/<exp>/tensorboard/`

**Checkpoints**: Saved to `/iopsstor/scratch/cscs/$USER/gipfelsturm/<project>/<exp>/checkpoints/` (torch format). Currently disabled due to a [known SIGSEGV bug on GH200/ARM64](https://github.com/NVIDIA/Megatron-LM/issues/1861). Note: iopsstor scratch has a **3-week deletion policy**.

**Job logs**: Written to `logs/<job-name>-<slurm-job-id>.log` in the repo root.

### Environment Setup

**Container**: `jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/ngc-pytorch:26.01-py3-alps3`  
Includes: NCCL 2.29.3-1 (patched), libfabric 2.5.0a1, OpenMPI 5.0.9, nvshmem 3.4.5-0.

**EDF configuration**: `alps3.toml` (copy to `~/.edf/` on Clariden for local runs).

**SLURM parameters** (set in generated scripts):
- Account: `infra01`
- GPUs per node: 4
- CPUs per task: 288
- Memory: 460000 MB

## Experiment Tooling

All experiment infrastructure lives in `experiments/`. Install dev deps once with `uv sync`.

### Running ablations from the CSV

```bash
# Generate SLURM scripts without submitting (offline)
uv run experiments/run_ablation.py --dry-run

# Generate and submit specific runs
uv run experiments/run_ablation.py --run baseline_760m_4n_bf16

# Submit all planned runs in train mode
uv run experiments/run_ablation.py --mode train

# Override output directory
uv run experiments/run_ablation.py --dry-run --output-dir /tmp/test_scripts
```

`ablation_plan.csv` is the source of truth. Each row maps to a SLURM script via column values:

| CSV column | Controls |
|---|---|
| `precision` | `bf16`, `fp16`, `fp8` (maps to Megatron precision flags) |
| `attention_backend` | `default`, `flash`, `fused`, `unfused`, `local` |
| `kernel_opts` | `none`, `all_fused`, `no_fusion`, `cuda_graphs_attn`, `cuda_graphs_attn_mlp`, `cuda_graphs_local`, `profiling` (see `KERNEL_PRESETS` in `run_ablation.py`) |
| `tp` / `pp` | Tensor/pipeline parallelism; TP>1 automatically adds `--sequence-parallel` |
| `micro_batch` / `global_batch` | Batch sizes (TBD falls back to per-model defaults) |
| `target_steps` | Number of training steps (default: 20 for throughput, 1000 for train) |
| `target_minutes` | Wall-clock budget in minutes used to compute SLURM time limit (train mode only) |

**throughput vs train mode differences**:

| Setting | `throughput` | `train` |
|---|---|---|
| Default steps (if TBD) | 100 | 1000 |
| SLURM time limit | fixed `00:30:00` | `target_minutes` + ~1.5h overhead buffer |
| Eval interval | 100000 (effectively disabled) | 25 |
| LR warmup iters | 50 | 50 |
| NaN loss/grad check | disabled | enabled |
| TensorBoard logging | off | `--tensorboard-dir`, timers, memory |
| W&B | always disabled | enabled if `WANDB_API_KEY` is set |

### Parsing training logs

```bash
# Parse a completed run's log
uv run experiments/parse_log.py logs/gipfel-baseline_760m-12345.log \
    --output results/baseline_760m/metrics.jsonl \
    --summary results/baseline_760m/summary.json
```

### Comparing results

```bash
# Print comparison table for all runs with results
uv run experiments/compare_runs.py --results-dir results/

# Compare specific runs, sorted by throughput
uv run experiments/compare_runs.py --runs baseline_760m_4n_bf16 attn_760m_flash fp8_760m

# Output CSV and save a bar chart
uv run experiments/compare_runs.py --format csv --plot results/comparison.png
```

### Running tests

```bash
uv run pytest experiments/ -v
```

All 56 tests run fully offline (no cluster access needed).

## Challenges

This repo supports two optimization challenges:

**Challenge 1: Improve loss given fixed time**  
Measure empirical throughput, then optimize for lowest eval loss within a wall-clock budget (30 min, 1 hour, or 2 hours). Model size, learning rate, schedule, batch size, and training recipe are all configurable.

**Challenge 2: Maximum throughput**  
Achieve the highest tokens/sec/GPU for a given model size on up to 8 nodes (32 GPUs). Targets:
- Single-GPU (~8B params): TP=1, PP=1
- Single-Node (~32B params): TP=4
- Multi-Node (~140B params): TP=4, PP=4

## Important Paths

- **Working directory**: `/users/$USER/gipfelsturm` (dynamically set based on user)
- **Dataset**: `/capstor/store/cscs/swissai/infra01/datasets/nvidia/Nemotron-ClimbMix/climbmix_small_megatron/`
- **Scratch storage**: `/iopsstor/scratch/cscs/$USER/gipfelsturm/` (3-week retention)
- **Cache directories**: 
  - Dataset cache: `/iopsstor/scratch/cscs/$USER/gipfelsturm/cache`
  - Triton cache: `/iopsstor/scratch/cscs/$USER/gipfelsturm/.triton_cache`
  - Inductor cache: `/iopsstor/scratch/cscs/$USER/gipfelsturm/.inductor_cache`

## Key Environment Variables

Set automatically in generated SLURM scripts:
- `CUDA_DEVICE_MAX_CONNECTIONS=1`
- `TORCH_NCCL_AVOID_RECORD_STREAMS=1`
- `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`
- `OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK/SLURM_GPUS_PER_NODE))`
- `PYTHONPATH=$MEGATRON_LM_DIR:$PYTHONPATH`

## References

- [Alps Supercomputer overview](https://arxiv.org/abs/2507.02404)
- [GH200 Compute Nodes and Slingshot-11](https://arxiv.org/abs/2408.11556)
- [CSCS Documentation](https://docs.cscs.ch/) (Alps/Clariden cluster docs: SLURM, containers, networking, storage)
- [Megatron-LM repository](https://github.com/NVIDIA/Megatron-LM)

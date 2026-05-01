# Experiment Tooling

All experiment infrastructure lives in `experiments/`. Install dev dependencies once:

```bash
uv sync
```

The four scripts form a pipeline:

```
ablation_plan.csv
      │
      ▼
run_ablation.py  ──►  logs/gipfel-<run>.sbatch  ──► (sbatch on cluster)
                                                          │
                                                          ▼
                                                   logs/<run>-<jobid>.log
                                                          │
                                                          ▼
                                               parse_log.py
                                                          │
                                          ┌───────────────┴────────────────┐
                                          ▼                                ▼
                              results/<run>/metrics.jsonl   results/<run>/summary.json
                                                                           │
                                                                           ▼
                                                              compare_runs.py  ──► table / CSV / PNG
```

---

## `run_ablation.py` — SLURM script generator

Reads `ablation_plan.csv` and renders one self-contained `.sbatch` script per row, then optionally submits it. Each script replicates the structure of `launch.sh` but exposes every training dimension as a configurable parameter.

### Usage

```bash
# Preview all scripts without submitting
uv run experiments/run_ablation.py --dry-run

# Preview a specific run
uv run experiments/run_ablation.py --dry-run --run baseline_760m_4n_bf16

# Submit specific runs (requires cluster)
uv run experiments/run_ablation.py --run baseline_760m_4n_bf16 --run attn_760m_flash

# Submit all planned runs in train mode
uv run experiments/run_ablation.py --mode train

# Write scripts to a custom directory
uv run experiments/run_ablation.py --dry-run --output-dir /tmp/scripts
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--csv PATH` | `experiments/ablation_plan.csv` | Source CSV |
| `--run NAME` | all rows | Submit only this named run (repeatable) |
| `--dry-run` | off | Generate scripts without calling `sbatch` |
| `--output-dir PATH` | `logs/` | Where to write `.sbatch` files |
| `--mode throughput\|train` | `throughput` | Training mode (affects logging, eval, W&B) |

### CSV columns

`ablation_plan.csv` is the source of truth. Each row becomes one job. `TBD` values fall back to per-model defaults.

| Column | Values | Notes |
|---|---|---|
| `run_name` | string | Unique identifier; becomes job name and results directory |
| `model_size` | `125m` `350m` `760m` `1.5b` `3b` `8b` | Determines architecture (layers, hidden, FFN, heads) |
| `nodes` | int | SLURM node count |
| `tp` | int | Tensor parallel size; `>1` automatically adds `--sequence-parallel` |
| `pp` | int | Pipeline parallel size |
| `seq_len` | int | Sequence length (default 4096) |
| `micro_batch` | int | Micro-batch size per GPU (default: per-model value) |
| `global_batch` | int | Global batch size (default 256) |
| `precision` | `bf16` `fp16` `fp8` | See precision details below |
| `attention_backend` | `default` `flash` `fused` `unfused` `local` | See attention details below |
| `kernel_opts` | preset name | See kernel presets below |
| `target_minutes` | int | Used to compute SLURM walltime in train mode |
| `target_steps` | int | Training iteration count |

### Precision mapping

| CSV value | Megatron flags |
|---|---|
| `bf16` | `--bf16` |
| `fp16` | `--fp16` |
| `fp8` | `--fp8-format hybrid --fp8-recipe delayed` |

FP8 uses `hybrid` format (e4m3 for weights/activations, e5m2 for output gradients) with delayed scaling — the standard TE recipe for training stability. Do not use `e4m3` uniform format; it is less numerically stable.

### Attention backend mapping

| CSV value | Megatron flags | Notes |
|---|---|---|
| `default` / `auto` | (none) | Let Transformer Engine choose |
| `flash` | `--attention-backend flash` | FlashAttention-style kernel |
| `fused` | `--attention-backend fused` | cuDNN fused attention |
| `unfused` | `--attention-backend unfused` | Baseline unfused |
| `local` | `--attention-backend local` | Local PyTorch implementation |

To debug which backend is actually selected at runtime, set `NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=1` in the environment or use the `profiling` kernel preset.

### Kernel presets

The `kernel_opts` column maps to a named preset in `KERNEL_PRESETS` (defined at the top of `run_ablation.py`). Adding a new preset is one dict entry — no other code changes required.

| Preset | Megatron flags added | Notes |
|---|---|---|
| `none` | — | Megatron defaults; cross-entropy fusion always on |
| `all_fused` | `--bias-activation-fusion --masked-softmax-fusion --bias-dropout-fusion` | Additional TE fusions |
| `no_fusion` | `--no-rope-fusion --no-gradient-accumulation-fusion --no-persist-layer-norm` | Ablation: disable fusions |
| `transformer_engine` | — | TE already the default; kept for explicit labelling |
| `cuda_graphs_attn` | `--use-te-rng-tracker --cuda-graph-impl transformer_engine --cuda-graph-scope attn` | CUDA graphs over attention only (low risk) |
| `cuda_graphs_attn_mlp` | `--use-te-rng-tracker --cuda-graph-impl transformer_engine --cuda-graph-scope attn,mlp` | CUDA graphs over attention + MLP |
| `cuda_graphs_local` | `--use-te-rng-tracker --cuda-graph-impl local --cuda-graph-scope full_iteration` | Full-iteration MCore graph capture |
| `profiling` | — (env vars only) | Injects `NVTE_NVTX_ENABLED=1 NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=1` for NSYS tracing |

**CUDA graph requirements**: all CUDA graph presets require `--use-te-rng-tracker` (injected automatically). Full-iteration graphs also require `--no-check-for-nan-in-loss-and-grad` (always set in `TRAINING_ARGS`). Only use CUDA graphs with static tensor shapes: fixed `seq_len`, fixed MBS, no dynamic packing.

### Mode differences

| Setting | `throughput` | `train` |
|---|---|---|
| Steps | `target_steps` (default 50) | `target_steps` (default 1000) |
| Eval | disabled | every 1000 steps, 10 iters |
| LR warmup | 10 iters | 200 iters |
| W&B | disabled | enabled if `WANDB_API_KEY` set |
| TensorBoard | disabled | enabled with timers and memory |
| Timing detail | — | `--timing-log-level 1 --timing-log-option minmax` |
| SLURM time | 00:30:00 | derived from `target_minutes` + 1.5h buffer |

---

## `parse_log.py` — training log parser

Parses the Megatron stdout captured by SLURM (in `logs/<job>.log`) into structured per-iteration metrics.

### Usage

```bash
# Stream JSONL to stdout
uv run experiments/parse_log.py logs/gipfel-baseline_760m-12345.log

# Write JSONL to file
uv run experiments/parse_log.py logs/gipfel-baseline_760m-12345.log \
    --output results/baseline_760m/metrics.jsonl

# Write JSONL + summary statistics
uv run experiments/parse_log.py logs/gipfel-baseline_760m-12345.log \
    --output results/baseline_760m/metrics.jsonl \
    --summary results/baseline_760m/summary.json

# Skip more warmup iterations for summary stats
uv run experiments/parse_log.py logs/... --summary results/.../summary.json --skip-first 20
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `log_file` | (required) | Path to SLURM `.log` file |
| `--output PATH` | stdout | Write per-iteration JSONL |
| `--summary PATH` | (none) | Write aggregate statistics as JSON |
| `--run-name NAME` | stem of log filename | Embedded in summary `run_name` field |
| `--skip-first N` | 5 | Exclude first N iterations from summary stats (warmup) |

### Output formats

**`metrics.jsonl`** — one JSON object per training iteration:

```json
{"iteration": 42, "total_iters": 500, "consumed_samples": 10752,
 "elapsed_ms": 892.1, "tflops_per_gpu": 245.3, "tokens_per_sec_gpu": 74994,
 "learning_rate": 0.0003, "global_batch_size": 256, "lm_loss": 3.2145,
 "loss_scale": 1.0, "grad_norm": 0.543, "skipped_iters": 0, "nan_iters": 0}
```

Fields are present only when the corresponding log field exists. `tokens_per_sec_gpu` requires the `0001-log-tokens-per-sec-to-wandb.patch` to be applied (done automatically by `launch.sh` and `run_ablation.py`).

**`summary.json`** — aggregate statistics over steady-state iterations:

```json
{
  "run_name": "baseline_760m_4n_bf16",
  "total_iterations": 500,
  "steady_state_iterations": 495,
  "mean_tokens_per_sec_gpu": 74994,
  "median_tokens_per_sec_gpu": 75012,
  "p95_tokens_per_sec_gpu": 75800,
  "mean_tflops_per_gpu": 245.3,
  "median_elapsed_ms": 892.1,
  "final_lm_loss": 3.2145,
  "min_lm_loss": 3.1201,
  "mean_lm_loss_last_10pct": 3.2089
}
```

### Parser design

The parser uses field-by-field regex extraction rather than a monolithic pattern. Each field has its own independent regex, so the parser does not break if Megatron adds or reorders fields across versions. Non-training lines (headers, evaluation output, status messages) are silently skipped.

---

## `compare_runs.py` — results comparison

Loads `summary.json` files from multiple runs and prints a comparison table. Computes speedup relative to a baseline.

### Usage

```bash
# Compare all runs that have results
uv run experiments/compare_runs.py

# Compare specific runs
uv run experiments/compare_runs.py --runs baseline_760m_4n_bf16 attn_760m_flash fp8_760m

# Set explicit baseline for speedup calculation
uv run experiments/compare_runs.py --baseline baseline_760m_4n_bf16

# Output CSV instead of table
uv run experiments/compare_runs.py --format csv > results/comparison.csv

# Save bar chart
uv run experiments/compare_runs.py --plot results/comparison.png

# Sort by loss instead of throughput
uv run experiments/compare_runs.py --sort final_lm_loss
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--results-dir PATH` | `results/` | Directory containing `<run>/summary.json` subdirs |
| `--runs NAME...` | all | Specific runs to include |
| `--baseline NAME` | first run | Reference run for speedup column |
| `--sort METRIC` | `mean_tokens_per_sec_gpu` | Sort key (descending) |
| `--format table\|csv` | `table` | Output format |
| `--plot PATH` | (none) | Save bar chart PNG (requires matplotlib) |

### Output columns

| Column | Description |
|---|---|
| `Run` | Run name |
| `tok/s/GPU` | Mean tokens/sec/GPU (steady-state) |
| `TFLOP/s` | Mean TFLOP/s/GPU |
| `iter_ms` | Median iteration time (ms) |
| `final_loss` | LM loss at last iteration |
| `loss_10%` | Mean LM loss over last 10% of iterations |
| `iters` | Total iteration count |
| `speedup` | Throughput relative to baseline |

Example output:

```
Run                      tok/s/GPU    TFLOP/s    iter_ms    final_loss    loss_10%    iters    speedup
-----------------------  -----------  ---------  ---------  ------------  ----------  -------  ---------
fp8_760m                 89,993       294.2      743.1      3.2301        3.2280      500      1.20x
attn_760m_flash          82,493       269.8      810.5      3.2012        3.1990      500      1.10x
baseline_760m_4n_bf16    74,994       245.3      892.1      3.2145        3.2100      500      1.00x
```

---

## `compute_budget.py` — token budget calculator

Converts an empirical throughput measurement into a target step count for a fixed wall-clock budget.

### Usage

```bash
uv run experiments/compute_budget.py \
    --tok-s-gpu 74994 \
    --gpus 16 \
    --seq-len 4096 \
    --global-batch 256 \
    --minutes 60
```

Output:
```
total_tok_s:       1,199,904
target_tokens:     4,319,654,400
tokens_per_step:   1,048,576
target_steps:      4,119
estimated_minutes: 60.00
```

### Arguments

| Argument | Description |
|---|---|
| `--tok-s-gpu` | Measured tokens/sec/GPU from a throughput run |
| `--gpus` | Total GPU count (nodes × 4) |
| `--seq-len` | Sequence length (must match training config) |
| `--global-batch` | Global batch size (must match training config) |
| `--minutes` | Wall-clock budget in minutes |

### Typical workflow

1. Run `./launch.sh throughput <model_size>` to measure empirical throughput.
2. Read `tokens/sec/GPU` from the log output.
3. Feed into `compute_budget.py` to get `target_steps`.
4. Fill `target_steps` into `ablation_plan.csv` for the corresponding training rows.

---

## Typical end-to-end workflow

```bash
# 1. Edit experiments/ablation_plan.csv — fill in run parameters

# 2. Preview generated SLURM scripts
uv run experiments/run_ablation.py --dry-run
cat logs/gipfel-baseline_760m_4n_bf16.sbatch  # verify

# 3. On the cluster: measure throughput for a model
./launch.sh throughput 760m 50 4

# 4. Convert throughput → target steps
uv run experiments/compute_budget.py --tok-s-gpu 74994 --gpus 16 --seq-len 4096 --global-batch 256 --minutes 60

# 5. Fill target_steps in ablation_plan.csv, then submit train runs
uv run experiments/run_ablation.py --mode train

# 6. After jobs complete, parse logs
for log in logs/gipfel-baseline_760m_4n_bf16-*.log; do
    run=$(basename $log | sed 's/gipfel-//;s/-[0-9]*.log//')
    uv run experiments/parse_log.py "$log" \
        --output "results/$run/metrics.jsonl" \
        --summary "results/$run/summary.json"
done

# 7. Compare results
uv run experiments/compare_runs.py --baseline baseline_760m_4n_bf16
uv run experiments/compare_runs.py --format csv > results/comparison.csv
uv run experiments/compare_runs.py --plot results/comparison.png
```

## Running tests

All tests are fully offline:

```bash
uv run pytest experiments/ -v
```

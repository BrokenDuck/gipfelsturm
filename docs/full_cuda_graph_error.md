# Full-Iteration CUDA Graph Capture Failure

**Job**: `gipfel-full_graph-3348599` (4 nodes, 16 GPUs, TP=1, PP=1)  
**Date**: 2026-05-05  
**Status**: Will not fix — incompatible with planned TP>1 scale-up.

## Error

```
torch.AcceleratorError: CUDA error: operation not permitted when stream is capturing
Search for `cudaErrorStreamCaptureUnsupported' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html
```

## Flags Used

```
--cuda-graph-impl local
--te-rng-tracker
--cuda-graph-scope full_iteration
--cuda-graph-warmup-steps 3
--no-check-for-nan-in-loss-and-grad
--tensor-model-parallel-size 1
--use-distributed-optimizer
--overlap-grad-reduce
--overlap-param-gather
```

## Root Cause

`FullCudaGraphWrapper` (in `megatron/core/full_cuda_graph.py`) captures the entire training iteration as a CUDA graph. It pre-fetches data via `StaticBufferLoader.data_read()` **before** the capture begins, copying batch tensors into static CUDA buffers. However, during the actual graph capture, `forward_step` (in `pretrain_gpt.py:248`) still calls:

```
forward_step → get_batch → get_batch_on_this_tp_rank → _broadcast_cu_seqlens
```

Inside `_broadcast_cu_seqlens` (`megatron/training/utils.py:566-581`), two operations are incompatible with CUDA graph capture:

1. **Dynamic CUDA tensor allocation** (line 569):
   ```python
   n_tensor = torch.tensor(n, dtype=torch.int64, device=dev)
   ```
   This allocates a new CUDA tensor every iteration. CUDA graph capture forbids dynamic allocations — all memory must be pre-allocated during warmup.

2. **NCCL broadcast** (line 570, via `_broadcast(n_tensor)`):
   ```python
   torch.distributed.broadcast(item, src_rank, group=tp_group)
   ```
   NCCL collectives cannot be captured in a CUDA graph. With TP=1 (single-rank group), PyTorch elides the actual NCCL call, but the `torch.tensor()` allocation on line 569 still crashes the capture. With TP>1 the broadcast itself would also crash.

The `_broadcast_cu_seqlens` path executes unconditionally — even when `cu_seqlens` is `None` (non-packed/non-SFT data), it still allocates `torch.tensor(0, ...)` to broadcast the length.

## Why the Demo Command Works

The Megatron-LM documentation shows:
```
python pretrain_gpt.py \
    --enable-cuda-graph \
    --cuda-graph-scope full_iteration \
    --cuda-graph-warmup-steps 1 \
    --te-rng-tracker \
    --no-check-for-nan-in-loss-and-grad
```

`--enable-cuda-graph` is a deprecated alias for `--cuda-graph-impl local` (see `arguments.py:510-515`). The demo likely works with an older Megatron version where `_broadcast_cu_seqlens` did not exist (it was added to support packed-sequence SFT and variable-length batches). Our submodule (`core_v0.16.1`) includes this newer code path.

## Why We Will Not Fix This

A patch to `_broadcast_cu_seqlens` (pre-allocating static buffers, using `.fill_()` instead of `torch.tensor()`) would fix the TP=1 case. However:

- **TP>1 is planned.** With TP>1, the `torch.distributed.broadcast` calls inside `get_batch_on_this_tp_rank` become real multi-rank NCCL collectives that fundamentally cannot be captured in a CUDA graph. No local patch can fix this.
- **Design mismatch.** `FullCudaGraphWrapper.data_read` pre-fetches raw data, but `get_batch_on_this_tp_rank` is still invoked inside the captured `forward_step`. The entire TP broadcast logic (lines 585-605 of `utils.py`) runs inside the capture boundary. Moving `get_batch` outside the graph boundary would require significant restructuring of `forward_step`.
- **Fragility.** Any future Megatron update that adds a dynamic allocation or collective inside `forward_step` would silently break the graph again.

## Recommendation

Use the **`te` (TransformerEngine-scoped) CUDA graph preset** instead:
```
--cuda-graph-impl transformer_engine
--te-rng-tracker
--cuda-graph-scope attn,mlp
```

This captures only the attention and MLP transformer blocks — not data loading, not the optimizer step, not TP broadcasts. It is compatible with TP>1 and scales safely. The performance benefit is smaller than full-iteration capture but is robust.

## Related Files

| File | Role |
|------|------|
| `megatron/core/full_cuda_graph.py` | `FullCudaGraphWrapper` and `StaticBufferLoader` |
| `megatron/training/utils.py:523-605` | `get_batch_on_this_tp_rank` and `_broadcast_cu_seqlens` |
| `pretrain_gpt.py:248` | `forward_step` calls `get_batch` inside the capture |
| `megatron/training/training.py:2812-2813` | Where `FullCudaGraphWrapper` wraps `forward_backward_func` |
| `megatron/training/arguments.py:510-515` | `--enable-cuda-graph` → `--cuda-graph-impl local` alias |
| `logs/throughput/gipfel-full_graph-3348599.{out,err}` | Failure logs |
| `logs/throughput/gipfel-full_graph.sbatch` | Generated SLURM script |

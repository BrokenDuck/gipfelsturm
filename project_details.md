# Wildstrubel - 3243m

![Wildstrubel](https://www.meinweekend.ch/assets/img/offers/182/Wildstrubel_2.JPG)


# Challenge -> Eval Loss Minimizing

- Use Megegatron
- Figure Out Dataset Size
- Figure Out Model Size (Parameters, Starting small and getting bigger, hopefully with higher throughput.)
- Take advantage of CUDA 13
- Create MoE
- Train on 64 GPU's for an hour


Optional if implementation above works quickly:
- Quantization from Gemma 4
- Flashattention (Higher Throughput -> More intelligence)

Cuda 13 Updates:
- NVFP4 - FP4 Training (Only Blackwell afaik)
- CUDA Graphs: more flexible "Polymorphic" graph nodes (Giving all instructions at once instead of 1000 tasks after eachother, less communication)
- CuTile: Makes handling memory at the "tile" level easier (tile < threads, lower level). -> Flashattention and Mamba/Linear Attention is easier on Blackwell.

# Implementation

- NCCL for Communications (64 GPU's) -> Alps-optimized NCCL hooks
- Alps extended image (use AWS_OFI_NCCL)
- CUDA 13 -> FP8/MXFP8 routines
- L2 Multicasting for DRAM read/write pressure reduction
- THP (Hugepages) - optimizes memory access
- PyTorch

# Requirements / Given
- GPT2 Tokenizer


# First Steps

Setup Baseline Model. 
Setup Metrics
- tokens/sec/GPU
- memory usage
- max stable batch size
- effect on loss stability

## Comparisons - Upgrades
### Attention
- Flashattention 2
- Flashattention 3
- cuDNN SDPA backend

## Kernel 
Runtime Improvement
- fused RMSNorm
- fused cross-entropy
- fused optimizer kernels

- Liger kernels
- Megatron fused ops already available
- torch.compile where compatible
- CUDA Graphs for steady-state iteration speed

## FP Precision Reduction
- FP16
- BF16
- FP8
- FP4?!

Measure:
- throughput gain
- memory reduction
- training stability
- final eval loss impact

## Loss-optimization
### Optimizer
- AdamW (baseline)
- AdEMAMix 
### LR Scheduler
- cosine decay
- Warmup-Stable-Decay

## Batch-size Scaling
figure out optimal Batch-size

## Parallelism
- DP
- TP + DP
- TP + PP + DP
Tradeoff:
- model size reached
- pipeline bubble overhead
- communication cost
- actual scaling efficiency

### Single Node - Multi Node 
multi-node overhead too large.
optimal size?
Metrics:
- MFU / throughput
- tokens/sec/GPU
- final eval loss within same GPU-hour budget

## Checkpointing
Build robust long-run training support:
automatic checkpoint on time limit
graceful exit and resume
auto-resubmission workflow
This is very aligned with the project brief and easy to demonstrate

# Major Changes
- custom implement xIELU (unrealistic, maybe using existing)
- DeltaNet/DeltaProduct (activation functions)
[A Visual Guide to Gemma 4](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4)

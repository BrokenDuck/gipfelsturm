
# Wildstrubel 3244m — Gipfelsturm Project Proposal

## Team

- Jonathan Maillefaud
- Dominique Buob
- (Haven't found another student)

## Goal

- *Eval loss minimization*
- Target setup: *64 GPUs for 1 hour*
- Secondary goal: improve throughput enough to process more tokens within the same wall-clock budget

## Requirements

- *Megatron-LM* -> training framework
- *Nemotron-ClimbMix* -> dataset
- *GPT-2 BPE tokenizer*
- Multi-GPU / multi-node
- *PyTorch* -> training stack
- *NCCL* -> communication backend
- *Logging and evaluation* -> (eval loss, tokens/sec/GPU, memory usage, max stable batch size, scaling efficiency)
- *Checkpointing*

## Methods

Megatron baseline and compare targeted improvements (Gipfelsturm type of explanation^^)

### Base -> 1500m

1. *Dense baseline model* (Standard transformer, AdamW, cosine decay, warmup-stable-decay)
2. *Batch-size scaling*
   - largest stable batch size
   - impact on throughput and loss
3. *Precision comparison*
   - *BF16* baseline
   - *FP16*
   - *FP8* (FP4 available only on blackwell for cuda 13 afaik)
4. *Parallelism baseline*
   - *DP*
   - *TP + DP*

### 1500m -> 2750m

1. *Attention backend comparison*
   - PyTorch *scaled_dot_product_attention*
   - *FlashAttention-2*
   - *FlashAttention-3*
   - *cuDNN SDPA backend*
2. *Kernel/runtime improvements*
   - Megatron fused ops
   - fused RMSNorm
   - fused cross-entropy
   - fused optimizer kernels
   - *Liger kernels*
   - *torch.compile*
   - *CUDA Graphs*
3. *Parallelism scaling*
   - *TP + PP + DP*
   - Compare single-node vs multi-node overhead

### 2750m -> 3244m (Final Ascent)

1. *MoE variant*
   - maybe possible, deepseek style

## Model Sizes

- *Small:* ~135M–500M parameters
- *Medium:* ~1B–3B parameters
- *Large:* ~3B–8B parameters
- *Stretch / MoE:* equivalent quality target in the 3B–8B regime if feasible

We will choose size after the fact

## Milestones

1. *Run Megatron-LM successfully* (Dominique)
2. *Establish a baseline* (Dominique)
3. *Build a comparison pipeline*, quick iteration (Dominique)
4. *Explore attention methods* (Jonathan)
5. *Integrate MoE architectures* (Jonathan)
6. *Explore runtime optimizations (CUDA Graph, etc...)* (Jonathan)
7. *Scale up* (Jonathan)
8. *Try out quantization* (Dominique)
9. *Run final experiment* (Jonathan)

## Expected Outcome

We expect to identify which combination of model size, precision, attention backend, and runtime optimizations gives the *lowest eval loss within the fixed budget*, while also producing a clear throughput and scaling analysis.

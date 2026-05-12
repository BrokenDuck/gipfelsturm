# Implementation Progress

Progress on implementing various features.

- [x] *Auto Job Resubmission*: somewhat performed with [run_ablation.py](../experiments/run_ablation.py)
- [ ] *Graceful Exit Mechanism*: Not implemented
- [ ] *Torch Fault Tolerance*: Not enabled
- [x] *Attention kernels*: Can choose between Flash Attention 3 and cuDNN SPDA/FrontEnd
- [x] *Torch Fusion*: Natively supported and activated, can be disabled for ablation
- [x] *CUDA Graph Fusion*: Natively supported, can be activated and choose between TransformerEngine/Forward Pass/Forward+Backward Pass fusion
- [x] *Liger Kernel*: not implemented because implementation costs are high. Most of the liger kernel operations are already fused by megatron core. Liger would mostly be for ablations.
- [x] *Cut Cross Entropy*: not implemented because megatron already has its own alternative ([VocalParallelCrossEntropy](../Megatron-LM/megatron/core/tensor_parallel/cross_entropy.py)). Cut cross entropy would mostly be for ablations.
- [x] *Quack kernels*: not implemented because implementation costs are high. TransformEngine already exposes some highly optimized GEMM operations. Although this is very interesting, it would mostly be for ablations.
- [x] *FP8 Training*: Implemented, we support the 3 fp8 reciped [exposed by megatron bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/training/mixed_precision.py)
- [x] *Data Parallel*: we support DDP, the Megatron-Core distributed Optimizer, Torch/Megatron FSDP and DeepSpeed ZeRO-1/2/3
- [ ] *Long Context Training*: not implemented yet
- [ ] *MOE*: not implemented yet
- [ ] *Attention Alternatives*: not implemented yet
- [ ] *Hybrid Models*: not implemented yet
- [ ] *Precision-Awra
- [ ]
- [ ]
- [ ]
- [ ]
- [ ]
- [ ]
- [ ]
- [ ]
- [ ]
- [ ]
- [ ]
- [ ]
- [ ]
- [ ]
- [ ]
- [ ]
- [ ]

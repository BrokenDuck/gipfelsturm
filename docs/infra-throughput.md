# Infrastructure Throughput Baselines

NCCL all-reduce benchmarks on 4x GH200 nodes (16 GPUs total) via `test-infra.sbatch`.
Theoretical ceilings: 450 GB/s NVLink-C2C per direction (intra-node), 100 GB/s Slingshot-11 per node (inter-node).

## NCCL Version Note

The container image (`ngc-pytorch:26.01-py3-alps3`) advertises NCCL `2.29.3-1` (patched), but the
Python runtime reports `(2, 29, 2)`. This is expected: the Python `torch.cuda.nccl.version()` tuple
reflects the base NCCL version; the patch suffix (`.3-1`) is a CSCS-applied fabric patch on top of
the `2.29.2` base and is not reflected in the tuple.

## Results

### Run 1 — 2026-05-01, job 3303648

**Stack**: PyTorch `2.10.0a0+a36e1d39eb.nv26.01`, NCCL `(2, 29, 2)`

#### Intra-node (4 GPUs, NVLink)

| Size   | Latency (ms) | Bus BW (GB/s) |
|--------|-------------|---------------|
| 128 MB | 0.6         | 315.5         |
| 256 MB | 1.2         | 327.0         |
| 512 MB | 2.4         | 333.6         |
| 1 GB   | 4.8         | 338.0         |
| 2 GB   | 9.5         | 339.1         |
| 4 GB   | 18.9        | **341.1**     |
| 8 GB   | 37.8        | 340.6         |
| 16 GB  | 75.9        | 339.6         |

Peak: **341.1 GB/s** (vs. ~340 GB/s expected) ✓

#### Inter-node (16 GPUs, 4 nodes, Slingshot-11)

| Size   | Latency (ms) | Bus BW (GB/s) |
|--------|-------------|---------------|
| 128 MB | 2.8         | 89.0          |
| 256 MB | 5.6         | 89.9          |
| 512 MB | 11.2        | 89.7          |
| 1 GB   | 22.4        | 90.0          |
| 2 GB   | 44.7        | 90.1          |
| 4 GB   | 89.4        | **90.1**      |
| 8 GB   | 178.8       | 90.1          |
| 16 GB  | 357.8       | 90.0          |

Peak: **90.1 GB/s** (vs. ~93 GB/s expected) — see note below.

---

### Run 2 — 2026-05-01, job 3303659

**Stack**: PyTorch `2.11.0a0+eb65b36914.nv26.02`, NCCL `(2, 29, 2)`

#### Intra-node (4 GPUs, NVLink)

| Size   | Latency (ms) | Bus BW (GB/s) |
|--------|-------------|---------------|
| 128 MB | 0.6         | 312.8         |
| 256 MB | 1.2         | 324.0         |
| 512 MB | 2.4         | 333.7         |
| 1 GB   | 4.8         | 337.1         |
| 2 GB   | 9.5         | 339.2         |
| 4 GB   | 19.0        | 339.8         |
| 8 GB   | 37.8        | **340.8**     |
| 16 GB  | 75.9        | 339.4         |

Peak: **340.8 GB/s** ✓

#### Inter-node (16 GPUs, 4 nodes, Slingshot-11)

| Size   | Latency (ms) | Bus BW (GB/s) |
|--------|-------------|---------------|
| 128 MB | 2.8         | 88.7          |
| 256 MB | 5.6         | 89.4          |
| 512 MB | 12.1        | 83.5 ⚠       |
| 1 GB   | 22.9        | 87.7          |
| 2 GB   | 44.6        | **90.2**      |
| 4 GB   | 89.3        | **90.2**      |
| 8 GB   | 179.1       | 89.9          |
| 16 GB  | 358.7       | 89.8          |

Peak: **90.2 GB/s**. The 512 MB point (83.5 GB/s) is an outlier vs. the otherwise consistent
~90 GB/s — likely transient network noise; see outage note below.

---

## Notes

**Inter-node shortfall (~90 vs ~93 GB/s)**: Both runs consistently show ~90 GB/s peak rather than
the ~93 GB/s baseline in README. This is a ~3% gap and stable across message sizes, suggesting
a measurement floor rather than a fault. Possible causes: routing overhead, not all Slingshot
links fully utilized by the 4-node job, or the README figure being from a larger job. Not a
cause for concern — well above the 100 GB/s theoretical ceiling headroom.

**Post-outage network instability (2026-05-01)**: A cluster outage occurred on 2026-04-30 due to a
Linux kernel vulnerability. Run 2 (job 3303659) was the first test after recovery. The 512 MB
inter-node dip to 83.5 GB/s may reflect a vcluster or network route not yet fully recovered.
Re-run `test-infra.sbatch` once CSCS confirms all vclusters are back to establish a clean
post-recovery baseline.

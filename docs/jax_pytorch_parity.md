# JAX and PyTorch Pi0 parity test

This report was generated on September 18, 2026 using four NVIDIA RTX 5880 Ada GPUs,
the local `physical-intelligence/libero` dataset, and the official OpenPI checkpoints.
The reproducible benchmark and plotting entry points are `scripts/compare_jax_pytorch.py`
and `scripts/plot_jax_pytorch_parity.py`.

## Inference output parity

Inference uses the same normalized LIBERO observation, checkpoint, initial Gaussian noise,
and Euler step count. Only the seven environment action dimensions are included in the
error metrics.

| Model | Steps | Cosine similarity | MAE | RMSE | Max absolute error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pi0 | 1 | 0.999906 | 0.005576 | 0.007396 | 0.036131 |
| Pi0 | 10 | 0.983415 | 0.012073 | 0.103909 | 1.928929 |
| Pi0.5 | 1 | 0.999928 | 0.004044 | 0.005433 | 0.016428 |
| Pi0.5 | 10 | 0.999993 | 0.001430 | 0.001932 | 0.005621 |

Pi0's ten-step maximum error comes from one gripper value at horizon index 41. The
single-step velocity output is close, so this is an iterative ODE amplification effect.
Pi0.5 remains closely aligned through all ten Euler steps.

Steady eager inference latency for batch size 1 was 69.62 ms for JAX and 273.68 ms for
PyTorch on Pi0, and 77.82 ms for JAX and 331.94 ms for PyTorch on Pi0.5. Compiling the
entire PyTorch Euler loop with either `max-autotune` or `reduce-overhead` did not reach a
usable warm state within several minutes, so eager numbers are reported.

Output plots are generated at:

- `artifacts/parity/plots/pi0_action_outputs.png`
- `artifacts/parity/plots/pi05_action_outputs.png`

## 200-step LIBERO training

All runs use four GPUs, seed 7, global batch size 4, AdamW, the official learning-rate
schedule, image augmentation, gradient clipping, and the same shuffled dataset order.
PyTorch DDP loss is reduced across all four ranks before logging. Full fine-tuning enables
EMA with the official decay (`0.99` for Pi0 and `0.999` for Pi0.5); LoRA follows the official
configs and disables EMA.

| Model | Mode | JAX result | PyTorch first-20 mean | PyTorch last-20 mean | PyTorch throughput |
| --- | --- | --- | ---: | ---: | ---: |
| Pi0 | Full | non-finite from step 0 | 0.13234 | 0.16092 | 0.79 step/s |
| Pi0.5 | Full | step 0=0.1165, non-finite from step 1 | 0.08888 | 0.07762 | 0.74 step/s |
| Pi0 | LoRA | step 0=0.1455, non-finite from step 1 | 0.15051 | 0.13649 | 1.47 step/s |
| Pi0.5 | LoRA | loss about 1.05 and parameter norm `inf` | 0.08356 | 0.07885 | 1.48 step/s |

The PyTorch runs completed all 200 steps with finite losses. Pi0.5 full, Pi0 LoRA, and
Pi0.5 LoRA reduced their 20-step moving-average loss. Pi0 full stayed finite but did not
show a downward trend over only 200 warmup steps.

The official JAX runs are not generally expected to use a global batch size of 4: the
repository defaults are 32 for Pi0 and 256 for Pi0.5. Therefore these non-finite results
characterize this requested four-GPU micro-benchmark, not the published large-batch JAX
training recipe. A one-step Pi0 JAX health run at the same seed and batch size was finite,
which indicates that the instability depends on the longer compiled training execution.

The complete curve plot is generated at `artifacts/parity/plots/training_curves_200.png`.
Raw logs and parsed JSON are under `artifacts/parity/`.

## PyTorch EMA implementation

PyTorch full fine-tuning now tracks EMA in float32. Under DDP, EMA parameters are sharded
across ranks so each GPU owns only `1 / world_size` of the shadow parameters. At checkpoint
time the shards are merged into `model.safetensors`, which is the inference/EMA model;
`train_model.safetensors` stores the live training parameters for resume, together with the
optimizer state. This avoids the rank-0 out-of-memory failure observed with a complete
13 GiB fp32 EMA copy on one GPU.

## PyTorch LoRA implementation

The PyTorch Gemma attention now supports LoRA for query, joint key/value, and output
projections using the same einsum layouts as JAX. Full checkpoints are reshaped into the
LoRA attention layout at load time, and only the intended base expert parameters are
frozen. Both Pi0 and Pi0.5 LoRA configurations completed the 200-step DDP test.

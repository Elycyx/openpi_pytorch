# PyTorch inference optimizations

Updated: 2026-09-21

The PyTorch Pi0/Pi0.5 implementation enables four inference-only optimizations by default. They preserve the model equations and the existing Euler integration schedule. Each optimization has an independent `Pi0Config` switch so the previous implementation remains available for parity checks and debugging.

## Configuration

| Option | Default | Effect |
| --- | --- | --- |
| `pytorch_optimize_prefix_cache` | `True` | In the final prefix transformer layer, computes only the K/V projections required by suffix attention. It skips the unused prefix Q projection, attention output, output projection, FFN, residuals, and final norm. |
| `pytorch_discard_suffix_cache` | `True` | Stops retaining and returning the newly generated prefix+suffix K/V cache from each Euler step. The input prefix cache is still used normally. |
| `pytorch_precompute_suffix_metadata` | `True` | Builds the suffix attention mask and position IDs once before Euler integration instead of rebuilding identical tensors at every step. |
| `pytorch_cache_time_embedding_frequencies` | `True` | Caches the fixed sinusoidal time-embedding frequencies and reuses them at every Euler step. |

The defaults are defined in `src/openpi/models/pi0_config.py`. To restore the previous inference path completely:

```python
config = Pi0Config(
    pytorch_optimize_prefix_cache=False,
    pytorch_discard_suffix_cache=False,
    pytorch_precompute_suffix_metadata=False,
    pytorch_cache_time_embedding_frequencies=False,
)
```

The switches do not affect `compute_loss` or training. The cached time-frequency vector is used only by the suffix path called from the Euler sampler.

## Implementation notes

### Prefix cache-only final layer

Every intermediate prefix layer must produce its full hidden states because the following layer consumes them. In the final layer, however, action-suffix inference only consumes the prefix K/V cache. The optimized path therefore performs the same pre-attention normalization and K/V projections as before, including RoPE on K, then stops without producing an unused prefix hidden state.

### Suffix cache lifetime

Each suffix layer still concatenates the input prefix cache with the current suffix K/V tensors because attention needs the combined sequence. The optimization only prevents those combined tensors from being accumulated into a returned per-layer cache that the Euler sampler immediately discards.

### Precomputed suffix metadata

For a fixed observation and action horizon, suffix validity masks, autoregressive block masks, full prefix-to-suffix masks, and suffix positions are invariant across Euler steps. The optimized sampler constructs these tensors once after the prefix cache is built.

### Cached time frequencies

The frequency vector depends only on the action-expert width and the fixed `[4e-3, 4.0]` period range. The timestep-dependent outer product and sine/cosine operations are unchanged.

## Validation

`src/openpi/models_pytorch/pi0_pytorch_test.py` compares the fully optimized and fully legacy paths with identical weights, observations, noise, and Euler steps. It requires exact equality (`rtol=0`, `atol=0`) for both Pi0 and Pi0.5 dummy models. A separate test verifies that the optimized prefix path does not invoke the final layer's prefix query projection, output projection, or MLP.

## Latency benchmark

`scripts/benchmark_pi05_latency.py` runs the following comparison matrix:

1. JAX.
2. PyTorch eager with all four optimizations disabled.
3. PyTorch `max-autotune` compile with all four optimizations disabled.
4. PyTorch `max-autotune` compile with all four optimizations enabled.

The fourth worker writes `pytorch-compile-optimized.json` and `pytorch-compile-optimized.log`. Passing `--skip-compile` skips both compiled PyTorch workers.

Both compile workers store their persistent Inductor caches inside the checkpoint directory:

```text
<checkpoint-dir>/.torchinductor_cache/max-autotune-legacy/
<checkpoint-dir>/.torchinductor_cache/max-autotune-optimized/
```

The legacy and optimized graphs therefore cannot overwrite each other's cache entries. Re-running the benchmark with the same checkpoint, software stack, GPU architecture, and input shapes reuses these caches.

Each compiled benchmark JSON records the selected directory in `compile_cache_dir` so cache placement can be verified from the result artifact.

`create_trained_policy` uses the same checkpoint-local cache convention for deployed PyTorch policies. It selects a cache profile from the compile mode and the four optimization switches before constructing the model. Graph tracing and guard checks still run in a new process, but cached Inductor kernels, FX graphs, and autotune results can be reused.

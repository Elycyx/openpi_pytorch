from __future__ import annotations

from collections.abc import MutableMapping, Sequence
import os
import pathlib
import re

INFERENCE_OPTIMIZATION_FIELDS = (
    "pytorch_optimize_prefix_cache",
    "pytorch_discard_suffix_cache",
    "pytorch_precompute_suffix_metadata",
    "pytorch_cache_time_embedding_frequencies",
)


def optimization_flags(model_config) -> tuple[bool, ...]:
    return tuple(bool(getattr(model_config, field, True)) for field in INFERENCE_OPTIMIZATION_FIELDS)


def compile_cache_profile(compile_mode: str, flags: Sequence[bool]) -> str:
    if all(flags):
        optimization_profile = "optimized"
    elif not any(flags):
        optimization_profile = "legacy"
    else:
        optimization_profile = "opts-" + "".join("1" if flag else "0" for flag in flags)
    safe_compile_mode = re.sub(r"[^A-Za-z0-9_.-]+", "-", compile_mode)
    return f"{safe_compile_mode}-{optimization_profile}"


def configure_checkpoint_compile_cache(
    checkpoint_path: str | pathlib.Path,
    *,
    compile_mode: str | None,
    flags: Sequence[bool],
    environ: MutableMapping[str, str] | None = None,
) -> pathlib.Path | None:
    """Configure a persistent Inductor cache stored beside a checkpoint."""
    if compile_mode is None:
        return None

    checkpoint_path = pathlib.Path(checkpoint_path)
    checkpoint_dir = checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent
    cache_dir = checkpoint_dir / ".torchinductor_cache" / compile_cache_profile(compile_mode, flags)
    cache_dir.mkdir(parents=True, exist_ok=True)

    environment = os.environ if environ is None else environ
    environment["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
    environment.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
    environment.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "1")
    return cache_dir

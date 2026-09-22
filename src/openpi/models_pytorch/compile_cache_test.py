import os

from openpi.models_pytorch import compile_cache


def test_configure_checkpoint_compile_cache_uses_checkpoint_directory(tmp_path):
    checkpoint = tmp_path / "step_10000" / "model.safetensors"
    checkpoint.parent.mkdir()
    environment = {}

    cache_dir = compile_cache.configure_checkpoint_compile_cache(
        checkpoint,
        compile_mode="max-autotune",
        flags=(True, True, True, True),
        environ=environment,
    )

    assert cache_dir == checkpoint.parent / ".torchinductor_cache" / "max-autotune-optimized"
    assert cache_dir.is_dir()
    assert environment["TORCHINDUCTOR_CACHE_DIR"] == str(cache_dir)
    assert environment["TORCHINDUCTOR_FX_GRAPH_CACHE"] == "1"
    assert environment["TORCHINDUCTOR_AUTOGRAD_CACHE"] == "1"


def test_compile_cache_profiles_keep_legacy_and_optimized_separate():
    assert compile_cache.compile_cache_profile("max-autotune", (False, False, False, False)) == "max-autotune-legacy"
    assert compile_cache.compile_cache_profile("max-autotune", (True, True, True, True)) == "max-autotune-optimized"


def test_no_compile_does_not_change_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", "existing")

    cache_dir = compile_cache.configure_checkpoint_compile_cache(
        tmp_path / "model.safetensors",
        compile_mode=None,
        flags=(True, True, True, True),
    )

    assert cache_dir is None
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == "existing"

import dataclasses
import pathlib

import torch

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.training import config as config_lib
from scripts import train_pytorch


def test_configured_weight_source_falls_back_to_jax_loader():
    config = config_lib.get_config("pi05_libero")

    assert train_pytorch.configured_weight_source(config) == "gs://openpi-assets/checkpoints/pi05_base/params"


def test_configured_weight_source_prefers_explicit_pytorch_path():
    config = dataclasses.replace(
        config_lib.get_config("pi0_libero"),
        pytorch_weight_path="/tmp/model",
    )

    assert train_pytorch.configured_weight_source(config) == "/tmp/model"


def test_checkpoint_root_accepts_params_directory():
    assert train_pytorch.checkpoint_root(pathlib.Path("/tmp/pi0/params")) == pathlib.Path("/tmp/pi0")
    assert train_pytorch.checkpoint_root(pathlib.Path("/tmp/pi0")) == pathlib.Path("/tmp/pi0")


def test_resolve_pytorch_weight_path_converts_jax_checkpoint(monkeypatch, tmp_path):
    source_root = tmp_path / "pi0_base"
    (source_root / "params").mkdir(parents=True)
    config = dataclasses.replace(
        config_lib.get_config("pi0_libero"),
        checkpoint_base_dir=str(tmp_path / "checkpoints"),
    )
    monkeypatch.setattr(train_pytorch.download_lib, "maybe_download", lambda _: source_root / "params")

    def fake_run(command, **kwargs):
        output_dir = pathlib.Path(command[command.index("--output_path") + 1])
        output_dir.mkdir(parents=True)
        (output_dir / "model.safetensors").touch()

    monkeypatch.setattr(train_pytorch.subprocess, "run", fake_run)

    result = train_pytorch.resolve_pytorch_weight_path(
        config,
        is_main=True,
        use_ddp=False,
        local_rank=0,
    )

    assert result == (tmp_path / "checkpoints" / ".converted_weights" / "pi0_libero" / "bfloat16" / "pi0_base")


def test_lora_trainable_parameters_use_float32():
    model_config = config_lib.get_config("pi05_libero_low_mem_finetune").model
    with torch.device("meta"):
        model = PI0Pytorch(model_config)

    train_pytorch.configure_trainable_parameters(model, model_config)

    assert all(parameter.dtype == torch.float32 for parameter in model.parameters() if parameter.requires_grad)
    assert any(parameter.dtype == torch.bfloat16 for parameter in model.parameters() if not parameter.requires_grad)

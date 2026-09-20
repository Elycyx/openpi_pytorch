from types import SimpleNamespace

import pytest
import torch
from torch import nn

from openpi.models_pytorch import checkpoint
from openpi.models_pytorch import model_pytorch
from openpi.models_pytorch import pi0_pytorch


class _TinyImageEncoder(nn.Module):
    def __init__(self, *, num_classes: int, **kwargs):
        super().__init__()
        self.proj = nn.Linear(3, num_classes)
        self.encoder = SimpleNamespace(gradient_checkpointing=False, gradient_checkpointing_use_reentrant=False)

    def forward(self, image: torch.Tensor):
        tokens = self.proj(image.mean(dim=(1, 2)))[:, None, :]
        return tokens.expand(-1, 4, -1), None


def _config(*, pi05: bool):
    return SimpleNamespace(
        action_dim=8,
        action_horizon=3,
        max_token_len=5,
        dtype="float32",
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        pi05=pi05,
        pytorch_compile_mode=None,
    )


def _observation(batch_size: int = 2):
    return model_pytorch.Observation(
        images={key: torch.randn(batch_size, 16, 16, 3) for key in model_pytorch.IMAGE_KEYS},
        image_masks={key: torch.ones(batch_size, dtype=torch.bool) for key in model_pytorch.IMAGE_KEYS},
        state=torch.randn(batch_size, 8),
        tokenized_prompt=torch.randint(0, 128, (batch_size, 5)),
        tokenized_prompt_mask=torch.ones(batch_size, 5, dtype=torch.bool),
    )


@pytest.mark.parametrize("pi05", [False, True])
def test_loss_and_sampling(monkeypatch, pi05):
    monkeypatch.setattr(pi0_pytorch.siglip, "SigLIPViT", _TinyImageEncoder)
    monkeypatch.setattr(pi0_pytorch.gemma, "PALIGEMMA_VOCAB_SIZE", 128)
    model = pi0_pytorch.PI0Pytorch(_config(pi05=pi05))
    observation = _observation()
    actions = torch.randn(2, 3, 8)

    loss = model.compute_loss(
        observation,
        actions,
        noise=torch.zeros_like(actions),
        time=torch.full((2,), 0.5),
    )
    sampled = model.sample_actions("cpu", observation, noise=torch.zeros_like(actions), num_steps=2)

    assert loss.shape == actions.shape
    assert sampled.shape == actions.shape
    assert torch.isfinite(loss).all()
    assert torch.isfinite(sampled).all()


def test_old_checkpoint_conversion_preserves_shared_embedder():
    embedder = torch.randn(128, 64)
    converted = checkpoint.old_to_new_state_dict(
        {
            "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight": embedder,
            "action_in_proj.weight": torch.randn(64, 8),
        }
    )

    assert converted["llm.embedder.embedding.weight"] is embedder
    assert "action_in_proj.weight" in converted


def test_selective_bfloat16_keeps_openpi_fp32_islands(monkeypatch):
    monkeypatch.setattr(pi0_pytorch.siglip, "SigLIPViT", _TinyImageEncoder)
    monkeypatch.setattr(pi0_pytorch.gemma, "PALIGEMMA_VOCAB_SIZE", 128)
    config = _config(pi05=False)
    config.dtype = "bfloat16"
    model = pi0_pytorch.PI0Pytorch(config)

    assert model.llm.layers[0].attn.q_proj[0].weight.dtype == torch.bfloat16
    assert model.llm.layers[0].pre_attention_norms[0].scale.dtype == torch.float32
    assert model.action_out_proj.weight.dtype == torch.float32


def test_lora_model_accepts_full_checkpoint_attention_weights():
    config = _config(pi05=False)
    config.paligemma_variant = "gemma_2b_lora"
    config.action_expert_variant = "gemma_300m_lora"
    with torch.device("meta"):
        model = pi0_pytorch.PI0Pytorch(config)
    state_dict = {}
    for expert_index, gemma_config in enumerate(model.llm.configs):
        prefix = "llm.layers.0.attn."
        state_dict[f"{prefix}q_proj.{expert_index}.weight"] = torch.empty(
            gemma_config.num_heads * gemma_config.head_dim,
            gemma_config.width,
            device="meta",
        )
        state_dict[f"{prefix}k_proj.{expert_index}.weight"] = torch.empty(
            gemma_config.num_kv_heads * gemma_config.head_dim,
            gemma_config.width,
            device="meta",
        )
        state_dict[f"{prefix}v_proj.{expert_index}.weight"] = torch.empty(
            gemma_config.num_kv_heads * gemma_config.head_dim,
            gemma_config.width,
            device="meta",
        )
        state_dict[f"{prefix}o_proj.{expert_index}.weight"] = torch.empty(
            gemma_config.width,
            gemma_config.num_heads * gemma_config.head_dim,
            device="meta",
        )

    adapted = model.adapt_attention_weights_for_lora(state_dict)

    assert "llm.layers.0.attn.q_proj.0.w" in adapted
    assert "llm.layers.0.attn.k_proj.0.w" in adapted
    assert "llm.layers.0.attn.o_proj.1.w" in adapted


def test_lora_attention_forward():
    lora_config = pi0_pytorch.gemma.lora.LoRAConfig(rank=4, alpha=4.0)
    config = pi0_pytorch.gemma.Config(
        width=32,
        depth=1,
        mlp_dim=64,
        num_heads=4,
        num_kv_heads=1,
        head_dim=8,
        lora_configs={"attn": lora_config},
    )
    attention = pi0_pytorch.gemma.Attention([config, config])
    inputs = [torch.randn(2, 3, 32), torch.randn(2, 2, 32)]
    positions = torch.arange(5).expand(2, -1)
    mask = torch.ones(2, 1, 5, 5, dtype=torch.bool)

    outputs, _ = attention(inputs, positions, mask)

    assert outputs[0].shape == inputs[0].shape
    assert outputs[1].shape == inputs[1].shape

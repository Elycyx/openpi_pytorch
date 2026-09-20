"""Checkpoint compatibility for the JAX-aligned PyTorch Pi0 implementation."""

from __future__ import annotations

import torch


def old_to_new_state_dict(
    old_sd: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert an old-format state dict to the new ``Pi0`` layout.

    Handles key renaming and weight transformations:
      - SigLIP Q/K/V concat -> in_proj_weight/bias
      - LLM MLP gate/up transpose+stack -> w_gating (2, features, hidden_dim)
      - LLM MLP down transpose -> w_linear
      - Action-expert RMSNorm: Pi0 ``*.weight`` -> ``*.1.scale``;
        Pi0.5 AdaRMS ``*.dense.*`` -> ``*.1.ada_modulation.*``
    """
    openpi_pytorch_state_dict = old_sd
    openpi_rlinf_state_dict: dict[str, torch.Tensor] = {}

    openpi_pytorch_siglip = "paligemma_with_expert.paligemma.model.vision_tower.vision_model."

    # Stem
    for suffix in (".weight", ".bias"):
        source_key = openpi_pytorch_siglip + "embeddings.patch_embedding" + suffix
        if source_key in openpi_pytorch_state_dict:
            openpi_rlinf_state_dict["img.stem" + suffix] = openpi_pytorch_state_dict[source_key]

    # OpenPI stores this as an (num_patches, width) nn.Embedding weight; RLinf
    # holds a (1, num_patches, width) parameter, so add the broadcast dimension.
    source_key = openpi_pytorch_siglip + "embeddings.position_embedding.weight"
    if source_key in openpi_pytorch_state_dict:
        position_embedding = openpi_pytorch_state_dict[source_key]
        openpi_rlinf_state_dict["img.pos_embedding"] = (
            position_embedding.unsqueeze(0) if position_embedding.dim() == 2 else position_embedding
        )

    # Encoder layers (0..26)
    for layer_index in range(27):
        source_prefix = f"{openpi_pytorch_siglip}encoder.layers.{layer_index}."
        target_prefix = f"img.encoder.layers.{layer_index}."

        for source_name, target_name in [
            ("layer_norm1", "norm1"),
            ("layer_norm2", "norm2"),
        ]:
            for suffix in (".weight", ".bias"):
                source_key = f"{source_prefix}{source_name}{suffix}"
                if source_key in openpi_pytorch_state_dict:
                    openpi_rlinf_state_dict[f"{target_prefix}{target_name}{suffix}"] = openpi_pytorch_state_dict[
                        source_key
                    ]

        qkv_weights = []
        qkv_biases = []
        for projection in ("q_proj", "k_proj", "v_proj"):
            weight_key = f"{source_prefix}self_attn.{projection}.weight"
            bias_key = f"{source_prefix}self_attn.{projection}.bias"
            if weight_key in openpi_pytorch_state_dict:
                qkv_weights.append(openpi_pytorch_state_dict[weight_key])
            if bias_key in openpi_pytorch_state_dict:
                qkv_biases.append(openpi_pytorch_state_dict[bias_key])
        if qkv_weights:
            openpi_rlinf_state_dict[f"{target_prefix}attn.in_proj_weight"] = torch.cat(qkv_weights, dim=0)
        if qkv_biases:
            openpi_rlinf_state_dict[f"{target_prefix}attn.in_proj_bias"] = torch.cat(qkv_biases, dim=0)

        for suffix in (".weight", ".bias"):
            source_key = f"{source_prefix}self_attn.out_proj{suffix}"
            if source_key in openpi_pytorch_state_dict:
                openpi_rlinf_state_dict[f"{target_prefix}attn.out_proj{suffix}"] = openpi_pytorch_state_dict[source_key]

        for name in ("fc1", "fc2"):
            for suffix in (".weight", ".bias"):
                source_key = f"{source_prefix}mlp.{name}{suffix}"
                if source_key in openpi_pytorch_state_dict:
                    openpi_rlinf_state_dict[f"{target_prefix}mlp.{name}{suffix}"] = openpi_pytorch_state_dict[
                        source_key
                    ]

    # Post layernorm
    for suffix in (".weight", ".bias"):
        source_key = openpi_pytorch_siglip + "post_layernorm" + suffix
        if source_key in openpi_pytorch_state_dict:
            openpi_rlinf_state_dict["img.encoder.norm" + suffix] = openpi_pytorch_state_dict[source_key]

    # Multi-modal projector
    for suffix in (".weight", ".bias"):
        source_key = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear" + suffix
        if source_key in openpi_pytorch_state_dict:
            openpi_rlinf_state_dict["img.head" + suffix] = openpi_pytorch_state_dict[source_key]

    # PaliGemma LLM (expert 0)
    pali_llm = "paligemma_with_expert.paligemma.model.language_model."
    for layer_index in range(18):
        source_prefix = f"{pali_llm}layers.{layer_index}."
        target_prefix = f"llm.layers.{layer_index}."

        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            source_key = f"{source_prefix}self_attn.{projection}.weight"
            if source_key in openpi_pytorch_state_dict:
                openpi_rlinf_state_dict[f"{target_prefix}attn.{projection}.0.weight"] = openpi_pytorch_state_dict[
                    source_key
                ]

        gate_key = f"{source_prefix}mlp.gate_proj.weight"
        up_key = f"{source_prefix}mlp.up_proj.weight"
        if gate_key in openpi_pytorch_state_dict and up_key in openpi_pytorch_state_dict:
            gate_transposed = openpi_pytorch_state_dict[gate_key].T.contiguous()
            up_transposed = openpi_pytorch_state_dict[up_key].T.contiguous()
            openpi_rlinf_state_dict[f"{target_prefix}mlps.0.w_gating"] = torch.stack(
                [gate_transposed, up_transposed], dim=0
            )

        down_key = f"{source_prefix}mlp.down_proj.weight"
        if down_key in openpi_pytorch_state_dict:
            openpi_rlinf_state_dict[f"{target_prefix}mlps.0.w_linear"] = openpi_pytorch_state_dict[
                down_key
            ].T.contiguous()

        for source_name, target_name in [
            ("input_layernorm", "pre_attention_norms"),
            ("post_attention_layernorm", "pre_ffw_norms"),
        ]:
            source_key = f"{source_prefix}{source_name}.weight"
            if source_key in openpi_pytorch_state_dict:
                openpi_rlinf_state_dict[f"{target_prefix}{target_name}.0.scale"] = openpi_pytorch_state_dict[source_key]

    source_key = pali_llm + "norm.weight"
    if source_key in openpi_pytorch_state_dict:
        openpi_rlinf_state_dict["llm.final_norms.0.scale"] = openpi_pytorch_state_dict[source_key]

    # Gemma action expert (expert 1)
    gemma_expert = "paligemma_with_expert.gemma_expert.model."
    for layer_index in range(18):
        source_prefix = f"{gemma_expert}layers.{layer_index}."
        target_prefix = f"llm.layers.{layer_index}."

        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            source_key = f"{source_prefix}self_attn.{projection}.weight"
            if source_key in openpi_pytorch_state_dict:
                openpi_rlinf_state_dict[f"{target_prefix}attn.{projection}.1.weight"] = openpi_pytorch_state_dict[
                    source_key
                ]

        gate_key = f"{source_prefix}mlp.gate_proj.weight"
        up_key = f"{source_prefix}mlp.up_proj.weight"
        if gate_key in openpi_pytorch_state_dict and up_key in openpi_pytorch_state_dict:
            gate_transposed = openpi_pytorch_state_dict[gate_key].T.contiguous()
            up_transposed = openpi_pytorch_state_dict[up_key].T.contiguous()
            openpi_rlinf_state_dict[f"{target_prefix}mlps.1.w_gating"] = torch.stack(
                [gate_transposed, up_transposed], dim=0
            )

        down_key = f"{source_prefix}mlp.down_proj.weight"
        if down_key in openpi_pytorch_state_dict:
            openpi_rlinf_state_dict[f"{target_prefix}mlps.1.w_linear"] = openpi_pytorch_state_dict[
                down_key
            ].T.contiguous()

        for source_name, target_name in [
            ("input_layernorm", "pre_attention_norms"),
            ("post_attention_layernorm", "pre_ffw_norms"),
        ]:
            # Pi0.5: AdaRMS stored as ``*.dense.{weight,bias}``.
            for suffix in (".weight", ".bias"):
                source_key = f"{source_prefix}{source_name}.dense{suffix}"
                if source_key in openpi_pytorch_state_dict:
                    openpi_rlinf_state_dict[f"{target_prefix}{target_name}.1.ada_modulation{suffix}"] = (
                        openpi_pytorch_state_dict[source_key]
                    )
            # Pi0: regular RMSNorm stored as ``*.weight``.
            source_key = f"{source_prefix}{source_name}.weight"
            if source_key in openpi_pytorch_state_dict:
                openpi_rlinf_state_dict[f"{target_prefix}{target_name}.1.scale"] = openpi_pytorch_state_dict[source_key]

    for suffix in (".weight", ".bias"):
        source_key = gemma_expert + "norm.dense" + suffix
        if source_key in openpi_pytorch_state_dict:
            openpi_rlinf_state_dict["llm.final_norms.1.ada_modulation" + suffix] = openpi_pytorch_state_dict[source_key]
    source_key = gemma_expert + "norm.weight"
    if source_key in openpi_pytorch_state_dict:
        openpi_rlinf_state_dict["llm.final_norms.1.scale"] = openpi_pytorch_state_dict[source_key]

    # The RLinf shared token embedder is PaliGemma's embedding (tied with
    # ``paligemma.lm_head``, width = PaliGemma width, e.g. 2048). The action
    # expert's 1024-wide head must not be used as the embedder.
    embedder_key = next(
        (
            key
            for key in (
                "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight",
                "paligemma_with_expert.paligemma.lm_head.weight",
                "paligemma_with_expert.gemma_expert.lm_head.weight",
            )
            if key in openpi_pytorch_state_dict
        ),
        None,
    )
    if embedder_key is not None:
        openpi_rlinf_state_dict["llm.embedder.embedding.weight"] = openpi_pytorch_state_dict[embedder_key]

    # Action head (same names in both layouts)
    for key in openpi_pytorch_state_dict:
        if key.startswith(
            (
                "action_in_proj",
                "action_out_proj",
                "time_mlp_",
                "state_proj",
                "action_time_mlp_",
                "pointnet.",
            )
        ):
            openpi_rlinf_state_dict[key] = openpi_pytorch_state_dict[key]

    return openpi_rlinf_state_dict

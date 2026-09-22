"""Pi0 model for PyTorch, aligned with the official JAX ``models/pi0.py`` implementation.

The network implementation is ported from RLinf while preserving openpi's existing PyTorch public API and checkpoint format compatibility.
"""

from __future__ import annotations

import einops
import torch
import torch.nn as nn
import torch.nn.functional as functional

from openpi.models_pytorch import checkpoint
from openpi.models_pytorch import gemma
from openpi.models_pytorch import model_pytorch as model
from openpi.models_pytorch import siglip
from openpi.models_pytorch.utils import _str_to_dtype


def make_attn_mask(input_mask: torch.Tensor, mask_ar: torch.Tensor) -> torch.Tensor:
    """Create attention mask from input mask and autoregressive mask.

    Tokens can attend to valid input tokens which have a cumulative mask_ar
    smaller or equal to theirs.

    Args:
        input_mask: bool[B, N] - true if token is valid
        mask_ar: bool[N] - true where next token starts a new autoregressive block
    """
    mask_ar = mask_ar.expand(input_mask.shape[0], -1)
    cumsum = torch.cumsum(mask_ar.int(), dim=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return torch.logical_and(attn_mask, valid_mask)


def posemb_sincos(
    pos: torch.Tensor,
    embedding_dim: int,
    min_period: float = 4e-3,
    max_period: float = 4.0,
) -> torch.Tensor:
    """Sine-cosine positional embedding for scalar positions.

    Args:
        pos: (B,) float positions
        embedding_dim: output dimension (must be even)

    Returns:
        (B, embedding_dim) positional embedding
    """
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = torch.linspace(0.0, 1.0, embedding_dim // 2, device=pos.device, dtype=torch.float32)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = torch.einsum("i,j->ij", pos.float(), 1.0 / period * 2 * torch.pi)
    # Match JAX which keeps posemb in float32. However, PT Linear does not support
    # mixed float32/bf16 matmul, so cast back to the model's embed_dtype.
    # The caller should upcast to float32 if needed for high-precision ops.
    return torch.cat([torch.sin(sinusoid_input), torch.cos(sinusoid_input)], dim=-1).to(pos.dtype)


class PI0Pytorch(model.BaseModel):
    """Pi0 flow-matching model: network assembly plus RLinf SFT forward."""

    def __init__(self, config):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.config = config
        self.pi05 = config.pi05
        self.embed_dtype = _str_to_dtype(config.dtype)
        self.optimize_prefix_cache = getattr(config, "pytorch_optimize_prefix_cache", True)
        self.discard_suffix_cache = getattr(config, "pytorch_discard_suffix_cache", True)
        self.precompute_suffix_metadata = getattr(config, "pytorch_precompute_suffix_metadata", True)
        self.cache_time_embedding_frequencies = getattr(config, "pytorch_cache_time_embedding_frequencies", True)

        paligemma_config = gemma.get_config(config.paligemma_variant)
        action_expert_config = gemma.get_config(config.action_expert_variant)
        self.llm = gemma.Module(
            configs=[paligemma_config, action_expert_config],
            embed_dtype=config.dtype,
            adarms=[False, config.pi05],
            use_gradient_checkpointing=False,
        )
        self.img = siglip.SigLIPViT(
            variant="So400m/14",
            pool_type="none",
            num_classes=paligemma_config.width,
            use_gradient_checkpointing=False,
            dtype_mm=config.dtype,
        )

        action_expert_width = action_expert_config.width
        self.action_in_proj = nn.Linear(config.action_dim, action_expert_width)
        if config.pi05:
            self.time_mlp_in = nn.Linear(action_expert_width, action_expert_width)
            self.time_mlp_out = nn.Linear(action_expert_width, action_expert_width)
        else:
            self.state_proj = nn.Linear(config.action_dim, action_expert_width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_width, action_expert_width)
            self.action_time_mlp_out = nn.Linear(action_expert_width, action_expert_width)
        self.action_out_proj = nn.Linear(action_expert_width, config.action_dim)

        fraction = torch.linspace(0.0, 1.0, action_expert_width // 2, dtype=torch.float32)
        period = 4e-3 * (4.0 / 4e-3) ** fraction
        self.register_buffer(
            "time_embedding_frequencies",
            1.0 / period * 2 * torch.pi,
            persistent=False,
        )

        self._init_weights()
        self.to_bfloat16_for_selected_params(config.dtype)
        torch.set_float32_matmul_precision("high")
        compile_mode = getattr(config, "pytorch_compile_mode", None)
        if compile_mode is not None:
            self.sample_actions = torch.compile(self.sample_actions, mode=compile_mode)

    def load_state_dict(self, state_dict, strict=True, assign=False):  # noqa: FBT002
        if any(key.startswith("paligemma_with_expert.") for key in state_dict):
            state_dict = checkpoint.old_to_new_state_dict(dict(state_dict))
        state_dict = self.adapt_attention_weights_for_lora(dict(state_dict))
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def adapt_attention_weights_for_lora(self, state_dict):
        for layer_index, layer in enumerate(self.llm.layers):
            for expert_index, config in enumerate(self.llm.configs):
                if not isinstance(layer.attn.q_proj[expert_index], gemma.lora.Einsum):
                    continue
                prefix = f"llm.layers.{layer_index}.attn."
                q_key = f"{prefix}q_proj.{expert_index}.weight"
                o_key = f"{prefix}o_proj.{expert_index}.weight"
                if q_key in state_dict:
                    q_weight = state_dict.pop(q_key)
                    state_dict[f"{prefix}q_proj.{expert_index}.w"] = q_weight.reshape(
                        config.num_heads, config.head_dim, config.width
                    ).permute(0, 2, 1)
                if layer.attn.k_proj[expert_index] is not None:
                    k_key = f"{prefix}k_proj.{expert_index}.weight"
                    v_key = f"{prefix}v_proj.{expert_index}.weight"
                    if k_key in state_dict and v_key in state_dict:
                        k_weight = (
                            state_dict.pop(k_key)
                            .reshape(config.num_kv_heads, config.head_dim, config.width)
                            .permute(0, 2, 1)
                        )
                        v_weight = (
                            state_dict.pop(v_key)
                            .reshape(config.num_kv_heads, config.head_dim, config.width)
                            .permute(0, 2, 1)
                        )
                        state_dict[f"{prefix}k_proj.{expert_index}.w"] = torch.stack([k_weight, v_weight])
                if o_key in state_dict:
                    o_weight = state_dict.pop(o_key)
                    state_dict[f"{prefix}o_proj.{expert_index}.w"] = o_weight.reshape(
                        config.width, config.num_heads, config.head_dim
                    ).permute(1, 2, 0)
        return state_dict

    def _init_weights(self):
        """Initialize projection weights."""
        nn.init.normal_(self.action_in_proj.weight, std=0.02)
        nn.init.zeros_(self.action_in_proj.bias)
        nn.init.normal_(self.action_out_proj.weight, std=0.02)
        nn.init.zeros_(self.action_out_proj.bias)

        if self.pi05:
            nn.init.normal_(self.time_mlp_in.weight, std=0.02)
            nn.init.zeros_(self.time_mlp_in.bias)
            nn.init.normal_(self.time_mlp_out.weight, std=0.02)
            nn.init.zeros_(self.time_mlp_out.bias)
        else:
            nn.init.normal_(self.state_proj.weight, std=0.02)
            nn.init.zeros_(self.state_proj.bias)
            nn.init.normal_(self.action_time_mlp_in.weight, std=0.02)
            nn.init.zeros_(self.action_time_mlp_in.bias)
            nn.init.normal_(self.action_time_mlp_out.weight, std=0.02)
            nn.init.zeros_(self.action_time_mlp_out.bias)

    def embed_prefix(self, obs: model.Observation) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed the prefix (images + language + optional point cloud).

        Returns:
            tokens: (B, S, emb_dim) embedded tokens
            input_mask: (B, S) mask of valid tokens
            ar_mask: (S,) autoregressive mask (all False for prefix)
        """
        tokens = []
        input_mask = []
        ar_mask = []

        # Embed images through SigLIP in IMAGE_KEYS order (not dict iteration
        # order). Official OpenPI preprocess rebuilds cameras this way, then
        # concatenates ``list(observation.images.values())``.
        image_names = [name for name in model.IMAGE_KEYS if name in obs.images]
        image_names.extend(name for name in obs.images if name not in model.IMAGE_KEYS)
        for name in image_names:
            image_tokens, _ = self.img(obs.images[name])  # (B, num_patches, width)
            tokens.append(image_tokens)

            # Image tokens use bidirectional attention
            input_mask.append(einops.repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1]))
            ar_mask += [False] * image_tokens.shape[1]

        # Add language tokens
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.llm.embed(obs.tokenized_prompt)
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * tokenized_inputs.shape[1]

        tokens = torch.cat(tokens, dim=1)
        input_mask = torch.cat(input_mask, dim=1)
        ar_mask = torch.tensor(ar_mask, device=tokens.device)
        return tokens, input_mask, ar_mask

    def embed_suffix_tokens(
        self,
        obs: model.Observation,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        *,
        cache_time_embedding_frequencies: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Embed suffix tokens without constructing attention metadata.

        Args:
            obs: observation
            noisy_actions: (B, action_horizon, action_dim)
            timestep: (B,) float timestep values

        Returns:
            tokens: (B, S, emb_dim)
            adarms_cond: (B, emb_dim) or None
        """
        tokens = []

        if not self.pi05:
            # Official PI0Pytorch: upcast state only when state_proj is fp32.
            state = obs.state
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)
            state_token = self.state_proj(state)[:, None, :]
            tokens.append(state_token)

        # Embed actions
        action_tokens = self.action_in_proj(noisy_actions)

        # Time embedding
        if cache_time_embedding_frequencies:
            sinusoid_input = torch.einsum("i,j->ij", timestep.float(), self.time_embedding_frequencies)
            time_emb = torch.cat([torch.sin(sinusoid_input), torch.cos(sinusoid_input)], dim=-1).to(timestep.dtype)
        else:
            time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)

        if self.pi05:
            # Time MLP for adaRMS conditioning
            time_emb = self.time_mlp_in(time_emb)
            time_emb = functional.silu(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = functional.silu(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # Mix timestep + action through MLP
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = torch.cat([action_tokens, time_tokens], dim=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = functional.silu(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None

        tokens.append(action_expert_tokens)
        tokens = torch.cat(tokens, dim=1)
        return tokens, adarms_cond

    def make_suffix_masks(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        suffix_len = self.action_horizon + (0 if self.pi05 else 1)
        input_mask = torch.ones(batch_size, suffix_len, dtype=torch.bool, device=device)

        # Build ar_mask with correct length matching input_mask.shape[1]
        ar_mask = torch.zeros(input_mask.shape[1], dtype=torch.bool, device=device)
        if not self.pi05:
            ar_mask[:2] = True
        else:
            ar_mask[0] = True

        return input_mask, ar_mask

    def embed_suffix(
        self,
        obs: model.Observation,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        *,
        cache_time_embedding_frequencies: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Embed suffix tokens and construct their attention metadata."""
        tokens, adarms_cond = self.embed_suffix_tokens(
            obs,
            noisy_actions,
            timestep,
            cache_time_embedding_frequencies=cache_time_embedding_frequencies,
        )
        input_mask, ar_mask = self.make_suffix_masks(noisy_actions.shape[0], tokens.device)
        return tokens, input_mask, ar_mask, adarms_cond

    def compute_loss(
        self,
        observation: model.Observation,
        actions: torch.Tensor,
        *,
        train: bool = False,
        rng: torch.Generator | None = None,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute flow matching loss.

        Returns:
            loss: (B, action_horizon, action_dim) per-element MSE
        """
        batch_size = actions.shape[0]
        device = actions.device

        # Preprocess first (requires float32 for image ops), then cast the
        # observation to embed_dtype for Gemma / SigLIP. Keep actions in
        # fp32 like the RL sampler: action / time projections stay fp32.
        observation = model.Observation.from_observation_like(observation)
        observation = model.preprocess_observation(observation, train=train, rng=rng)

        observation = model.observation_to_dtype(observation, self.embed_dtype)
        actions = actions.to(dtype=torch.float32)
        dtype = actions.dtype

        # Sample noise and time (or use provided values for reproducibility)
        if noise is None:
            noise = torch.randn(actions.shape, device=device, dtype=dtype, generator=rng)
        else:
            noise = noise.to(dtype=dtype)
        if time is None:
            time = (
                torch.distributions.Beta(torch.tensor(1.5), torch.tensor(1.0))
                .sample((batch_size,))
                .to(device=device, dtype=dtype)
            )
            time = time * 0.999 + 0.001
        else:
            time = time.to(dtype=dtype)
        time_expanded = time[:, None, None]

        # Flow matching interpolation
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # One forward pass for prefix + suffix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)

        input_mask = torch.cat([prefix_mask, suffix_mask], dim=1)
        ar_mask = torch.cat([prefix_ar_mask, suffix_ar_mask], dim=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = torch.cumsum(input_mask.int(), dim=1) - 1

        prefix_out, suffix_out = self.llm(
            [prefix_tokens, suffix_tokens],
            positions=positions,
            mask=attn_mask,
            adarms_cond=[None, adarms_cond],
        )[0]

        v_t = self.velocity_from_suffix(suffix_out[:, -self.action_horizon :])

        return torch.square(v_t - u_t)

    def build_prefix_cache(self, observation: model.Observation) -> tuple[torch.Tensor, torch.Tensor, tuple]:
        """Embed prefix tokens and run one LLM pass to build the KV cache.

        The caller is responsible for preprocessing the observation (image
        resize/pad, mask defaults) — this method only consumes the prepared
        observation so it can be shared between the eval Euler sampler and the
        RL train-time forward where the observation has already been built.

        Returns:
            prefix_out:  (B, prefix_len, paligemma_width) paligemma-side hidden states.
            prefix_mask: (B, prefix_len) bool mask of valid prefix positions.
            kv_cache:    per-layer KV cache to feed into subsequent suffix passes.
        """
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = torch.cumsum(prefix_mask.int(), dim=1) - 1
        outputs, kv_cache = self.llm(
            [prefix_tokens, None],
            positions=positions,
            mask=prefix_attn_mask,
            cache_only_last_layer=self.optimize_prefix_cache,
        )
        return outputs[0], prefix_mask, kv_cache

    def build_suffix_attention_metadata(
        self,
        prefix_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build suffix attention mask and positions shared by every Euler step."""
        suffix_mask, suffix_ar_mask = self.make_suffix_masks(prefix_mask.shape[0], prefix_mask.device)
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_to_suffix_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_mask.shape[1])
        full_attn_mask = torch.cat([prefix_to_suffix_mask, suffix_attn_mask], dim=-1)
        suffix_positions = torch.sum(prefix_mask, dim=-1)[:, None] + torch.cumsum(suffix_mask.int(), dim=-1) - 1
        return full_attn_mask, suffix_positions

    def run_suffix(
        self,
        observation: model.Observation,
        x_t: torch.Tensor,
        t_tensor: torch.Tensor,
        kv_cache: tuple,
        prefix_mask: torch.Tensor,
        suffix_attention_metadata: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """One suffix forward pass (action expert) given the prefix KV cache.

        Returns the action-expert hidden states sliced to the last
        ``action_horizon`` positions: (B, action_horizon, action_expert_width).
        """
        if suffix_attention_metadata is None:
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation,
                x_t,
                t_tensor,
                cache_time_embedding_frequencies=self.cache_time_embedding_frequencies,
            )
            suffix_len = suffix_tokens.shape[1]
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_to_suffix_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_len)
            full_attn_mask = torch.cat([prefix_to_suffix_mask, suffix_attn_mask], dim=-1)
            suffix_positions = torch.sum(prefix_mask, dim=-1)[:, None] + torch.cumsum(suffix_mask.int(), dim=-1) - 1
        else:
            suffix_tokens, adarms_cond = self.embed_suffix_tokens(
                observation,
                x_t,
                t_tensor,
                cache_time_embedding_frequencies=self.cache_time_embedding_frequencies,
            )
            full_attn_mask, suffix_positions = suffix_attention_metadata
        outputs, _ = self.llm(
            [None, suffix_tokens],
            positions=suffix_positions,
            mask=full_attn_mask,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
            return_kv_cache=not self.discard_suffix_cache,
        )
        # Official PI0Pytorch casts suffix hidden states to fp32 before
        # ``action_out_proj`` / the value head.
        return outputs[1][:, -self.action_horizon :].to(dtype=torch.float32)

    def velocity_from_suffix(self, suffix_out_act: torch.Tensor) -> torch.Tensor:
        """Project action-expert hidden states to a velocity prediction v_t."""
        return self.action_out_proj(suffix_out_act.to(dtype=torch.float32))

    def to_bfloat16_for_selected_params(self, precision: str = "bfloat16") -> None:
        """OpenPI ``PaliGemmaWithExpertModel.to_bfloat16_for_selected_params``.

        OpenPI calls ``self.to(dtype)`` on ``paligemma_with_expert`` only
        (SigLIP + PaliGemma Gemma + action-expert Gemma), then restores a
        subset of those weights to fp32. Action / value heads live outside
        that module and must not be converted — so this uses ``llm`` +
        ``img``, never ``self.to()``.
        """
        # Same branch order as OpenPI: bf16 converts then falls through to
        # the fp32 restore; float32 converts and returns.
        if precision == "bfloat16":
            self.llm.to(dtype=torch.bfloat16)
            self.img.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.llm.to(dtype=torch.float32)
            self.img.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        # 1:1 with OpenPI's substring list. Names come from
        # ``openpi_pytorch_to_openpi_rlinf``:
        #   patch_embedding.weight/bias  -> img.stem.weight/bias
        #   position_embedding.weight    -> img.pos_embedding
        #   input_layernorm              -> pre_attention_norms
        #   post_attention_layernorm     -> pre_ffw_norms
        #   model.norm                   -> final_norms
        #     (language_model.norm + gemma_expert.model.norm)
        params_to_keep_float32 = [
            "img.stem.weight",
            "img.stem.bias",
            "img.pos_embedding",
            "pre_attention_norms",
            "pre_ffw_norms",
            "final_norms",
        ]

        # OpenPI iterates paligemma_with_expert.named_parameters(); llm + img
        # are that module. Do not scan action/value heads.
        for name, param in self.named_parameters():
            if not name.startswith(("llm.", "img.")):
                continue
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    @torch.no_grad()
    def sample_actions(
        self,
        device,
        observation,
        noise: torch.Tensor | None = None,
        num_steps: int = 10,
        rng: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample actions using Euler ODE solver.

        Args:
            observation: input observation
            num_steps: number of ODE solver steps
            noise: optional initial noise of shape (B, action_horizon, action_dim)
            rng: random generator

        Returns:
            actions: (B, action_horizon, action_dim)
        """
        del device
        observation = model.Observation.from_observation_like(observation)
        observation = model.preprocess_observation(observation, train=False)
        observation = model.observation_to_dtype(observation, self.embed_dtype)

        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        device = observation.state.device

        if noise is None:
            noise = torch.randn(batch_size, self.action_horizon, self.action_dim, device=device, generator=rng)

        _, prefix_mask, kv_cache = self.build_prefix_cache(observation)
        suffix_attention_metadata = None
        if self.precompute_suffix_metadata:
            suffix_attention_metadata = self.build_suffix_attention_metadata(prefix_mask)

        x_t = noise
        t = 1.0

        # Euler integration
        while t >= -dt / 2:
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.float32)
            suffix_out_act = self.run_suffix(
                observation,
                x_t,
                t_tensor,
                kv_cache,
                prefix_mask,
                suffix_attention_metadata,
            )
            v_t = self.velocity_from_suffix(suffix_out_act)
            x_t = x_t + dt * v_t
            t = t + dt

        return x_t

    def forward(self, observation, actions, noise=None, time=None):
        return self.compute_loss(
            observation,
            actions,
            train=True,
            noise=noise,
            time=time,
        )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        kwargs = gradient_checkpointing_kwargs or {}
        use_reentrant = kwargs.get("use_reentrant", False)
        self.llm.gradient_checkpointing = True
        self.llm.gradient_checkpointing_use_reentrant = use_reentrant
        self.img.encoder.gradient_checkpointing = True
        self.img.encoder.gradient_checkpointing_use_reentrant = use_reentrant

    def gradient_checkpointing_disable(self):
        self.llm.gradient_checkpointing = False
        self.img.encoder.gradient_checkpointing = False

    def is_gradient_checkpointing_enabled(self):
        return self.llm.gradient_checkpointing or self.img.encoder.gradient_checkpointing

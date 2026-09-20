"""Base model classes for PyTorch models, aligned with JAX models/model.py."""

from __future__ import annotations

import abc
from collections.abc import Mapping, Sequence
import dataclasses
import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional

logger = logging.getLogger("openpi")

# Image keys expected by the model
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)

IMAGE_RESOLUTION = (224, 224)


def resize_with_pad_torch(
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Resize images to target size with padding to preserve aspect ratio.

    Args:
        images: (..., H, W, C) float32 tensor in [-1, 1] or uint8 in [0, 255]
        height: target height
        width: target width
        mode: interpolation mode

    Returns:
        Resized and padded tensor
    """
    if images.dim() == 3:
        images = images.unsqueeze(0)
        squeeze_batch = True
    else:
        squeeze_batch = False

    # (B, H, W, C) -> (B, C, H, W)
    images = images.permute(0, 3, 1, 2)
    batch_size, channels, cur_h, cur_w = images.shape

    ratio = max(cur_w / width, cur_h / height)
    resized_h = int(cur_h / ratio)
    resized_w = int(cur_w / ratio)

    resized = functional.interpolate(images, size=(resized_h, resized_w), mode=mode, align_corners=False)

    if images.dtype == torch.uint8:
        resized = torch.round(resized).clamp(0, 255).to(torch.uint8)
    elif images.dtype in (torch.float32, torch.float16, torch.bfloat16):
        resized = resized.clamp(-1.0, 1.0)

    pad_h0, rem_h = divmod(height - resized_h, 2)
    pad_h1 = pad_h0 + rem_h
    pad_w0, rem_w = divmod(width - resized_w, 2)
    pad_w1 = pad_w0 + rem_w

    fill_val = 0 if images.dtype == torch.uint8 else -1.0
    padded = functional.pad(resized, (pad_w0, pad_w1, pad_h0, pad_h1), mode="constant", value=fill_val)

    # (B, C, H, W) -> (B, H, W, C)
    padded = padded.permute(0, 2, 3, 1)

    if squeeze_batch:
        padded = padded.squeeze(0)
    return padded


@dataclasses.dataclass
class Observation:
    """Holds observations, i.e., inputs to the model. PyTorch-compatible version."""

    images: dict[str, torch.Tensor]
    image_masks: dict[str, torch.Tensor]
    state: torch.Tensor
    tokenized_prompt: torch.Tensor | None = None
    tokenized_prompt_mask: torch.Tensor | None = None
    token_ar_mask: torch.Tensor | None = None
    token_loss_mask: torch.Tensor | None = None
    pcd_xyz: torch.Tensor | None = None

    @classmethod
    def from_observation_like(cls, observation: Any) -> Observation:
        """Convert a local mapping or official OpenPI Observation to this type.

        The official ``openpi.models.model.Observation`` exposes ``to_dict()``,
        whose keys match the local ``from_dict`` boundary. Its loader has already
        applied all data transforms, so this only changes the Python container.
        """
        if isinstance(observation, cls):
            return observation
        if isinstance(observation, Mapping):
            return cls.from_dict(dict(observation))

        to_dict = getattr(observation, "to_dict", None)
        if callable(to_dict):
            return cls.from_dict(to_dict())
        raise TypeError(
            "SFT observation must be a local Observation, mapping, or an "
            f"OpenPI Observation with to_dict(); got {type(observation)!r}."
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Observation:
        """Convert a nested dict to an Observation."""
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")

        images = data["image"]
        image_masks = data["image_mask"]

        # Convert uint8 images to [-1, 1] float32
        for key in images:
            if images[key].dtype == torch.uint8:
                images[key] = images[key].to(torch.float32) / 255.0 * 2.0 - 1.0
            elif images[key].dtype == np.uint8:
                images[key] = torch.from_numpy(images[key].astype(np.float32)) / 255.0 * 2.0 - 1.0

        return cls(
            images=images,
            image_masks=image_masks,
            state=data["state"],
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
            pcd_xyz=data.get("pcd_xyz"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the Observation to a nested dict."""
        result = dataclasses.asdict(self)
        result["image"] = result.pop("images")
        result["image_mask"] = result.pop("image_masks")
        if "actions" in result:
            del result["actions"]
        return result


def _tensor_to_dtype(t: torch.Tensor | None, dtype: torch.dtype) -> torch.Tensor | None:
    """Cast a tensor to dtype, but skip integer tensors (e.g. token indices)."""
    if t is None:
        return None
    if t.dtype in (torch.long, torch.int, torch.int32, torch.int64, torch.bool):
        return t
    return t.to(dtype=dtype)


def observation_to_dtype(obs: Observation, dtype: torch.dtype) -> Observation:
    """Cast all float tensors in an Observation to the target dtype.

    Used to ensure inputs match FSDP2 MixedPrecisionPolicy parameter dtype,
    since cast_forward_inputs cannot reach tensors nested inside dataclasses.

    Images are kept in float32 — SigLIPViT runs stem in float32 internally (matching JAX).
    """
    return Observation(
        images=obs.images,  # Leave images in float32 for SigLIPViT stem
        image_masks={k: _tensor_to_dtype(v, dtype) for k, v in obs.image_masks.items()},
        state=obs.state.to(dtype=dtype),
        tokenized_prompt=_tensor_to_dtype(obs.tokenized_prompt, dtype),
        tokenized_prompt_mask=_tensor_to_dtype(obs.tokenized_prompt_mask, dtype),
        token_ar_mask=_tensor_to_dtype(obs.token_ar_mask, dtype),
        token_loss_mask=_tensor_to_dtype(obs.token_loss_mask, dtype),
        pcd_xyz=_tensor_to_dtype(obs.pcd_xyz, dtype),
    )


def preprocess_observation(
    observation: Observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
    rng: torch.Generator | None = None,
) -> Observation:
    """Preprocess observations with optional image augmentations.

    For training, applies random crop, rotate, and color jitter augmentations.
    Resizes images to the target resolution with padding.
    """
    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]

    out_images = {}
    for key in image_keys:
        image = observation.images[key]

        if image.shape[-1] != 3 and image.shape[1] == 3:
            image = image.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)

        if image.shape[1:3] != image_resolution:
            image = resize_with_pad_torch(image, *image_resolution)

        if train:
            image = image / 2.0 + 0.5
            augmented = []
            for batch_index in range(image.shape[0]):
                aug_image = image[batch_index].permute(2, 0, 1)

                if "wrist" not in key:
                    h, w = aug_image.shape[1], aug_image.shape[2]
                    crop_h, crop_w = int(h * 0.95), int(w * 0.95)
                    top = torch.randint(
                        0,
                        h - crop_h + 1,
                        (),
                        device=aug_image.device,
                        generator=rng,
                    ).item()
                    left = torch.randint(
                        0,
                        w - crop_w + 1,
                        (),
                        device=aug_image.device,
                        generator=rng,
                    ).item()
                    aug_image = aug_image[:, top : top + crop_h, left : left + crop_w]
                    aug_image = functional.interpolate(
                        aug_image[None], size=(h, w), mode="bilinear", align_corners=False
                    )[0]

                    angle = torch.rand((), device=aug_image.device, generator=rng) * 10.0 - 5.0
                    radians = angle * torch.pi / 180.0
                    theta = torch.stack(
                        [
                            torch.cos(radians),
                            -torch.sin(radians),
                            torch.zeros((), device=aug_image.device),
                            torch.sin(radians),
                            torch.cos(radians),
                            torch.zeros((), device=aug_image.device),
                        ]
                    ).reshape(1, 2, 3)
                    grid = functional.affine_grid(theta, (1, *aug_image.shape), align_corners=False)
                    aug_image = functional.grid_sample(
                        aug_image[None], grid, mode="bilinear", padding_mode="zeros", align_corners=False
                    )[0]

                brightness = 0.7 + torch.rand((), device=aug_image.device, generator=rng) * 0.6
                contrast = 0.6 + torch.rand((), device=aug_image.device, generator=rng) * 0.8
                saturation = 0.5 + torch.rand((), device=aug_image.device, generator=rng)
                aug_image = aug_image * brightness
                aug_image = (aug_image - aug_image.mean()) * contrast + aug_image.mean()
                gray = aug_image.mean(dim=0, keepdim=True)
                aug_image = gray + (aug_image - gray) * saturation
                augmented.append(aug_image.clamp(0.0, 1.0).permute(1, 2, 0))
            image = torch.stack(augmented, dim=0)
            image = image * 2.0 - 1.0

        out_images[key] = image

    # Build masks
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            out_masks[key] = torch.ones(batch_shape, dtype=torch.bool, device=observation.state.device)
        else:
            mask = observation.image_masks[key]
            if not isinstance(mask, torch.Tensor):
                mask = torch.as_tensor(mask)
            out_masks[key] = mask.to(device=observation.state.device, dtype=torch.bool)

    return Observation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
        pcd_xyz=observation.pcd_xyz,
    )


@dataclasses.dataclass
class BaseModelConfig(abc.ABC):
    """Configuration shared by all models."""

    # Action space dimension.
    action_dim: int
    # Action sequence length.
    action_horizon: int
    # Tokenized prompt maximum length.
    max_token_len: int

    @abc.abstractmethod
    def create(self, **kwargs) -> BaseModel:
        """Create a new model, initializing parameters."""

    def fake_obs(self, batch_size: int = 1) -> Observation:
        """Create fake observations for testing."""
        return Observation(
            images={k: torch.ones(batch_size, *IMAGE_RESOLUTION, 3) for k in IMAGE_KEYS},
            image_masks={k: torch.ones(batch_size, dtype=torch.bool) for k in IMAGE_KEYS},
            state=torch.ones(batch_size, self.action_dim),
            tokenized_prompt=torch.ones(batch_size, self.max_token_len, dtype=torch.long),
            tokenized_prompt_mask=torch.ones(batch_size, self.max_token_len, dtype=torch.bool),
        )

    def fake_act(self, batch_size: int = 1) -> torch.Tensor:
        """Create fake actions for testing."""
        return torch.ones(batch_size, self.action_horizon, self.action_dim)


class BaseModel(nn.Module, abc.ABC):
    """Base class for all model implementations."""

    def __init__(self, action_dim: int, action_horizon: int, max_token_len: int):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.max_token_len = max_token_len

    @abc.abstractmethod
    def compute_loss(
        self,
        observation: Observation,
        actions: torch.Tensor,
        *,
        train: bool = False,
        rng: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Compute the loss for a batch of observations and actions."""

    @abc.abstractmethod
    def sample_actions(
        self,
        observation: Observation,
        *,
        num_steps: int = 10,
        noise: torch.Tensor | None = None,
        rng: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample actions given an observation."""

    def forward(
        self,
        observation: Observation,
        actions: torch.Tensor,
        *,
        train: bool = True,
        rng: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Default forward pass computes the loss."""
        return self.compute_loss(observation, actions, train=train, rng=rng)

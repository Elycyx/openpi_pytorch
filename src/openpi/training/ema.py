from collections.abc import Mapping

import torch
from torch import nn


class ExponentialMovingAverage:
    def __init__(
        self,
        model: nn.Module,
        decay: float,
        *,
        device: torch.device | str = "cpu",
        rank: int = 0,
        world_size: int = 1,
    ):
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"EMA decay must be in [0, 1), got {decay}.")
        self.decay = decay
        self.device = torch.device(device)
        self.rank = rank
        self.world_size = world_size
        self.shadow = {
            name: parameter.detach().to(device=self.device, dtype=torch.float32).clone()
            for index, (name, parameter) in enumerate(model.named_parameters())
            if index % world_size == rank
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        parameters = dict(model.named_parameters())
        update_weight = 1.0 - self.decay
        for name, shadow_parameter in self.shadow.items():
            if name not in parameters:
                raise ValueError(f"EMA model parameter disappeared: {name}")
            shadow_parameter.lerp_(
                parameters[name].detach().to(device=self.device, dtype=torch.float32),
                update_weight,
            )

    def local_state_dict(self) -> dict[str, torch.Tensor]:
        return {name: tensor.detach().to(device="cpu") for name, tensor in self.shadow.items()}

    def model_state_dict(self, model: nn.Module) -> dict[str, torch.Tensor]:
        if self.world_size != 1:
            raise ValueError("Distributed EMA shards must be merged before creating a model state dict.")
        state_dict = {name: tensor.detach().to(device="cpu") for name, tensor in model.state_dict().items()}
        state_dict.update(self.local_state_dict())
        return state_dict

    def load_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        missing = self.shadow.keys() - state_dict.keys()
        if missing:
            raise ValueError(f"EMA checkpoint is missing parameters: {sorted(missing)[:5]}")
        for name, shadow_parameter in self.shadow.items():
            shadow_parameter.copy_(state_dict[name].to(device=shadow_parameter.device, dtype=torch.float32))

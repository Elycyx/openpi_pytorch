import torch
from torch import nn

from openpi.training.ema import ExponentialMovingAverage


def test_ema_updates_in_float32():
    model = nn.Linear(2, 1, bias=False).to(dtype=torch.bfloat16)
    model.weight.data.fill_(1.0)
    ema = ExponentialMovingAverage(model, decay=0.9)
    model.weight.data.fill_(2.0)

    ema.update(model)

    assert ema.shadow["weight"].dtype == torch.float32
    assert ema.shadow["weight"].device.type == "cpu"
    torch.testing.assert_close(ema.shadow["weight"], torch.full((1, 2), 1.1))


def test_ema_state_dict_replaces_train_parameters():
    model = nn.Linear(2, 1, bias=False)
    model.weight.data.fill_(1.0)
    ema = ExponentialMovingAverage(model, decay=0.5)
    model.weight.data.fill_(3.0)
    ema.update(model)

    state_dict = ema.model_state_dict(model)

    torch.testing.assert_close(state_dict["weight"], torch.full((1, 2), 2.0))


def test_ema_shards_parameters_across_ranks():
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
    first = ExponentialMovingAverage(model, decay=0.9, rank=0, world_size=2)
    second = ExponentialMovingAverage(model, decay=0.9, rank=1, world_size=2)

    assert set(first.shadow).isdisjoint(second.shadow)
    assert set(first.shadow) | set(second.shadow) == set(dict(model.named_parameters()))

import dataclasses
import pathlib
from unittest import mock

import jax

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_create_torch_dataset_with_revision():
    data_config = _config.DataConfig(
        repo_id="owner/dataset",
        revision="main",
        action_sequence_keys=("action",),
    )
    model_config = pi0_config.Pi0Config(action_horizon=10)
    dataset_root = pathlib.Path("/tmp/lerobot/owner/dataset")

    with (
        mock.patch.object(_data_loader, "HF_LEROBOT_HOME", pathlib.Path("/tmp/lerobot")),
        mock.patch.object(_data_loader, "snapshot_download") as snapshot_download,
        mock.patch.object(_data_loader.lerobot_dataset, "LeRobotDatasetMetadata") as metadata_cls,
        mock.patch.object(_data_loader.lerobot_dataset, "LeRobotDataset") as dataset_cls,
    ):
        metadata_cls.return_value.fps = 30
        dataset = _data_loader.create_torch_dataset(data_config, model_config.action_horizon, model_config)

    snapshot_download.assert_called_once_with(
        "owner/dataset",
        repo_type="dataset",
        revision="main",
        local_dir=dataset_root,
    )
    metadata_cls.assert_called_once_with("owner/dataset", root=dataset_root, revision="main")
    dataset_cls.assert_called_once_with(
        "owner/dataset",
        root=dataset_root,
        revision="main",
        delta_timestamps={"action": [step / 30 for step in range(model_config.action_horizon)]},
    )
    assert dataset is dataset_cls.return_value


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)

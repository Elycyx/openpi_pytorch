"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import logging
import pathlib

import filelock
import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def compute_norm_stats(config: _config.TrainConfig, max_frames: int | None = None) -> pathlib.Path:
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    asset_id = data_config.asset_id or data_config.repo_id
    if asset_id is None:
        raise ValueError("Data config must have an asset_id or repo_id")
    assets_dir = config.data.assets.assets_dir or config.assets_dirs
    if "://" in str(assets_dir):
        raise ValueError(f"Cannot write computed norm stats to remote assets directory: {assets_dir}")
    output_path = pathlib.Path(assets_dir) / asset_id
    logging.info("Writing norm stats to %s", output_path)
    normalize.save(output_path, norm_stats)
    return output_path


def ensure_norm_stats(config: _config.TrainConfig, max_frames: int | None = None) -> bool:
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.norm_stats is not None or data_config.repo_id in (None, "fake"):
        return False

    asset_id = data_config.asset_id or data_config.repo_id
    assets_dir = config.data.assets.assets_dir or config.assets_dirs
    if "://" in str(assets_dir):
        raise FileNotFoundError(
            f"Norm stats are missing from remote assets directory {assets_dir}; "
            "set --data.assets.assets-dir to a writable local directory."
        )
    output_path = pathlib.Path(assets_dir) / asset_id
    lock_path = output_path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with filelock.FileLock(lock_path):
        data_config = config.data.create(config.assets_dirs, config.model)
        if data_config.norm_stats is not None:
            return False
        logging.info("Norm stats not found at %s; computing them before training", output_path)
        compute_norm_stats(config, max_frames=max_frames)
    return True


def main(config_name: str, max_frames: int | None = None):
    config = _config.get_config(config_name)
    compute_norm_stats(config, max_frames=max_frames)


if __name__ == "__main__":
    tyro.cli(main)

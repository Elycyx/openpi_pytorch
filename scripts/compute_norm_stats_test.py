import dataclasses

import numpy as np

from openpi.shared import normalize
from openpi.training import config as config_lib
from scripts import compute_norm_stats


def test_ensure_norm_stats_uses_existing_file(tmp_path):
    config = config_lib.get_config("pi0_libero")
    data = dataclasses.replace(
        config.data,
        assets=config_lib.AssetsConfig(assets_dir=str(tmp_path)),
    )
    config = dataclasses.replace(config, data=data)
    stats = {
        "state": normalize.NormStats(mean=np.zeros(32), std=np.ones(32)),
        "actions": normalize.NormStats(mean=np.zeros(32), std=np.ones(32)),
    }
    normalize.save(tmp_path / "physical-intelligence/libero", stats)

    computed = compute_norm_stats.ensure_norm_stats(config)

    assert not computed


def test_ensure_norm_stats_skips_fake_data():
    config = config_lib.get_config("debug")

    assert not compute_norm_stats.ensure_norm_stats(config)


def test_ensure_norm_stats_computes_missing_file(tmp_path, monkeypatch):
    config = config_lib.get_config("pi0_libero")
    data = dataclasses.replace(
        config.data,
        assets=config_lib.AssetsConfig(assets_dir=str(tmp_path)),
    )
    config = dataclasses.replace(config, data=data)

    def fake_compute(config, max_frames=None):
        del config, max_frames
        output_path = tmp_path / "physical-intelligence/libero"
        stats = {
            "state": normalize.NormStats(mean=np.zeros(32), std=np.ones(32)),
            "actions": normalize.NormStats(mean=np.zeros(32), std=np.ones(32)),
        }
        normalize.save(output_path, stats)
        return output_path

    monkeypatch.setattr(compute_norm_stats, "compute_norm_stats", fake_compute)

    assert compute_norm_stats.ensure_norm_stats(config)
    assert (tmp_path / "physical-intelligence/libero/norm_stats.json").exists()

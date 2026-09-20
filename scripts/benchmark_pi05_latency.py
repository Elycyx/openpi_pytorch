import argparse
import dataclasses
import json
import os
import pathlib
import statistics
import subprocess
import sys
import time

import numpy as np


def percentile(values: list[float], value: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), value))


def summarize(durations: list[float], batch_size: int) -> dict[str, float]:
    return {
        "mean_ms": statistics.mean(durations) * 1000,
        "std_ms": statistics.pstdev(durations) * 1000,
        "p50_ms": percentile(durations, 50) * 1000,
        "p90_ms": percentile(durations, 90) * 1000,
        "p95_ms": percentile(durations, 95) * 1000,
        "min_ms": min(durations) * 1000,
        "max_ms": max(durations) * 1000,
        "samples_per_second": batch_size / statistics.mean(durations),
    }


def config_with_assets(config_name: str, batch_size: int, assets_dir: pathlib.Path):
    from openpi.training import config as config_lib

    config = config_lib.get_config(config_name)
    data = dataclasses.replace(
        config.data,
        assets=config_lib.AssetsConfig(assets_dir=str(assets_dir)),
    )
    return dataclasses.replace(config, data=data, batch_size=batch_size, num_workers=0, exp_name="latency")


def prepare_libero_batch(args) -> None:
    import torch

    from openpi.training import data_loader

    if args.batch_file.exists() and not args.force_prepare:
        return
    loader = data_loader.create_data_loader(
        config_with_assets("pi05_libero", args.batch_size, args.assets_dir),
        shuffle=False,
        num_batches=1,
        framework="pytorch",
    )
    observation, actions = next(iter(loader))
    observation_dict = observation.to_dict()
    observation_dict["image"] = {
        key: image.permute(0, 2, 3, 1) if image.ndim == 4 and image.shape[1] == 3 else image
        for key, image in observation_dict["image"].items()
    }
    generator = torch.Generator().manual_seed(args.seed)
    noise = torch.randn(actions.shape, generator=generator)
    args.batch_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "observation": observation_dict,
            "noise": noise,
        },
        args.batch_file,
    )


def run_jax_worker(args) -> dict:
    import jax
    import jax.numpy as jnp
    import torch

    from openpi.models import model as model_lib
    from openpi.shared import download
    from openpi.shared import nnx_utils
    from openpi.training import config as config_lib

    payload = torch.load(args.batch_file, map_location="cpu", weights_only=False)
    checkpoint = download.maybe_download(str(args.jax_checkpoint))
    config = config_lib.get_config("pi05_libero")

    load_started = time.perf_counter()
    params = model_lib.restore_params(pathlib.Path(checkpoint) / "params", dtype=jnp.bfloat16)
    model = config.model.load(params)
    load_seconds = time.perf_counter() - load_started

    observation = model_lib.Observation.from_dict(jax.tree.map(jnp.asarray, payload["observation"]))
    noise = jnp.asarray(payload["noise"].numpy())
    sample_actions = nnx_utils.module_jit(model.sample_actions)

    compile_started = time.perf_counter()
    actions = sample_actions(jax.random.key(args.seed), observation, noise=noise, num_steps=args.denoise_steps)
    actions.block_until_ready()
    compile_seconds = time.perf_counter() - compile_started

    warmup_started = time.perf_counter()
    for _ in range(args.warmup):
        actions = sample_actions(jax.random.key(args.seed), observation, noise=noise, num_steps=args.denoise_steps)
        actions.block_until_ready()
    warmup_seconds = time.perf_counter() - warmup_started

    durations = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        actions = sample_actions(jax.random.key(args.seed), observation, noise=noise, num_steps=args.denoise_steps)
        actions.block_until_ready()
        durations.append(time.perf_counter() - started)

    result = {
        "backend": "jax",
        "model_load_seconds": load_seconds,
        "compile_and_first_inference_seconds": compile_seconds,
        "warmup_seconds": warmup_seconds,
        "batch_size": int(noise.shape[0]),
        "denoise_steps": args.denoise_steps,
        "latency": summarize(durations, int(noise.shape[0])),
        "action_checksum": float(np.asarray(actions, dtype=np.float32).sum()),
    }
    memory_stats = jax.devices()[0].memory_stats()
    if memory_stats and "peak_bytes_in_use" in memory_stats:
        result["peak_memory_gib"] = memory_stats["peak_bytes_in_use"] / 1024**3
    return result


def run_pytorch_worker(args, *, compile_model: bool) -> dict:
    import jax
    import safetensors.torch
    import torch

    from openpi.models import model as model_lib
    from openpi.models_pytorch import pi0_pytorch
    from openpi.training import config as config_lib

    payload = torch.load(args.batch_file, map_location="cpu", weights_only=False)
    device = torch.device("cuda:0")
    compile_mode = "max-autotune" if compile_model else None
    model_config = dataclasses.replace(
        config_lib.get_config("pi05_libero_low_mem_finetune").model,
        pytorch_compile_mode=compile_mode,
    )

    load_started = time.perf_counter()
    model = pi0_pytorch.PI0Pytorch(model_config).to(device)
    safetensors.torch.load_model(model, args.pytorch_checkpoint, device=str(device))
    model.eval()
    load_seconds = time.perf_counter() - load_started

    observation = model_lib.Observation.from_dict(
        jax.tree.map(lambda value: torch.as_tensor(value).to(device), payload["observation"])
    )
    noise = payload["noise"].to(device=device, dtype=torch.float32)

    torch.cuda.reset_peak_memory_stats(device)
    compile_started = time.perf_counter()
    actions = model.sample_actions(device, observation, noise=noise, num_steps=args.denoise_steps)
    torch.cuda.synchronize(device)
    compile_seconds = time.perf_counter() - compile_started

    warmup_started = time.perf_counter()
    for _ in range(args.warmup):
        actions = model.sample_actions(device, observation, noise=noise, num_steps=args.denoise_steps)
        torch.cuda.synchronize(device)
    warmup_seconds = time.perf_counter() - warmup_started

    durations = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        actions = model.sample_actions(device, observation, noise=noise, num_steps=args.denoise_steps)
        torch.cuda.synchronize(device)
        durations.append(time.perf_counter() - started)

    return {
        "backend": "pytorch_compile_max_autotune" if compile_model else "pytorch_eager",
        "model_load_seconds": load_seconds,
        "compile_and_first_inference_seconds": compile_seconds,
        "warmup_seconds": warmup_seconds,
        "batch_size": int(noise.shape[0]),
        "denoise_steps": args.denoise_steps,
        "latency": summarize(durations, int(noise.shape[0])),
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "action_checksum": float(actions.float().sum().cpu()),
    }


def worker_main(args) -> None:
    if args.worker == "jax":
        result = run_jax_worker(args)
    elif args.worker == "pytorch-eager":
        result = run_pytorch_worker(args, compile_model=False)
    else:
        result = run_pytorch_worker(args, compile_model=True)
    args.worker_output.write_text(json.dumps(result, indent=2))


def run_worker(args, worker: str) -> dict:
    output = args.output_dir / f"{worker}.json"
    command = [
        sys.executable,
        str(pathlib.Path(__file__).resolve()),
        "--worker",
        worker,
        "--worker-output",
        str(output),
        "--batch-file",
        str(args.batch_file),
        "--jax-checkpoint",
        str(args.jax_checkpoint),
        "--pytorch-checkpoint",
        str(args.pytorch_checkpoint),
        "--repeats",
        str(args.repeats),
        "--warmup",
        str(args.warmup),
        "--denoise-steps",
        str(args.denoise_steps),
        "--seed",
        str(args.seed),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    environment.setdefault("TORCHINDUCTOR_CACHE_DIR", str(args.output_dir / "torchinductor_cache"))
    log_path = args.output_dir / f"{worker}.log"
    print(f"Running {worker}; log: {log_path}", flush=True)
    started = time.perf_counter()
    with log_path.open("w") as log_file:
        subprocess.run(
            command,
            check=True,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            timeout=args.worker_timeout,
        )
    print(f"Finished {worker} in {time.perf_counter() - started:.1f}s", flush=True)
    return json.loads(output.read_text())


def print_results(results: list[dict]) -> None:
    print("\nPi0.5 latency benchmark")
    print(
        f"{'Backend':34} {'Compile/first(s)':>18} {'Mean(ms)':>10} {'P50(ms)':>10} {'P95(ms)':>10} {'Samples/s':>10} {'Peak GiB':>10}"
    )
    for result in results:
        latency = result["latency"]
        print(
            f"{result['backend']:34} "
            f"{result['compile_and_first_inference_seconds']:18.2f} "
            f"{latency['mean_ms']:10.2f} "
            f"{latency['p50_ms']:10.2f} "
            f"{latency['p95_ms']:10.2f} "
            f"{latency['samples_per_second']:10.2f} "
            f"{result.get('peak_memory_gib', float('nan')):10.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark JAX and PyTorch Pi0.5 inference latency.")
    parser.add_argument("--jax-checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    parser.add_argument(
        "--pytorch-checkpoint",
        type=pathlib.Path,
        default=pathlib.Path("checkpoints/pi05_libero_low_mem_finetune/pi05_libero_lora/30000/model.safetensors"),
    )
    parser.add_argument(
        "--assets-dir",
        type=pathlib.Path,
        default=pathlib.Path("/home/fuxin/.cache/openpi/openpi-assets/checkpoints/pi05_libero/assets"),
    )
    parser.add_argument(
        "--batch-file", type=pathlib.Path, default=pathlib.Path("artifacts/pi05_latency/libero_batch.pt")
    )
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("artifacts/pi05_latency"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--denoise-steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--worker-timeout", type=int, default=1800)
    parser.add_argument("--worker", choices=("jax", "pytorch-eager", "pytorch-compile"))
    parser.add_argument("--worker-output", type=pathlib.Path)
    args = parser.parse_args()

    if args.worker:
        if args.worker_output is None:
            raise ValueError("--worker-output is required in worker mode")
        worker_main(args)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepare_libero_batch(args)
    workers = ["jax", "pytorch-eager"]
    if not args.skip_compile:
        workers.append("pytorch-compile")
    results = [run_worker(args, worker) for worker in workers]
    summary = {
        "jax_checkpoint": str(args.jax_checkpoint),
        "pytorch_checkpoint": str(args.pytorch_checkpoint),
        "gpu": args.gpu,
        "results": results,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print_results(results)


if __name__ == "__main__":
    main()

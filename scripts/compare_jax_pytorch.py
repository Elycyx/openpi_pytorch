import argparse
import dataclasses
import json
import pathlib
import time

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch

from openpi.models import model as model_lib
from openpi.models import pi0 as pi0_jax
from openpi.models_pytorch import pi0_pytorch
from openpi.shared import nnx_utils
from openpi.training import config as config_lib
from openpi.training import data_loader


def _config(config_name: str, batch_size: int, assets_dir: pathlib.Path):
    config = config_lib.get_config(config_name)
    data = dataclasses.replace(
        config.data,
        assets=config_lib.AssetsConfig(assets_dir=str(assets_dir)),
    )
    return dataclasses.replace(config, data=data, batch_size=batch_size, num_workers=0, exp_name="parity")


def prepare_batch(config_name: str, output: pathlib.Path, batch_size: int, assets_dir: pathlib.Path) -> None:
    loader = data_loader.create_data_loader(
        _config(config_name, batch_size, assets_dir),
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
    generator = torch.Generator().manual_seed(20260918)
    noise = torch.randn(actions.shape, generator=generator)
    time_tensor = torch.rand(actions.shape[0], generator=generator) * 0.999 + 0.001
    payload = {
        "observation": observation_dict,
        "actions": actions,
        "noise": noise,
        "time": time_tensor,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)


def _load_payload(path: pathlib.Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _jax_observation(payload):
    return model_lib.Observation.from_dict(jax.tree.map(jnp.asarray, payload["observation"]))


def _torch_observation(payload, device: torch.device):
    return model_lib.Observation.from_dict(
        jax.tree.map(lambda value: torch.as_tensor(value).to(device), payload["observation"])
    )


def _jax_fixed_loss(model, observation, actions, noise, time_tensor):
    observation = model_lib.preprocess_observation(None, observation, train=False)
    time_expanded = time_tensor[..., None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    target = noise - actions
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(observation, x_t, time_tensor)
    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    attention_mask = pi0_jax.make_attn_mask(input_mask, ar_mask)
    positions = jnp.cumsum(input_mask, axis=1) - 1
    (_, suffix_out), _ = model.PaliGemma.llm(
        [prefix_tokens, suffix_tokens],
        mask=attention_mask,
        positions=positions,
        adarms_cond=[None, adarms_cond],
    )
    velocity = model.action_out_proj(suffix_out[:, -model.action_horizon :])
    return jnp.mean(jnp.square(velocity - target))


def run_jax(args) -> None:
    payload = _load_payload(args.batch_file)
    config = _config(args.config_name, payload["actions"].shape[0], args.assets_dir)
    restore_dtype = jnp.bfloat16 if args.mode == "inference" else None
    params = model_lib.restore_params(args.jax_checkpoint / "params", dtype=restore_dtype)
    model = config.model.load(params)
    observation = _jax_observation(payload)
    actions = jnp.asarray(payload["actions"].numpy())
    noise = jnp.asarray(payload["noise"].numpy())
    time_tensor = jnp.asarray(payload["time"].numpy())

    if args.mode == "inference":
        sample = nnx_utils.module_jit(model.sample_actions)
        result = sample(jax.random.key(0), observation, noise=noise, num_steps=args.num_denoise_steps)
        result.block_until_ready()
        durations = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            result = sample(jax.random.key(0), observation, noise=noise, num_steps=args.num_denoise_steps)
            result.block_until_ready()
            durations.append(time.perf_counter() - started)
        np.save(args.action_output, np.asarray(result))
        steady_durations = durations[1:] if len(durations) > 1 else durations
        metrics = {
            "framework": "jax",
            "mode": "inference",
            "mean_seconds": float(np.mean(durations)),
            "std_seconds": float(np.std(durations)),
            "actions_per_second": float(actions.shape[0] / np.mean(durations)),
        }
    else:
        trainable_filter = nnx_utils.PathRegex(".*action_out_proj.*")
        optimizer = optax.adamw(args.learning_rate, b1=0.9, b2=0.95, eps=1e-8, weight_decay=0.0)
        params = nnx.state(model, trainable_filter)
        optimizer_state = optimizer.init(params)

        @nnx.jit
        def train_step(model, optimizer_state):
            loss, grads = nnx.value_and_grad(
                _jax_fixed_loss,
                argnums=nnx.DiffState(0, trainable_filter),
            )(model, observation, actions, noise, time_tensor)
            trainable_params = nnx.state(model, trainable_filter)
            updates, optimizer_state = optimizer.update(grads, optimizer_state, trainable_params)
            nnx.update(model, optax.apply_updates(trainable_params, updates))
            return loss, optimizer_state

        loss, optimizer_state = train_step(model, optimizer_state)
        loss.block_until_ready()
        initial_loss = float(loss)
        losses = []
        durations = []
        for _ in range(args.train_steps):
            started = time.perf_counter()
            loss, optimizer_state = train_step(model, optimizer_state)
            loss.block_until_ready()
            durations.append(time.perf_counter() - started)
            losses.append(float(loss))
        metrics = {
            "framework": "jax",
            "mode": "train",
            "initial_loss": initial_loss,
            "losses": losses,
            "step_seconds": durations,
            "mean_step_seconds": float(np.mean(durations)),
            "steady_mean_step_seconds": float(np.mean(steady_durations)),
            "steps_per_second": float(1.0 / np.mean(durations)),
            "steady_steps_per_second": float(1.0 / np.mean(steady_durations)),
        }
    args.output.write_text(json.dumps(metrics, indent=2))


def run_pytorch(args) -> None:
    payload = _load_payload(args.batch_file)
    device = torch.device("cuda")
    config = _config(args.config_name, payload["actions"].shape[0], args.assets_dir)
    compile_mode = args.torch_compile_mode
    model_config = dataclasses.replace(config.model, pytorch_compile_mode=compile_mode)
    model = pi0_pytorch.PI0Pytorch(model_config).to(device)
    import safetensors.torch

    safetensors.torch.load_model(model, args.pytorch_checkpoint)
    observation = _torch_observation(payload, device)
    actions = payload["actions"].to(device=device, dtype=torch.float32)
    noise = payload["noise"].to(device=device, dtype=torch.float32)
    time_tensor = payload["time"].to(device=device, dtype=torch.float32)

    if args.mode == "inference":
        model.eval()
        result = model.sample_actions(device, observation, noise=noise, num_steps=args.num_denoise_steps)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        durations = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            result = model.sample_actions(device, observation, noise=noise, num_steps=args.num_denoise_steps)
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - started)
        np.save(args.action_output, result.float().cpu().numpy())
        steady_durations = durations[1:] if len(durations) > 1 else durations
        metrics = {
            "framework": "pytorch",
            "mode": "inference",
            "mean_seconds": float(np.mean(durations)),
            "std_seconds": float(np.std(durations)),
            "actions_per_second": float(actions.shape[0] / np.mean(durations)),
            "peak_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "compile_mode": args.torch_compile_mode,
        }
    else:
        model.train()
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("action_out_proj."))
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=args.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.0,
        )
        loss = model.compute_loss(observation, actions, train=False, noise=noise, time=time_tensor).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        initial_loss = float(loss.detach())
        torch.cuda.reset_peak_memory_stats()
        losses = []
        durations = []
        for _ in range(args.train_steps):
            started = time.perf_counter()
            loss = model.compute_loss(observation, actions, train=False, noise=noise, time=time_tensor).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - started)
            losses.append(float(loss.detach()))
        metrics = {
            "framework": "pytorch",
            "mode": "train",
            "initial_loss": initial_loss,
            "losses": losses,
            "step_seconds": durations,
            "mean_step_seconds": float(np.mean(durations)),
            "steady_mean_step_seconds": float(np.mean(steady_durations)),
            "steps_per_second": float(1.0 / np.mean(durations)),
            "steady_steps_per_second": float(1.0 / np.mean(steady_durations)),
            "peak_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
        }
    args.output.write_text(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", choices=("prepare", "jax", "pytorch"), required=True)
    parser.add_argument("--config-name", default="pi0_libero")
    parser.add_argument("--mode", choices=("inference", "train"), default="inference")
    parser.add_argument("--batch-file", type=pathlib.Path, default=pathlib.Path("artifacts/parity/libero_batch.pt"))
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("artifacts/parity/result.json"))
    parser.add_argument("--action-output", type=pathlib.Path, default=pathlib.Path("artifacts/parity/actions.npy"))
    parser.add_argument(
        "--jax-checkpoint",
        type=pathlib.Path,
        default=pathlib.Path("/home/fuxin/.cache/openpi/openpi-assets/checkpoints/pi0_libero"),
    )
    parser.add_argument(
        "--assets-dir",
        type=pathlib.Path,
        default=pathlib.Path("/home/fuxin/.cache/openpi/openpi-assets/checkpoints/pi0_libero/assets"),
    )
    parser.add_argument(
        "--pytorch-checkpoint",
        type=pathlib.Path,
        default=pathlib.Path("artifacts/parity/pi0_libero_pytorch/model.safetensors"),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--num-denoise-steps", type=int, default=10)
    parser.add_argument("--train-steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--torch-compile-mode",
        choices=("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.action_output.parent.mkdir(parents=True, exist_ok=True)
    if args.framework == "prepare":
        prepare_batch(args.config_name, args.batch_file, args.batch_size, args.assets_dir)
    elif args.framework == "jax":
        run_jax(args)
    else:
        run_pytorch(args)


if __name__ == "__main__":
    main()

"""
PyTorch training entrypoint for PI0/PI05 with multi-GPU and multi-node (DDP) support.
This script mirrors the behavior of the JAX trainer (`scripts/train.py`) but runs
entirely in PyTorch using the `PI0Pytorch` model and your existing config/data
pipeline from `src/openpi/training/config.py` and `src/openpi/training/data_loader.py`.

Usage
Single GPU:
  python scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>
  Example:
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test --resume  # Resume from latest checkpoint
Multi-GPU (single node):
  torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>
  Example:
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume
Multi-Node Training:
	torchrun \
    --nnodes=<num_nodes> --nproc_per_node=<gpus_per_node> --node_rank=<rank_of_node> \
    --master_addr=<master_ip> --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>

"""

import dataclasses
import gc
import logging
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import time

import filelock

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.parallel
import tqdm
import wandb

import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import openpi.shared.download as download_lib
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data
from openpi.training.ema import ExponentialMovingAverage
import openpi.training.weight_loaders as _weight_loaders

if __package__:
    from scripts import compute_norm_stats as _compute_norm_stats
else:
    import compute_norm_stats as _compute_norm_stats


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    """Initialize wandb logging."""
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    if use_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        init_kwargs = {"backend": backend, "init_method": "env://"}
        if device.type == "cuda":
            init_kwargs["device_id"] = device
        torch.distributed.init_process_group(**init_kwargs)

        # Set up debugging environment variables for DDP issues
        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    return use_ddp, local_rank, device


def cleanup_ddp():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def set_seed(seed: int, local_rank: int):
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


def build_datasets(config: _config.TrainConfig):
    # Use the unified data loader with PyTorch framework
    data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=config.data_shuffle)
    return data_loader, data_loader.data_config()


def get_model_state_dict(model):
    """Get state dict from model, handling DDP wrapper."""
    return (
        model.module.state_dict()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.state_dict()
    )


def get_model_parameters(model):
    """Get parameters from model, handling DDP wrapper."""
    return (
        model.module.parameters()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.parameters()
    )


def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model


def is_lora_parameter_name(name: str) -> bool:
    return name.endswith((".w_a", ".w_b")) or "_lora_" in name


def configure_trainable_parameters(model, model_config) -> None:
    paligemma_lora = "lora" in model_config.paligemma_variant
    action_expert_lora = "lora" in model_config.action_expert_variant
    if not paligemma_lora and not action_expert_lora:
        return

    for name, parameter in model.named_parameters():
        if not name.startswith("llm."):
            continue
        freeze = False
        if name.startswith("llm.embedder."):
            freeze = paligemma_lora
        expert_zero = any(
            selector in name
            for selector in (".q_proj.0.", ".k_proj.0.", ".v_proj.0.", ".o_proj.0.", ".mlps.0.", "_norms.0.")
        )
        expert_one = any(
            selector in name
            for selector in (".q_proj.1.", ".k_proj.1.", ".v_proj.1.", ".o_proj.1.", ".mlps.1.", "_norms.1.")
        )
        freeze = freeze or (paligemma_lora and expert_zero) or (action_expert_lora and expert_one)
        is_lora_parameter = is_lora_parameter_name(name)
        if freeze and not is_lora_parameter:
            parameter.requires_grad = False

    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(dtype=torch.float32)


def configured_weight_source(config: _config.TrainConfig) -> str | None:
    explicit_path = config.pytorch_weight_path
    if explicit_path and not explicit_path.startswith("/path/to/your"):
        return explicit_path
    if isinstance(config.weight_loader, _weight_loaders.CheckpointWeightLoader):
        return config.weight_loader.params_path
    return None


def checkpoint_root(path: pathlib.Path) -> pathlib.Path:
    if path.is_file() and path.name == "model.safetensors":
        return path.parent
    if path.name == "params":
        return path.parent
    return path


def _conversion_cuda_device(local_rank: int) -> str:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible_devices:
        return str(local_rank)
    devices = [device.strip() for device in visible_devices.split(",") if device.strip()]
    return devices[local_rank] if local_rank < len(devices) else devices[0]


def resolve_pytorch_weight_path(
    config: _config.TrainConfig,
    *,
    is_main: bool,
    use_ddp: bool,
    local_rank: int,
) -> pathlib.Path | None:
    source = configured_weight_source(config)
    if source is None:
        return None

    resolved_source = None
    if is_main:
        resolved_source = checkpoint_root(download_lib.maybe_download(source))
    if use_ddp:
        shared_source = [str(resolved_source) if is_main else None]
        dist.broadcast_object_list(shared_source, src=0)
        resolved_source = pathlib.Path(shared_source[0])
    assert resolved_source is not None

    if (resolved_source / "model.safetensors").exists():
        return resolved_source
    if not (resolved_source / "params").exists():
        raise FileNotFoundError(
            f"Configured weight source is neither a PyTorch checkpoint nor an Orbax checkpoint: {resolved_source}"
        )

    output_dir = (
        pathlib.Path(config.checkpoint_base_dir).resolve()
        / ".converted_weights"
        / config.name
        / config.pytorch_training_precision
        / resolved_source.name
    )
    model_path = output_dir / "model.safetensors"
    if is_main and not model_path.exists():
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        lock_path = output_dir.with_suffix(".lock")
        with filelock.FileLock(lock_path):
            if not model_path.exists():
                logging.info("Converting configured JAX checkpoint %s to %s", resolved_source, output_dir)
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = _conversion_cuda_device(local_rank)
                environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
                environment.pop("JAX_PLATFORMS", None)
                subprocess.run(
                    [
                        sys.executable,
                        str(pathlib.Path(__file__).resolve().parents[1] / "examples/convert_jax_model_to_pytorch.py"),
                        "--checkpoint_dir",
                        str(resolved_source),
                        "--config_name",
                        config.name,
                        "--output_path",
                        str(output_dir),
                        "--precision",
                        config.pytorch_training_precision,
                    ],
                    check=True,
                    env=environment,
                )
                logging.info("Finished converting JAX checkpoint to %s", output_dir)
    if use_ddp:
        if is_main:
            logging.info("Waiting for all ranks to observe the converted checkpoint")
        dist.barrier()
    if not model_path.exists():
        raise FileNotFoundError(f"Automatic PyTorch conversion did not create {model_path}")
    if is_main:
        logging.info("Using converted PyTorch checkpoint at %s", output_dir)
    return output_dir


def save_checkpoint(model, optimizer, global_step, config, is_main, data_config, ema=None):
    """Save a checkpoint with model state, optimizer state, and metadata."""
    # Only save if it's time to save or if it's the final step
    should_save_final = config.save_final_checkpoint and global_step >= config.num_train_steps
    should_save = (global_step % config.save_interval == 0 and global_step > 0) or should_save_final
    if not should_save:
        return

    distributed = dist.is_initialized()
    if is_main:
        # Create temporary directory for atomic checkpoint saving
        final_ckpt_dir = config.checkpoint_dir / f"{global_step}"
        tmp_ckpt_dir = config.checkpoint_dir / f"tmp_{global_step}"

        # Remove any existing temp directory and create new one
        if tmp_ckpt_dir.exists():
            shutil.rmtree(tmp_ckpt_dir)
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

    if distributed:
        dist.barrier()

    final_ckpt_dir = config.checkpoint_dir / f"{global_step}"
    tmp_ckpt_dir = config.checkpoint_dir / f"tmp_{global_step}"
    if ema is not None:
        ema_shard_path = tmp_ckpt_dir / f"ema_rank_{ema.rank}.safetensors"
        safetensors.torch.save_file(ema.local_state_dict(), ema_shard_path)

    if distributed:
        dist.barrier()

    if is_main:
        # Save model state using safetensors (handle shared tensors)
        model_to_save = unwrap_model(model)
        if ema is None:
            safetensors.torch.save_model(model_to_save, tmp_ckpt_dir / "model.safetensors")
        else:
            safetensors.torch.save_model(model_to_save, tmp_ckpt_dir / "train_model.safetensors")
            ema_state = {
                name: tensor.detach().to(device="cpu")
                for name, tensor in model_to_save.state_dict().items()
                if name not in dict(model_to_save.named_parameters())
            }
            for rank in range(ema.world_size):
                shard_path = tmp_ckpt_dir / f"ema_rank_{rank}.safetensors"
                ema_state.update(safetensors.torch.load_file(shard_path, device="cpu"))
                shard_path.unlink()
            safetensors.torch.save_file(ema_state, tmp_ckpt_dir / "model.safetensors")

        # Save optimizer state using PyTorch format
        torch.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")

        # Save training metadata (avoid saving full config to prevent JAX/Flax compatibility issues)
        metadata = {
            "global_step": global_step,
            "config": dataclasses.asdict(config),
            "timestamp": time.time(),
        }
        torch.save(metadata, tmp_ckpt_dir / "metadata.pt")

        # save norm stats
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(tmp_ckpt_dir / "assets" / data_config.asset_id, norm_stats)

        # Atomically move temp directory to final location
        if final_ckpt_dir.exists():
            shutil.rmtree(final_ckpt_dir)
        tmp_ckpt_dir.rename(final_ckpt_dir)

        logging.info(f"Saved checkpoint at step {global_step} -> {final_ckpt_dir}")

        # Log checkpoint to wandb
        if config.wandb_enabled:
            wandb.log({"checkpoint_step": global_step}, step=global_step)

    if distributed:
        dist.barrier()


def load_checkpoint(model, optimizer, checkpoint_dir, device, ema=None):
    """Load the latest checkpoint and return the global step."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]

    if not checkpoint_steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    latest_step = max(checkpoint_steps)
    ckpt_dir = checkpoint_dir / f"{latest_step}"

    # Clear memory before loading checkpoints
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "before_loading_checkpoint")

    try:
        # Load model state with error handling
        logging.info("Loading model state...")
        train_safetensors_path = ckpt_dir / "train_model.safetensors"
        safetensors_path = train_safetensors_path if train_safetensors_path.exists() else ckpt_dir / "model.safetensors"

        if safetensors_path.exists():
            model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            safetensors.torch.load_model(model_to_load, safetensors_path, device=str(device))
            logging.info("Loaded model state from safetensors format")
        else:
            raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")

        if ema is not None:
            ema_path = ckpt_dir / "model.safetensors"
            ema.load_state_dict(safetensors.torch.load_file(ema_path, device=str(device)))
            logging.info("Loaded EMA state from safetensors format")

        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_model")

        # Load optimizer state with error handling
        logging.info("Loading optimizer state...")
        optimizer_path = ckpt_dir / "optimizer.pt"

        if optimizer_path.exists():
            optimizer_state_dict = torch.load(optimizer_path, map_location=device, weights_only=False)
            logging.info("Loaded optimizer state from pt format")
        else:
            raise FileNotFoundError(f"No optimizer checkpoint found at {ckpt_dir}")

        optimizer.load_state_dict(optimizer_state_dict)
        del optimizer_state_dict
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_optimizer")

        # Load metadata
        logging.info("Loading metadata...")
        metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device, weights_only=False)
        global_step = metadata.get("global_step", latest_step)
        del metadata
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_metadata")

        logging.info(f"Successfully loaded all checkpoint components from step {latest_step}")
        return global_step

    except RuntimeError as e:
        if "out of memory" in str(e):
            # Clear memory and provide detailed error message
            torch.cuda.empty_cache()
            gc.collect()
            logging.error(f"Out of memory error while loading checkpoint: {e!s}")
            log_memory_usage(device, latest_step, "after_oom_error")
            raise RuntimeError(
                "Out of memory while loading checkpoint. Try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            ) from e
        raise


def get_latest_checkpoint_step(checkpoint_dir):
    """Get the latest checkpoint step number from a checkpoint directory."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    return max(checkpoint_steps) if checkpoint_steps else None


def log_memory_usage(device, step, phase="unknown"):
    """Log detailed memory usage information."""
    if not torch.cuda.is_available():
        return

    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    memory_free = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    memory_free = memory_free / 1e9

    # Get more detailed memory info
    memory_stats = torch.cuda.memory_stats(device)
    max_memory_allocated = memory_stats.get("allocated_bytes.all.peak", 0) / 1e9
    max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9

    # Get DDP info if available
    ddp_info = ""
    if dist.is_initialized():
        ddp_info = f" | DDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}"

    logging.info(
        f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB, free: {memory_free:.2f}GB, peak_allocated: {max_memory_allocated:.2f}GB, peak_reserved: {max_memory_reserved:.2f}GB{ddp_info}"
    )


def train_loop(config: _config.TrainConfig):
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    set_seed(config.seed, local_rank)

    # Initialize checkpoint directory and wandb
    resuming = False
    if config.resume:
        # Find checkpoint directory based on experiment name
        exp_checkpoint_dir = config.checkpoint_dir
        if exp_checkpoint_dir.exists():
            # Use validation to find the latest working checkpoint
            latest_step = get_latest_checkpoint_step(exp_checkpoint_dir)
            if latest_step is not None:
                resuming = True
                logging.info(
                    f"Resuming from experiment checkpoint directory: {exp_checkpoint_dir} at step {latest_step}"
                )
            else:
                raise FileNotFoundError(f"No valid checkpoints found in {exp_checkpoint_dir} for resume")
        else:
            raise FileNotFoundError(f"Experiment checkpoint directory {exp_checkpoint_dir} does not exist for resume")
    elif is_main and config.overwrite and config.checkpoint_dir.exists():
        shutil.rmtree(config.checkpoint_dir)
        logging.info(f"Overwriting checkpoint directory: {config.checkpoint_dir}")

    if use_ddp:
        dist.barrier()

    # Create checkpoint directory with experiment name
    if not resuming:
        # For new runs, create experiment-specific checkpoint directory
        exp_checkpoint_dir = config.checkpoint_dir
        if is_main:
            exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created experiment checkpoint directory: {exp_checkpoint_dir}")
        if use_ddp:
            dist.barrier()
    elif is_main:
        # For resume, checkpoint_dir is already set to the experiment directory
        logging.info(f"Using existing experiment checkpoint directory: {config.checkpoint_dir}")

    # Initialize wandb (only on main process)
    if is_main:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    if is_main:
        _compute_norm_stats.ensure_norm_stats(config)
    if use_ddp:
        dist.barrier()

    # Build data loader using the unified data loader
    # Calculate effective batch size per GPU for DDP
    # For N GPUs, each GPU should get batch_size/N samples, so total across all GPUs is batch_size
    world_size = torch.distributed.get_world_size() if use_ddp else 1
    effective_batch_size = config.batch_size // world_size
    logging.info(
        f"Using batch size per GPU: {effective_batch_size} (total batch size across {world_size} GPUs: {config.batch_size})"
    )

    # Pass the original batch size to data loader - it will handle DDP splitting internally
    loader, data_config = build_datasets(config)

    pretrained_weight_path = None
    if not resuming:
        pretrained_weight_path = resolve_pytorch_weight_path(
            config,
            is_main=is_main,
            use_ddp=use_ddp,
            local_rank=local_rank,
        )

    # Log sample images to wandb on first batch
    if is_main and config.wandb_enabled and not resuming:
        # Create a separate data loader for sample batch to avoid consuming the main loader
        sample_data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)
        sample_batch = next(iter(sample_data_loader))
        # Convert observation and actions to torch tensors
        observation, actions = sample_batch
        sample_batch = observation.to_dict()
        sample_batch["actions"] = actions

        # Create sample images for wandb
        images_to_log = []
        # Get batch size from the first image tensor
        batch_size = next(iter(sample_batch["image"].values())).shape[0]
        for i in range(min(5, batch_size)):
            # Concatenate all camera views horizontally for this batch item
            # Convert from NCHW to NHWC format for wandb
            img_concatenated = torch.cat([img[i].permute(1, 2, 0) for img in sample_batch["image"].values()], axis=1)
            img_concatenated = img_concatenated.cpu().numpy()
            images_to_log.append(wandb.Image(img_concatenated))

        wandb.log({"camera_views": images_to_log}, step=0)

        # Clear sample batch from memory aggressively
        del sample_batch, observation, actions, images_to_log, img_concatenated
        del sample_data_loader  # Also delete the sample data loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Cleared sample batch and data loader from memory")

    if is_main:
        logging.info("Building PyTorch model on %d rank(s)", world_size)

    # Build model
    if not isinstance(config.model, openpi.models.pi0_config.Pi0Config):
        # Convert dataclass to Pi0Config if needed
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
        )
    else:
        model_cfg = config.model
        # Update dtype to match pytorch_training_precision
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)
    configure_trainable_parameters(model, model_cfg)
    if is_main:
        logging.info("Finished building PyTorch model")
        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        frozen_parameters = [parameter for parameter in model.parameters() if not parameter.requires_grad]
        trainable_by_dtype = {}
        for parameter in trainable_parameters:
            trainable_by_dtype[str(parameter.dtype)] = (
                trainable_by_dtype.get(str(parameter.dtype), 0) + parameter.numel()
            )
        logging.info(
            "Parameters: trainable=%d frozen=%d trainable_by_dtype=%s",
            sum(parameter.numel() for parameter in trainable_parameters),
            sum(parameter.numel() for parameter in frozen_parameters),
            trainable_by_dtype,
        )

    if hasattr(model, "gradient_checkpointing_enable"):
        enable_gradient_checkpointing = True
        model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing for memory optimization")
    else:
        enable_gradient_checkpointing = False
        logging.info("Gradient checkpointing is not supported for this model")

    # Log initial memory usage after model creation
    if is_main and torch.cuda.is_available():
        log_memory_usage(device, 0, "after_model_creation")

    # Enable memory optimizations for large-scale training
    if world_size >= 8:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set memory allocation configuration
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        logging.info("Enabled memory optimizations for 8+ GPU training")

    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,  # Disable for memory efficiency
            gradient_as_bucket_view=True,  # Enable for memory efficiency
            static_graph=world_size >= 8,  # Enable for 8+ GPUs
        )

    # Load weights from weight_loader if specified (for fine-tuning)
    if pretrained_weight_path is not None:
        logging.info(f"Loading weights from: {pretrained_weight_path}")

        model_path = pretrained_weight_path / "model.safetensors"
        missing, unexpected = safetensors.torch.load_model(unwrap_model(model), model_path, strict=False)
        invalid_missing = [name for name in missing if not is_lora_parameter_name(name)]
        if invalid_missing or unexpected:
            raise RuntimeError(
                f"Invalid pretrained checkpoint: missing={invalid_missing[:10]}, unexpected={unexpected[:10]}"
            )
        logging.info(f"Loaded PyTorch weights from {pretrained_weight_path}")

    # Optimizer + learning rate schedule from config
    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    # Create optimizer with config parameters
    optim = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    ema = (
        ExponentialMovingAverage(
            unwrap_model(model),
            config.ema_decay,
            device=device,
            rank=rank,
            world_size=world_size,
        )
        if config.ema_decay is not None
        else None
    )

    # Load checkpoint if resuming
    global_step = 0
    if resuming:
        global_step = load_checkpoint(model, optim, config.checkpoint_dir, device, ema=ema)
        logging.info(f"Resumed training from step {global_step}")

    def lr_schedule(step: int):
        if step < warmup_steps:
            # Match JAX behavior: start from peak_lr / (warmup_steps + 1)
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        # cosine decay
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    model.train()
    start_time = time.time()
    infos = []  # Collect stats over log interval
    if is_main:
        logging.info(
            f"Running on: {platform.node()} | world_size={torch.distributed.get_world_size() if use_ddp else 1}"
        )
        logging.info(
            f"Training config: batch_size={config.batch_size}, effective_batch_size={effective_batch_size}, num_train_steps={config.num_train_steps}"
        )
        logging.info(f"Memory optimizations: gradient_checkpointing={enable_gradient_checkpointing}")
        logging.info(
            f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, decay_steps={decay_steps}, end_lr={end_lr:.2e}"
        )
        logging.info(
            f"Optimizer: {type(config.optimizer).__name__}, weight_decay={config.optimizer.weight_decay}, clip_norm={config.optimizer.clip_gradient_norm}"
        )
        logging.info(f"EMA decay: {config.ema_decay}")
        logging.info(f"Training precision: {model_cfg.dtype}")

    # Training loop - iterate until we reach num_train_steps
    pbar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not is_main)
        if is_main
        else None
    )

    while global_step < config.num_train_steps:
        # Set epoch for distributed training
        if use_ddp and hasattr(loader, "set_epoch"):
            loader.set_epoch(global_step // len(loader))

        for observation, actions in loader:
            # Check if we've reached the target number of steps
            if global_step >= config.num_train_steps:
                break

            # The unified data loader returns (observation, actions) tuple
            observation = jax.tree.map(lambda x: x.to(device), observation)  # noqa: PLW2901
            actions = actions.to(torch.float32)  # noqa: PLW2901
            actions = actions.to(device)  # noqa: PLW2901

            # Update LR
            for pg in optim.param_groups:
                pg["lr"] = lr_schedule(global_step)

            # Forward pass. LoRA and other trainable parameters remain fp32 while
            # matrix multiplications use bf16, matching the JAX training layout.
            use_autocast = device.type == "cuda" and model_cfg.dtype == "bfloat16"
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_autocast):
                losses = model(observation, actions)
            # Ensure losses is a tensor and handle different return types
            if isinstance(losses, list | tuple):
                losses = torch.stack(losses)
            elif not isinstance(losses, torch.Tensor):
                losses = torch.tensor(losses, device=device, dtype=torch.float32)

            loss = losses.mean()
            reported_loss = loss.detach()
            if dist.is_initialized():
                dist.all_reduce(reported_loss, op=dist.ReduceOp.SUM)
                reported_loss /= dist.get_world_size()

            # Backward pass
            loss.backward()

            # Log memory usage after backward pass
            if global_step < 5 and is_main and torch.cuda.is_available():
                log_memory_usage(device, global_step, "after_backward")

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.optimizer.clip_gradient_norm)

            # Optimizer step
            optim.step()
            if ema is not None:
                ema.update(unwrap_model(model))
            optim.zero_grad(set_to_none=True)

            # Clear gradients more aggressively
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad = None

            # Collect stats
            if is_main:
                infos.append(
                    {
                        "loss": reported_loss.item(),
                        "learning_rate": optim.param_groups[0]["lr"],
                        "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    }
                )

            if is_main and (global_step % config.log_interval == 0):
                elapsed = time.time() - start_time

                # Average stats over log interval
                avg_loss = sum(info["loss"] for info in infos) / len(infos)
                avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)

                avg_grad_norm = None
                if any("grad_norm" in info for info in infos):
                    vals = [
                        info["grad_norm"] for info in infos if "grad_norm" in info and info["grad_norm"] is not None
                    ]
                    if len(vals) > 0:
                        avg_grad_norm = sum(vals) / len(vals)
                logging.info(
                    f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} grad_norm={avg_grad_norm:.2f} time={elapsed:.1f}s"
                    if avg_grad_norm is not None
                    else f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} time={elapsed:.1f}s"
                )

                # Log to wandb
                if config.wandb_enabled and len(infos) > 0:
                    log_payload = {
                        "loss": avg_loss,
                        "learning_rate": avg_lr,
                        "step": global_step,
                        "time_per_step": elapsed / config.log_interval,
                    }
                    if avg_grad_norm is not None:
                        log_payload["grad_norm"] = avg_grad_norm
                    wandb.log(log_payload, step=global_step)

                start_time = time.time()
                infos = []  # Reset stats collection

            global_step += 1
            # Save checkpoint using the new mechanism
            save_checkpoint(model, optim, global_step, config, is_main, data_config, ema=ema)

            # Update progress bar
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    {
                        "loss": f"{reported_loss.item():.4f}",
                        "lr": f"{optim.param_groups[0]['lr']:.2e}",
                        "step": global_step,
                    }
                )

    # Close progress bar
    if pbar is not None:
        pbar.close()

    # Finish wandb run
    if is_main and config.wandb_enabled:
        wandb.finish()

    cleanup_ddp()


def main():
    init_logging()
    config = _config.cli()
    train_loop(config)


if __name__ == "__main__":
    main()

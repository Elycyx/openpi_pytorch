import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path("artifacts/parity")
PLOT_DIR = ROOT / "plots"


def rolling_mean(values: np.ndarray, window: int = 15) -> np.ndarray:
    if len(values) < window:
        return values
    finite = np.isfinite(values).astype(np.float64)
    filled = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    kernel = np.ones(window)
    numerator = np.convolve(filled, kernel, mode="same")
    denominator = np.convolve(finite, kernel, mode="same")
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)


def plot_training() -> None:
    summary = json.loads((ROOT / "training_200_summary.json").read_text())
    titles = {
        "pi0_full": "Pi0 full fine-tuning",
        "pi05_full": "Pi0.5 full fine-tuning",
        "pi0_lora": "Pi0 LoRA",
        "pi05_lora": "Pi0.5 LoRA",
    }
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for axis, name in zip(axes.flat, titles, strict=True):
        for framework, color in (("jax", "#d62728"), ("pytorch", "#1f77b4")):
            rows = summary[name][framework]["rows"]
            steps = np.array([row["step"] for row in rows])
            losses = np.array([row["loss"] for row in rows])
            axis.plot(steps, losses, color=color, alpha=0.18, linewidth=0.8)
            axis.plot(steps, rolling_mean(losses), color=color, linewidth=2.0, label=framework.upper())
            first_bad = summary[name][framework]["first_nonfinite_loss_step"]
            if first_bad is not None:
                axis.axvline(first_bad, color=color, linestyle="--", alpha=0.8)
                axis.text(
                    first_bad + 2,
                    0.95,
                    f"{framework.upper()} non-finite @ {first_bad}",
                    transform=axis.get_xaxis_transform(),
                    color=color,
                    va="top",
                )
        axis.set_title(titles[name])
        axis.set_xlabel("Training step")
        axis.set_ylabel("Flow-matching loss")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.savefig(PLOT_DIR / "training_curves_200.png", dpi=180)
    plt.close(figure)


def plot_actions(model_name: str, horizon: int, *, file_prefix: str) -> None:
    jax_actions = np.load(ROOT / f"{file_prefix}jax_actions.npy")[0, :horizon, :7]
    pytorch_actions = np.load(ROOT / f"{file_prefix}pytorch_actions.npy")[0, :horizon, :7]
    figure, axes = plt.subplots(4, 2, figsize=(13, 11), constrained_layout=True)
    for action_index, axis in enumerate(axes.flat[:7]):
        axis.plot(jax_actions[:, action_index], label="JAX", color="#d62728", linewidth=1.8)
        axis.plot(pytorch_actions[:, action_index], label="PyTorch", color="#1f77b4", linestyle="--", linewidth=1.5)
        axis.set_title(f"Action dimension {action_index}")
        axis.set_xlabel("Action horizon index")
        axis.grid(alpha=0.25)
    axes.flat[0].legend()
    error_axis = axes.flat[7]
    error = np.abs(jax_actions - pytorch_actions)
    image = error_axis.imshow(error.T, aspect="auto", origin="lower", cmap="magma")
    error_axis.set_title("Absolute error")
    error_axis.set_xlabel("Action horizon index")
    error_axis.set_ylabel("Action dimension")
    figure.colorbar(image, ax=error_axis)
    figure.savefig(PLOT_DIR / f"{model_name}_action_outputs.png", dpi=180)
    plt.close(figure)


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plot_training()
    plot_actions("pi0", horizon=50, file_prefix="")
    plot_actions("pi05", horizon=10, file_prefix="pi05_")


if __name__ == "__main__":
    main()

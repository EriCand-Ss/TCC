"""
Visualize predictions from a trained D-RAMZI Photonic KAN without retraining.

Example on Windows:

python visualize_trained_DRAMZI_KAN.py ^
  --results-root "C:\\Users\\erick.sousa\\Downloads\\TCC\\Paper algorithm\\dramzi_photonic_kan_results" ^
  --dataset FashionMNIST ^
  --checkpoint best

For MNIST, replace --dataset FashionMNIST with --dataset MNIST.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


CLASS_NAMES: Dict[str, Tuple[str, ...]] = {
    "MNIST": tuple(str(index) for index in range(10)),
    "FashionMNIST": (
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ),
}


class DRAMZILUTBank(nn.Module):
    """Evaluate fixed D-RAMZI transfer curves by linear LUT interpolation."""

    def __init__(self, normalized_curves: np.ndarray) -> None:
        super().__init__()
        curves = np.asarray(normalized_curves, dtype=np.float32)
        if curves.ndim != 2:
            raise ValueError("normalized_curves must have shape [K, M].")
        self.register_buffer("table", torch.from_numpy(curves.T.copy()))

    @property
    def n_points(self) -> int:
        return int(self.table.shape[0])

    @property
    def n_bases(self) -> int:
        return int(self.table.shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, 0.0, 1.0)
        position = x * float(self.n_points - 1)
        lower_index = torch.floor(position).to(torch.long)
        lower_index = torch.clamp(lower_index, 0, self.n_points - 2)
        upper_index = lower_index + 1
        fraction = (position - lower_index.to(position.dtype)).unsqueeze(-1)
        lower_value = self.table[lower_index]
        upper_value = self.table[upper_index]
        return lower_value + fraction * (upper_value - lower_value)


class DRAMZIKANLayer(nn.Module):
    """KAN layer whose edge functions use a physical D-RAMZI basis bank."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        basis_bank: DRAMZILUTBank,
        use_linear_bypass: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.basis_bank = basis_bank
        self.use_linear_bypass = bool(use_linear_bypass)

        self.photonic_weight = nn.Parameter(
            torch.empty(out_features, in_features, basis_bank.n_bases)
        )
        if self.use_linear_bypass:
            self.linear_weight = nn.Parameter(
                torch.empty(out_features, in_features)
            )
        else:
            self.register_parameter("linear_weight", None)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            photonic_flat = self.photonic_weight.view(self.out_features, -1)
            gain = 1.0 / math.sqrt(2.0) if self.linear_weight is not None else 1.0
            nn.init.xavier_uniform_(photonic_flat, gain=gain)
            if self.linear_weight is not None:
                nn.init.xavier_uniform_(self.linear_weight, gain=gain)
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        basis_values = self.basis_bank(x)
        basis_flat = basis_values.flatten(start_dim=1)
        weight_flat = self.photonic_weight.flatten(start_dim=1)
        output = F.linear(basis_flat, weight_flat, self.bias)
        if self.linear_weight is not None:
            output = output + F.linear(x, self.linear_weight)
        return output


class DRAMZIPhotonicKAN(nn.Module):
    """Two-layer [784, 64, 10] D-RAMZI Photonic KAN."""

    def __init__(
        self,
        normalized_curves: np.ndarray,
        hidden_features: int = 64,
        n_classes: int = 10,
        dropout: float = 0.0,
        use_linear_bypass: bool = True,
    ) -> None:
        super().__init__()
        bank1 = DRAMZILUTBank(normalized_curves)
        bank2 = DRAMZILUTBank(normalized_curves)
        self.layer1 = DRAMZIKANLayer(
            784,
            hidden_features,
            bank1,
            use_linear_bypass=use_linear_bypass,
        )
        self.hidden_norm = nn.LayerNorm(hidden_features)
        self.hidden_dropout = nn.Dropout(dropout)
        self.layer2 = DRAMZIKANLayer(
            hidden_features,
            n_classes,
            bank2,
            use_linear_bypass=use_linear_bypass,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = images.flatten(start_dim=1)
        hidden = self.layer1(x)
        hidden = torch.sigmoid(self.hidden_norm(hidden))
        hidden = self.hidden_dropout(hidden)
        return self.layer2(hidden)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize predictions of a trained D-RAMZI Photonic KAN."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("dramzi_photonic_kan_results"),
        help="Folder containing configuration.json, physical_basis/, MNIST/, etc.",
    )
    parser.add_argument(
        "--dataset",
        choices=("MNIST", "FashionMNIST"),
        default="FashionMNIST",
    )
    parser.add_argument(
        "--checkpoint",
        choices=("best", "final"),
        default="best",
        help="Use the best validation checkpoint or the final-epoch checkpoint.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Dataset folder. Defaults to <results-root parent>/data.",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--grid-size", type=int, default=25)
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Optional test-set index to inspect in detail.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save figures without opening them interactively.",
    )
    return parser.parse_args()


def select_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


def load_configuration(results_root: Path) -> dict:
    path = results_root / "configuration.json"
    if not path.exists():
        raise FileNotFoundError(f"Configuration not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_test_dataset(name: str, data_root: Path):
    transform = transforms.ToTensor()
    if name == "MNIST":
        return datasets.MNIST(
            root=str(data_root), train=False, download=True, transform=transform
        )
    return datasets.FashionMNIST(
        root=str(data_root), train=False, download=True, transform=transform
    )


def load_model(
    results_root: Path,
    dataset_name: str,
    checkpoint_kind: str,
    device: torch.device,
) -> Tuple[nn.Module, dict]:
    configuration = load_configuration(results_root)
    lut_path = results_root / "physical_basis" / "lut_curves.npy"
    if not lut_path.exists():
        raise FileNotFoundError(f"Physical LUT not found: {lut_path}")

    normalized_curves = np.load(lut_path)
    architecture = configuration.get("architecture", [784, 64, 10])
    model = DRAMZIPhotonicKAN(
        normalized_curves=normalized_curves,
        hidden_features=int(architecture[1]),
        n_classes=int(architecture[2]),
        dropout=float(configuration.get("hidden_dropout", 0.0)),
        use_linear_bypass=bool(configuration.get("use_linear_bypass", True)),
    ).to(device)

    checkpoint_filename = (
        "best_validation_model.pt"
        if checkpoint_kind == "best"
        else "final_model.pt"
    )
    checkpoint_path = results_root / dataset_name / checkpoint_filename
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")
    if "epoch" in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")
    return model, checkpoint


@torch.inference_mode()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image_chunks = []
    label_chunks = []
    probability_chunks = []

    for images, labels in loader:
        logits = model(images.to(device, non_blocking=True))
        probabilities = torch.softmax(logits.float(), dim=1).cpu()
        image_chunks.append(images.cpu())
        label_chunks.append(labels.cpu())
        probability_chunks.append(probabilities)

    return (
        torch.cat(image_chunks, dim=0),
        torch.cat(label_chunks, dim=0),
        torch.cat(probability_chunks, dim=0),
    )


def save_figure(figure, path: Path, show: bool) -> None:
    figure.savefig(path, dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def format_class_name(name: str, max_length: int = 12) -> str:
    if len(name) <= max_length:
        return name
    return name.replace("/", "/\n").replace(" ", "\n", 1)


def plot_prediction_grid(
    images: torch.Tensor,
    labels: torch.Tensor,
    probabilities: torch.Tensor,
    indices: Sequence[int],
    class_names: Sequence[str],
    title: str,
    output_path: Path,
    show: bool,
) -> None:
    count = len(indices)
    columns = int(math.ceil(math.sqrt(count)))
    rows = int(math.ceil(count / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(2.45 * columns, 2.7 * rows),
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes).reshape(-1)
    predictions = probabilities.argmax(dim=1)
    confidences = probabilities.max(dim=1).values

    for axis, index in zip(axes_array, indices):
        index = int(index)
        true_class = int(labels[index])
        predicted_class = int(predictions[index])
        confidence = float(confidences[index])
        correct = true_class == predicted_class

        axis.imshow(images[index, 0].numpy(), cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(
            f"Real: {format_class_name(class_names[true_class])}\n"
            f"Pred: {format_class_name(class_names[predicted_class])}\n"
            f"Conf.: {100.0 * confidence:.1f}%",
            fontsize=8,
            color="forestgreen" if correct else "crimson",
        )
        axis.axis("off")

    for axis in axes_array[count:]:
        axis.axis("off")

    figure.suptitle(title, fontsize=15)
    save_figure(figure, output_path, show)


def plot_prediction_details(
    images: torch.Tensor,
    labels: torch.Tensor,
    probabilities: torch.Tensor,
    selected_indices: Sequence[int],
    class_names: Sequence[str],
    output_path: Path,
    show: bool,
) -> None:
    n_rows = len(selected_indices)
    figure, axes = plt.subplots(
        n_rows,
        2,
        figsize=(12, 4.2 * n_rows),
        constrained_layout=True,
        squeeze=False,
    )

    for row, index in enumerate(selected_indices):
        index = int(index)
        probability_vector = probabilities[index].numpy()
        true_class = int(labels[index])
        predicted_class = int(np.argmax(probability_vector))
        confidence = float(probability_vector[predicted_class])
        correct = true_class == predicted_class

        image_axis = axes[row, 0]
        image_axis.imshow(images[index, 0].numpy(), cmap="gray", vmin=0.0, vmax=1.0)
        image_axis.set_title(
            f"Amostra {index}\n"
            f"Real: {class_names[true_class]} | "
            f"Predição: {class_names[predicted_class]} | "
            f"Confiança: {100.0 * confidence:.2f}%",
            color="forestgreen" if correct else "crimson",
        )
        image_axis.axis("off")

        probability_axis = axes[row, 1]
        positions = np.arange(len(class_names))
        bar_colors = ["0.65"] * len(class_names)
        bar_colors[predicted_class] = "forestgreen" if correct else "crimson"
        if true_class != predicted_class:
            bar_colors[true_class] = "royalblue"
        probability_axis.bar(positions, 100.0 * probability_vector, color=bar_colors)
        probability_axis.set_xticks(positions)
        probability_axis.set_xticklabels(class_names, rotation=35, ha="right")
        probability_axis.set_ylabel("Softmax score [%]")
        probability_axis.set_ylim(0.0, 100.0)
        probability_axis.grid(True, axis="y", alpha=0.3)
        probability_axis.set_title("Dez saídas da rede")

    save_figure(figure, output_path, show)


def plot_per_class_accuracy(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    class_names: Sequence[str],
    output_path: Path,
    show: bool,
) -> np.ndarray:
    accuracies = np.zeros(len(class_names), dtype=float)
    counts = np.zeros(len(class_names), dtype=int)

    for class_index in range(len(class_names)):
        mask = labels == class_index
        counts[class_index] = int(mask.sum().item())
        if counts[class_index] > 0:
            accuracies[class_index] = float(
                (predictions[mask] == labels[mask]).float().mean().item()
            )

    figure, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    positions = np.arange(len(class_names))
    bars = axis.bar(positions, 100.0 * accuracies)
    axis.set_xticks(positions)
    axis.set_xticklabels(class_names, rotation=35, ha="right")
    axis.set_ylabel("Accuracy [%]")
    axis.set_ylim(0.0, 100.0)
    axis.set_title("Test accuracy by class")
    axis.grid(True, axis="y", alpha=0.3)

    for bar, accuracy, count in zip(bars, accuracies, counts):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            100.0 * accuracy + 1.0,
            f"{100.0 * accuracy:.1f}%\n(n={count})",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    save_figure(figure, output_path, show)
    return accuracies


def plot_confidence_distribution(
    labels: torch.Tensor,
    probabilities: torch.Tensor,
    output_path: Path,
    show: bool,
) -> None:
    predictions = probabilities.argmax(dim=1)
    confidences = probabilities.max(dim=1).values.numpy()
    correct_mask = (predictions == labels).numpy()

    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    bins = np.linspace(0.0, 1.0, 21)
    axis.hist(
        confidences[correct_mask], bins=bins, alpha=0.65, label="Correct"
    )
    axis.hist(
        confidences[~correct_mask], bins=bins, alpha=0.65, label="Incorrect"
    )
    axis.set_xlabel("Maximum softmax score")
    axis.set_ylabel("Number of test samples")
    axis.set_title("Confidence distribution")
    axis.grid(True, alpha=0.3)
    axis.legend()
    save_figure(figure, output_path, show)


def save_predictions_csv(
    labels: torch.Tensor,
    probabilities: torch.Tensor,
    class_names: Sequence[str],
    output_path: Path,
) -> None:
    predictions = probabilities.argmax(dim=1)
    confidences = probabilities.max(dim=1).values

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "test_index",
                "true_class_index",
                "true_class_name",
                "predicted_class_index",
                "predicted_class_name",
                "softmax_score",
                "correct",
            ]
        )
        for index in range(len(labels)):
            true_class = int(labels[index])
            predicted_class = int(predictions[index])
            writer.writerow(
                [
                    index,
                    true_class,
                    class_names[true_class],
                    predicted_class,
                    class_names[predicted_class],
                    float(confidences[index]),
                    true_class == predicted_class,
                ]
            )


def main() -> None:
    args = parse_arguments()
    results_root = args.results_root.resolve()
    data_root = (
        args.data_root.resolve()
        if args.data_root is not None
        else (results_root.parent / "data").resolve()
    )
    output_dir = results_root / args.dataset / "prediction_visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    show = not args.no_show

    device = select_device()
    model, checkpoint = load_model(
        results_root,
        args.dataset,
        args.checkpoint,
        device,
    )

    test_dataset = load_test_dataset(args.dataset, data_root)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    images, labels, probabilities = collect_predictions(model, test_loader, device)
    predictions = probabilities.argmax(dim=1)
    confidences = probabilities.max(dim=1).values
    correct_mask = predictions == labels
    accuracy = float(correct_mask.float().mean().item())
    n_correct = int(correct_mask.sum().item())
    n_total = len(labels)
    n_errors = n_total - n_correct

    print("\n" + "=" * 80)
    print(f"PREDICTION VISUALIZATION: {args.dataset}")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test examples: {n_total}")
    print(f"Correct: {n_correct}")
    print(f"Incorrect: {n_errors}")
    print(f"Recomputed test accuracy: {100.0 * accuracy:.2f}%")
    print(
        "Note: the maximum softmax score is a model confidence score; "
        "it is not automatically a calibrated probability."
    )

    class_names = CLASS_NAMES[args.dataset]
    rng = np.random.default_rng(args.seed)
    all_indices = np.arange(n_total)
    random_count = min(args.grid_size, n_total)
    random_indices = rng.choice(all_indices, size=random_count, replace=False)

    wrong_indices = torch.where(~correct_mask)[0].numpy()
    if len(wrong_indices) > 0:
        wrong_confidences = confidences[wrong_indices].numpy()
        order = np.argsort(wrong_confidences)[::-1]
        selected_wrong = wrong_indices[order[: min(args.grid_size, len(order))]]
    else:
        selected_wrong = np.array([], dtype=int)

    correct_indices = torch.where(correct_mask)[0].numpy()
    selected_correct = int(correct_indices[np.argmax(confidences[correct_indices].numpy())])

    plot_prediction_grid(
        images,
        labels,
        probabilities,
        random_indices,
        class_names,
        f"{args.dataset}: random test predictions",
        output_dir / "random_test_predictions.png",
        show,
    )

    if len(selected_wrong) > 0:
        plot_prediction_grid(
            images,
            labels,
            probabilities,
            selected_wrong,
            class_names,
            f"{args.dataset}: most confident errors",
            output_dir / "most_confident_errors.png",
            show,
        )
        detail_indices = [selected_correct, int(selected_wrong[0])]
    else:
        detail_indices = [selected_correct]

    if args.index is not None:
        if not 0 <= args.index < n_total:
            raise ValueError(
                f"--index must be between 0 and {n_total - 1}, got {args.index}."
            )
        detail_indices = [int(args.index)] + [
            index for index in detail_indices if index != int(args.index)
        ]

    plot_prediction_details(
        images,
        labels,
        probabilities,
        detail_indices,
        class_names,
        output_dir / "prediction_details.png",
        show,
    )

    per_class_accuracy = plot_per_class_accuracy(
        labels,
        predictions,
        class_names,
        output_dir / "per_class_accuracy.png",
        show,
    )

    plot_confidence_distribution(
        labels,
        probabilities,
        output_dir / "confidence_distribution.png",
        show,
    )

    save_predictions_csv(
        labels,
        probabilities,
        class_names,
        output_dir / "test_predictions.csv",
    )

    summary = {
        "dataset": args.dataset,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "test_examples": n_total,
        "correct": n_correct,
        "incorrect": n_errors,
        "accuracy": accuracy,
        "class_names": list(class_names),
        "per_class_accuracy": per_class_accuracy.tolist(),
    }
    with (output_dir / "visualization_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    print(f"\nFigures and CSV saved in: {output_dir}")


if __name__ == "__main__":
    main()

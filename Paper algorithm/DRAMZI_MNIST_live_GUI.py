"""
Interactive digit recognizer for a trained D-RAMZI Photonic KAN.

Features
--------
- Split window: drawing board on the left, prediction on the right.
- Automatic inference after a configurable idle interval (default: 2 seconds).
- Mouse, touchpad, stylus, or touchscreen drawing through Tk mouse events.
- Load a photo/image from disk.
- Paste an image from the clipboard (Ctrl+V, when supported by Pillow/OS).
- Optional webcam capture when opencv-python is installed.
- MNIST-like preprocessing: polarity correction, crop, resize to 20x20,
  placement on 28x28 canvas, and center-of-mass alignment.
- Shows the exact 28x28 tensor seen by the network and all 10 softmax scores.

Expected result folder
----------------------
<results-root>/
    configuration.json
    physical_basis/lut_curves.npy
    MNIST/best_validation_model.pt
    MNIST/final_model.pt

Example (Windows)
-----------------
python DRAMZI_MNIST_live_GUI.py ^
  --results-root "C:\\Users\\erick.sousa\\Downloads\\TCC\\Paper algorithm\\dramzi_photonic_kan_results" ^
  --checkpoint best

Dependencies
------------
Required:
    pip install numpy pillow torch
Optional webcam:
    pip install opencv-python
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError as exc:
    raise RuntimeError(
        "Tkinter is required. Install a Python distribution that includes Tk/Tkinter."
    ) from exc

try:
    from PIL import Image, ImageDraw, ImageGrab, ImageOps, ImageTk
except ImportError as exc:
    raise RuntimeError("Pillow is required. Install it with: pip install pillow") from exc

try:
    import cv2  # type: ignore

    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False


# ============================================================================
# 1. D-RAMZI PHOTONIC KAN MODEL
# ============================================================================


class DRAMZILUTBank(nn.Module):
    """Evaluate fixed physical D-RAMZI curves by linear LUT interpolation."""

    def __init__(self, normalized_curves: np.ndarray) -> None:
        super().__init__()
        curves = np.asarray(normalized_curves, dtype=np.float32)
        if curves.ndim != 2:
            raise ValueError("normalized_curves must have shape [K, M].")
        if curves.shape[1] < 2:
            raise ValueError("The D-RAMZI LUT must contain at least two x points.")

        # Shape [M, K]. table[index] then returns [..., K].
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
            self.linear_weight = nn.Parameter(torch.empty(out_features, in_features))
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
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError(
                f"Expected [batch, {self.in_features}], got {tuple(x.shape)}."
            )

        basis_values = self.basis_bank(x)  # [B, I, K]
        basis_flat = basis_values.flatten(start_dim=1)
        weight_flat = self.photonic_weight.flatten(start_dim=1)
        output = F.linear(basis_flat, weight_flat, self.bias)

        if self.linear_weight is not None:
            output = output + F.linear(x, self.linear_weight)

        return output


class DRAMZIPhotonicKAN(nn.Module):
    """Two-layer [784, hidden, 10] D-RAMZI Photonic KAN."""

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


# ============================================================================
# 2. MODEL LOADING
# ============================================================================


@dataclass(frozen=True)
class ModelMetadata:
    results_root: Path
    checkpoint_path: Path
    checkpoint_kind: str
    checkpoint_epoch: Optional[int]
    validation_accuracy: Optional[float]
    architecture: Tuple[int, int, int]
    n_bases: int
    lut_points: int
    device_name: str


def select_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_torch_load(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_json_if_present(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_trained_model(
    results_root: Path,
    checkpoint_kind: str,
    force_cpu: bool = False,
) -> Tuple[DRAMZIPhotonicKAN, torch.device, ModelMetadata]:
    results_root = results_root.expanduser().resolve()
    lut_path = results_root / "physical_basis" / "lut_curves.npy"

    checkpoint_filename = (
        "best_validation_model.pt"
        if checkpoint_kind == "best"
        else "final_model.pt"
    )
    checkpoint_path = results_root / "MNIST" / checkpoint_filename

    missing = [path for path in (lut_path, checkpoint_path) if not path.exists()]
    if missing:
        expected = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Required trained-model files were not found:\n"
            f"{expected}\n\n"
            "Select the folder named dramzi_photonic_kan_results."
        )

    device = select_device(force_cpu=force_cpu)
    configuration = load_json_if_present(results_root / "configuration.json")
    checkpoint = safe_torch_load(checkpoint_path, device)
    curves = np.load(lut_path)

    architecture_raw = checkpoint.get(
        "architecture", configuration.get("architecture", [784, 64, 10])
    )
    architecture = tuple(int(value) for value in architecture_raw)
    if len(architecture) != 3:
        raise ValueError(f"Invalid architecture in checkpoint: {architecture_raw}")
    if architecture[0] != 784 or architecture[2] != 10:
        raise ValueError(
            "This GUI expects an MNIST [784, hidden, 10] model, but found "
            f"{architecture}."
        )

    model = DRAMZIPhotonicKAN(
        normalized_curves=curves,
        hidden_features=architecture[1],
        n_classes=architecture[2],
        dropout=float(configuration.get("hidden_dropout", 0.0)),
        use_linear_bypass=bool(configuration.get("use_linear_bypass", True)),
    ).to(device)

    if "model_state_dict" not in checkpoint:
        raise KeyError(f"No model_state_dict in checkpoint: {checkpoint_path}")

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    # Warm-up to avoid paying first-call overhead after the user draws a digit.
    with torch.inference_mode():
        dummy = torch.zeros(1, 1, 28, 28, device=device)
        _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()

    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(0)
    else:
        device_name = "CPU"

    metadata = ModelMetadata(
        results_root=results_root,
        checkpoint_path=checkpoint_path,
        checkpoint_kind=checkpoint_kind,
        checkpoint_epoch=(
            int(checkpoint["epoch"]) if "epoch" in checkpoint else None
        ),
        validation_accuracy=(
            float(checkpoint["validation_accuracy"])
            if "validation_accuracy" in checkpoint
            else None
        ),
        architecture=(architecture[0], architecture[1], architecture[2]),
        n_bases=int(curves.shape[0]),
        lut_points=int(curves.shape[1]),
        device_name=device_name,
    )

    return model, device, metadata


# ============================================================================
# 3. MNIST-LIKE IMAGE PREPROCESSING
# ============================================================================


class EmptyDigitError(ValueError):
    pass


def lanczos_resampling():
    return getattr(Image, "Resampling", Image).LANCZOS


def nearest_resampling():
    return getattr(Image, "Resampling", Image).NEAREST


def bilinear_resampling():
    return getattr(Image, "Resampling", Image).BILINEAR


def normalize_foreground(image: Image.Image) -> np.ndarray:
    """Return a float32 image with bright foreground and dark background."""
    image = ImageOps.exif_transpose(image)
    gray = ImageOps.grayscale(image)
    array = np.asarray(gray, dtype=np.float32)

    if array.size == 0:
        raise EmptyDigitError("Imagem vazia.")

    # Estimate background from the image border.
    border_width = max(1, min(array.shape) // 30)
    border = np.concatenate(
        [
            array[:border_width, :].ravel(),
            array[-border_width:, :].ravel(),
            array[:, :border_width].ravel(),
            array[:, -border_width:].ravel(),
        ]
    )
    border_median = float(np.median(border))

    # MNIST convention: white digit on black background.
    if border_median > 127.0:
        array = 255.0 - array
        border_median = 255.0 - border_median

    # Remove residual illumination/background and stretch useful contrast.
    array = np.clip(array - border_median, 0.0, None)
    positive = array[array > 0.0]
    if positive.size == 0:
        return np.zeros_like(array, dtype=np.float32)

    high = float(np.percentile(positive, 99.5))
    if high <= 1e-6:
        return np.zeros_like(array, dtype=np.float32)

    array = np.clip(array * (255.0 / high), 0.0, 255.0)

    # Suppress weak shadows/compression noise, preserving antialiased edges.
    noise_floor = max(4.0, 0.025 * float(array.max()))
    array[array < noise_floor] = 0.0
    return array.astype(np.float32)


def shift_without_wrap(array: np.ndarray, shift_y: int, shift_x: int) -> np.ndarray:
    """Translate a 2-D image without np.roll wraparound."""
    output = np.zeros_like(array)
    height, width = array.shape

    src_y0 = max(0, -shift_y)
    src_y1 = min(height, height - shift_y)
    src_x0 = max(0, -shift_x)
    src_x1 = min(width, width - shift_x)

    dst_y0 = max(0, shift_y)
    dst_y1 = dst_y0 + max(0, src_y1 - src_y0)
    dst_x0 = max(0, shift_x)
    dst_x1 = dst_x0 + max(0, src_x1 - src_x0)

    if src_y1 > src_y0 and src_x1 > src_x0:
        output[dst_y0:dst_y1, dst_x0:dst_x1] = array[
            src_y0:src_y1, src_x0:src_x1
        ]

    return output


def preprocess_to_mnist(
    source: Image.Image,
    inner_size: int = 20,
) -> Tuple[torch.Tensor, Image.Image]:
    """
    Convert a canvas/photo to the [1, 1, 28, 28] tensor used by the model.

    The digit is cropped, fitted inside a 20x20 box, placed on a 28x28 canvas,
    and translated so its intensity center of mass is near the image center.
    """
    foreground = normalize_foreground(source)
    peak = float(foreground.max())
    if peak <= 0.0:
        raise EmptyDigitError("Nenhum digito foi encontrado.")

    mask_threshold = max(8.0, 0.08 * peak)
    mask = foreground >= mask_threshold
    rows, columns = np.nonzero(mask)
    if len(rows) == 0:
        raise EmptyDigitError("Nenhum digito foi encontrado.")

    y0, y1 = int(rows.min()), int(rows.max()) + 1
    x0, x1 = int(columns.min()), int(columns.max()) + 1

    # Small source-space margin to retain antialiased edges.
    margin = max(1, int(round(0.02 * max(y1 - y0, x1 - x0))))
    y0 = max(0, y0 - margin)
    y1 = min(foreground.shape[0], y1 + margin)
    x0 = max(0, x0 - margin)
    x1 = min(foreground.shape[1], x1 + margin)

    crop = foreground[y0:y1, x0:x1]
    crop_height, crop_width = crop.shape
    if crop_height == 0 or crop_width == 0:
        raise EmptyDigitError("A regiao do digito ficou vazia.")

    scale = min(inner_size / crop_width, inner_size / crop_height)
    new_width = max(1, int(round(crop_width * scale)))
    new_height = max(1, int(round(crop_height * scale)))

    crop_image = Image.fromarray(np.clip(crop, 0, 255).astype(np.uint8), mode="L")
    resized = crop_image.resize((new_width, new_height), lanczos_resampling())

    mnist = np.zeros((28, 28), dtype=np.float32)
    paste_x = (28 - new_width) // 2
    paste_y = (28 - new_height) // 2
    resized_array = np.asarray(resized, dtype=np.float32)
    mnist[paste_y : paste_y + new_height, paste_x : paste_x + new_width] = (
        resized_array
    )

    # Center the intensity center of mass, close to standard MNIST processing.
    total_mass = float(mnist.sum())
    if total_mass > 1e-8:
        yy, xx = np.indices(mnist.shape)
        center_y = float((yy * mnist).sum() / total_mass)
        center_x = float((xx * mnist).sum() / total_mass)
        target_center = 13.5
        shift_y = int(round(target_center - center_y))
        shift_x = int(round(target_center - center_x))
        mnist = shift_without_wrap(mnist, shift_y, shift_x)

    mnist = np.clip(mnist / 255.0, 0.0, 1.0).astype(np.float32)
    tensor = torch.from_numpy(mnist).unsqueeze(0).unsqueeze(0)
    preview = Image.fromarray(np.round(mnist * 255.0).astype(np.uint8), mode="L")
    return tensor, preview


def fit_foreground_to_canvas(source: Image.Image, canvas_size: int) -> Image.Image:
    """Create a black/white display image for the left-side canvas."""
    foreground = normalize_foreground(source)
    peak = float(foreground.max())
    if peak <= 0.0:
        return Image.new("L", (canvas_size, canvas_size), 0)

    mask = foreground >= max(8.0, 0.06 * peak)
    rows, columns = np.nonzero(mask)
    if len(rows) == 0:
        return Image.new("L", (canvas_size, canvas_size), 0)

    y0, y1 = int(rows.min()), int(rows.max()) + 1
    x0, x1 = int(columns.min()), int(columns.max()) + 1
    crop = foreground[y0:y1, x0:x1]
    crop_image = Image.fromarray(np.clip(crop, 0, 255).astype(np.uint8), mode="L")

    inner = int(round(canvas_size * 0.78))
    scale = min(inner / crop_image.width, inner / crop_image.height)
    width = max(1, int(round(crop_image.width * scale)))
    height = max(1, int(round(crop_image.height * scale)))
    resized = crop_image.resize((width, height), lanczos_resampling())

    output = Image.new("L", (canvas_size, canvas_size), 0)
    output.paste(resized, ((canvas_size - width) // 2, (canvas_size - height) // 2))
    return output


# ============================================================================
# 4. INFERENCE WRAPPER
# ============================================================================


@dataclass(frozen=True)
class Prediction:
    digit: int
    confidence: float
    probabilities: np.ndarray
    logits: np.ndarray
    inference_ms: float
    input_preview: Image.Image


class DigitRecognizer:
    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model
        self.device = device

    def predict(self, source: Image.Image) -> Prediction:
        tensor, preview = preprocess_to_mnist(source)
        tensor = tensor.to(self.device, non_blocking=True)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.inference_mode():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = 1000.0 * (time.perf_counter() - start)

        probabilities_np = probabilities[0].detach().cpu().numpy()
        logits_np = logits[0].detach().cpu().numpy()
        digit = int(np.argmax(probabilities_np))

        return Prediction(
            digit=digit,
            confidence=float(probabilities_np[digit]),
            probabilities=probabilities_np,
            logits=logits_np,
            inference_ms=elapsed_ms,
            input_preview=preview,
        )


# ============================================================================
# 5. OPTIONAL WEBCAM WINDOW
# ============================================================================


class CameraDialog:
    def __init__(
        self,
        parent: tk.Misc,
        camera_index: int,
        on_capture: Callable[[Image.Image], None],
    ) -> None:
        if not CV2_AVAILABLE:
            raise RuntimeError("opencv-python is not installed.")

        self.parent = parent
        self.camera_index = camera_index
        self.on_capture = on_capture
        self.window = tk.Toplevel(parent)
        self.window.title("Capturar um digito")
        self.window.geometry("760x650")
        self.window.minsize(600, 520)
        self.window.protocol("WM_DELETE_WINDOW", self.close)

        self.video_label = ttk.Label(self.window, anchor="center")
        self.video_label.pack(fill="both", expand=True, padx=12, pady=12)

        instruction = ttk.Label(
            self.window,
            text=(
                "Centralize um unico digito dentro do quadrado. "
                "Use papel claro e tinta escura."
            ),
            anchor="center",
        )
        instruction.pack(fill="x", padx=12, pady=(0, 8))

        controls = ttk.Frame(self.window)
        controls.pack(fill="x", padx=12, pady=(0, 12))
        ttk.Button(controls, text="Capturar", command=self.capture).pack(
            side="left", expand=True, fill="x", padx=(0, 6)
        )
        ttk.Button(controls, text="Cancelar", command=self.close).pack(
            side="left", expand=True, fill="x", padx=(6, 0)
        )

        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        self.capture_device = cv2.VideoCapture(camera_index, backend)
        if not self.capture_device.isOpened():
            self.capture_device.release()
            self.capture_device = cv2.VideoCapture(camera_index)

        if not self.capture_device.isOpened():
            self.window.destroy()
            raise RuntimeError(f"Could not open camera index {camera_index}.")

        self.latest_rgb: Optional[np.ndarray] = None
        self.photo_reference = None
        self.after_id: Optional[str] = None
        self.update_frame()

    @staticmethod
    def center_square(array: np.ndarray, fraction: float = 0.72) -> Tuple[int, int, int, int]:
        height, width = array.shape[:2]
        side = int(round(min(height, width) * fraction))
        x0 = (width - side) // 2
        y0 = (height - side) // 2
        return x0, y0, x0 + side, y0 + side

    def update_frame(self) -> None:
        ok, frame = self.capture_device.read()
        if ok:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.latest_rgb = rgb

            display = rgb.copy()
            x0, y0, x1, y1 = self.center_square(display)
            cv2.rectangle(display, (x0, y0), (x1, y1), (0, 255, 0), 3)

            image = Image.fromarray(display)
            image.thumbnail((720, 520), bilinear_resampling())
            self.photo_reference = ImageTk.PhotoImage(image, master=self.window)
            self.video_label.configure(image=self.photo_reference)

        if self.window.winfo_exists():
            self.after_id = self.window.after(30, self.update_frame)

    def capture(self) -> None:
        if self.latest_rgb is None:
            return
        x0, y0, x1, y1 = self.center_square(self.latest_rgb)
        crop = self.latest_rgb[y0:y1, x0:x1]
        image = Image.fromarray(crop)
        self.on_capture(image)
        self.close()

    def close(self) -> None:
        if self.after_id is not None:
            try:
                self.window.after_cancel(self.after_id)
            except tk.TclError:
                pass
            self.after_id = None
        if hasattr(self, "capture_device"):
            self.capture_device.release()
        if self.window.winfo_exists():
            self.window.destroy()


# ============================================================================
# 6. TKINTER APPLICATION
# ============================================================================


class DRAMZIDigitApp:
    BACKGROUND = "#17191d"
    PANEL = "#22252b"
    PANEL_ALT = "#2b2f36"
    TEXT = "#f3f4f6"
    MUTED = "#aeb4bf"
    ACCENT = "#4cc2ff"
    SUCCESS = "#55d187"
    WARNING = "#ffbf69"
    DANGER = "#ff6b6b"
    CANVAS_BG = "#000000"
    CANVAS_FG = "#ffffff"

    def __init__(
        self,
        root: tk.Tk,
        recognizer: DigitRecognizer,
        metadata: ModelMetadata,
        idle_ms: int = 2000,
        canvas_size: int = 520,
        camera_index: int = 0,
    ) -> None:
        self.root = root
        self.recognizer = recognizer
        self.metadata = metadata
        self.idle_ms = max(100, int(idle_ms))
        self.canvas_size = max(280, int(canvas_size))
        self.camera_index = int(camera_index)

        self.root.title("D-RAMZI Photonic KAN - Reconhecedor MNIST")
        self.root.geometry("1260x790")
        self.root.minsize(1040, 700)
        self.root.configure(bg=self.BACKGROUND)

        self.canvas_image = Image.new(
            "L", (self.canvas_size, self.canvas_size), color=0
        )
        self.canvas_draw = ImageDraw.Draw(self.canvas_image)
        self.external_source: Optional[Image.Image] = None
        self.canvas_photo_reference = None
        self.preview_photo_reference = None
        self.last_prediction: Optional[Prediction] = None

        self.last_x: Optional[int] = None
        self.last_y: Optional[int] = None
        self.drawing = False
        self.history: list[Image.Image] = []
        self.max_history = 30

        self.pending_prediction_id: Optional[str] = None
        self.countdown_id: Optional[str] = None
        self.prediction_deadline: Optional[float] = None

        self.brush_size = tk.IntVar(value=max(18, self.canvas_size // 14))
        self.tool_mode = tk.StringVar(value="pen")
        self.status_text = tk.StringVar(value="Desenhe um unico digito de 0 a 9.")
        self.countdown_text = tk.StringVar(value="")

        self.configure_styles()
        self.build_interface()
        self.bind_shortcuts()
        self.refresh_canvas()
        self.draw_empty_probability_chart()

        self.root.after(150, self.draw_canvas.focus_set)

    def configure_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("App.TFrame", background=self.BACKGROUND)
        style.configure("Panel.TFrame", background=self.PANEL)
        style.configure("PanelAlt.TFrame", background=self.PANEL_ALT)
        style.configure(
            "Title.TLabel",
            background=self.BACKGROUND,
            foreground=self.TEXT,
            font=("Segoe UI", 20, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background=self.BACKGROUND,
            foreground=self.MUTED,
            font=("Segoe UI", 10),
        )
        style.configure(
            "PanelTitle.TLabel",
            background=self.PANEL,
            foreground=self.TEXT,
            font=("Segoe UI", 14, "bold"),
        )
        style.configure(
            "Body.TLabel",
            background=self.PANEL,
            foreground=self.TEXT,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Muted.TLabel",
            background=self.PANEL,
            foreground=self.MUTED,
            font=("Segoe UI", 9),
        )
        style.configure(
            "Status.TLabel",
            background=self.BACKGROUND,
            foreground=self.TEXT,
            font=("Segoe UI", 10),
        )
        style.configure(
            "TButton",
            padding=(10, 7),
            font=("Segoe UI", 10, "bold"),
        )
        style.configure("TRadiobutton", background=self.PANEL, foreground=self.TEXT)
        style.map(
            "TRadiobutton",
            background=[("active", self.PANEL)],
            foreground=[("active", self.TEXT)],
        )
        style.configure("Horizontal.TScale", background=self.PANEL)

    def build_interface(self) -> None:
        header = ttk.Frame(self.root, style="App.TFrame")
        header.pack(fill="x", padx=18, pady=(14, 8))

        ttk.Label(
            header,
            text="D-RAMZI Photonic KAN - MNIST ao vivo",
            style="Title.TLabel",
        ).pack(anchor="w")
        ttk.Label(
            header,
            text=(
                "Desenhe com mouse/caneta, carregue uma foto ou use a camera. "
                f"A rede responde apos {self.idle_ms / 1000.0:.1f} s sem atividade."
            ),
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        body = ttk.Panedwindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=18, pady=8)

        self.left_panel = ttk.Frame(body, style="Panel.TFrame", padding=14)
        self.right_panel = ttk.Frame(body, style="Panel.TFrame", padding=14)
        body.add(self.left_panel, weight=1)
        body.add(self.right_panel, weight=1)

        self.build_left_panel()
        self.build_right_panel()

        footer = ttk.Frame(self.root, style="App.TFrame")
        footer.pack(fill="x", padx=18, pady=(4, 12))
        ttk.Label(footer, textvariable=self.status_text, style="Status.TLabel").pack(
            side="left"
        )
        ttk.Label(
            footer,
            textvariable=self.countdown_text,
            style="Status.TLabel",
        ).pack(side="right")

    def build_left_panel(self) -> None:
        ttk.Label(
            self.left_panel, text="Entrada", style="PanelTitle.TLabel"
        ).pack(anchor="w")
        ttk.Label(
            self.left_panel,
            text=(
                "Use apenas um digito. Botao esquerdo desenha; botao direito apaga."
            ),
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(2, 10))

        self.draw_canvas = tk.Canvas(
            self.left_panel,
            width=self.canvas_size,
            height=self.canvas_size,
            bg=self.CANVAS_BG,
            highlightthickness=2,
            highlightbackground="#555b66",
            highlightcolor=self.ACCENT,
            cursor="crosshair",
        )
        self.draw_canvas.pack(anchor="center")
        self.draw_canvas.bind("<ButtonPress-1>", self.on_left_press)
        self.draw_canvas.bind("<B1-Motion>", self.on_left_drag)
        self.draw_canvas.bind("<ButtonRelease-1>", self.on_release)
        self.draw_canvas.bind("<ButtonPress-3>", self.on_right_press)
        self.draw_canvas.bind("<B3-Motion>", self.on_right_drag)
        self.draw_canvas.bind("<ButtonRelease-3>", self.on_release)

        first_controls = ttk.Frame(self.left_panel, style="Panel.TFrame")
        first_controls.pack(fill="x", pady=(12, 5))

        ttk.Button(first_controls, text="Limpar", command=self.clear_canvas).pack(
            side="left", padx=(0, 5)
        )
        ttk.Button(first_controls, text="Desfazer", command=self.undo).pack(
            side="left", padx=5
        )
        ttk.Button(first_controls, text="Predizer agora", command=self.predict_now).pack(
            side="left", padx=5
        )
        ttk.Button(first_controls, text="Abrir foto", command=self.open_image).pack(
            side="left", padx=5
        )

        camera_state = "normal" if CV2_AVAILABLE else "disabled"
        self.camera_button = ttk.Button(
            first_controls,
            text="Camera",
            command=self.open_camera,
            state=camera_state,
        )
        self.camera_button.pack(side="left", padx=(5, 0))

        second_controls = ttk.Frame(self.left_panel, style="Panel.TFrame")
        second_controls.pack(fill="x", pady=(5, 0))

        ttk.Radiobutton(
            second_controls,
            text="Caneta",
            variable=self.tool_mode,
            value="pen",
        ).pack(side="left")
        ttk.Radiobutton(
            second_controls,
            text="Borracha",
            variable=self.tool_mode,
            value="eraser",
        ).pack(side="left", padx=(8, 14))

        ttk.Label(second_controls, text="Espessura", style="Body.TLabel").pack(
            side="left"
        )
        ttk.Scale(
            second_controls,
            from_=10,
            to=max(80, self.canvas_size // 6),
            variable=self.brush_size,
            orient="horizontal",
            length=150,
        ).pack(side="left", padx=8)

        ttk.Button(
            second_controls,
            text="Colar imagem",
            command=self.paste_from_clipboard,
        ).pack(side="right", padx=(5, 0))
        ttk.Button(
            second_controls,
            text="Salvar 28x28",
            command=self.save_preprocessed_input,
        ).pack(side="right", padx=5)

        if not CV2_AVAILABLE:
            ttk.Label(
                self.left_panel,
                text="Camera opcional: pip install opencv-python",
                style="Muted.TLabel",
            ).pack(anchor="w", pady=(7, 0))

    def build_right_panel(self) -> None:
        ttk.Label(
            self.right_panel, text="Predicao da rede", style="PanelTitle.TLabel"
        ).pack(anchor="w")

        result_top = ttk.Frame(self.right_panel, style="Panel.TFrame")
        result_top.pack(fill="x", pady=(8, 6))

        self.predicted_digit_label = tk.Label(
            result_top,
            text="-",
            bg=self.PANEL,
            fg=self.ACCENT,
            font=("Segoe UI", 112, "bold"),
            width=3,
            anchor="center",
        )
        self.predicted_digit_label.pack(side="left", padx=(8, 24))

        result_text = ttk.Frame(result_top, style="Panel.TFrame")
        result_text.pack(side="left", fill="both", expand=True)

        self.confidence_label = tk.Label(
            result_text,
            text="Aguardando um digito",
            bg=self.PANEL,
            fg=self.TEXT,
            font=("Segoe UI", 19, "bold"),
            anchor="w",
            justify="left",
        )
        self.confidence_label.pack(anchor="w", pady=(24, 6))

        self.top3_label = tk.Label(
            result_text,
            text="",
            bg=self.PANEL,
            fg=self.MUTED,
            font=("Consolas", 11),
            anchor="w",
            justify="left",
        )
        self.top3_label.pack(anchor="w")

        middle = ttk.Frame(self.right_panel, style="Panel.TFrame")
        middle.pack(fill="both", expand=True, pady=(4, 4))
        middle.columnconfigure(0, weight=0)
        middle.columnconfigure(1, weight=1)
        middle.rowconfigure(0, weight=1)

        preview_frame = ttk.Frame(middle, style="PanelAlt.TFrame", padding=8)
        preview_frame.grid(row=0, column=0, sticky="nsw", padx=(0, 10))
        ttk.Label(
            preview_frame,
            text="Entrada real 28 x 28",
            style="PanelTitle.TLabel",
        ).pack()
        self.preview_canvas = tk.Canvas(
            preview_frame,
            width=224,
            height=224,
            bg="#000000",
            highlightthickness=0,
        )
        self.preview_canvas.pack(pady=(8, 0))

        chart_frame = ttk.Frame(middle, style="PanelAlt.TFrame", padding=8)
        chart_frame.grid(row=0, column=1, sticky="nsew")
        ttk.Label(
            chart_frame,
            text="Saidas softmax",
            style="PanelTitle.TLabel",
        ).pack(anchor="w")
        self.probability_canvas = tk.Canvas(
            chart_frame,
            bg=self.PANEL_ALT,
            highlightthickness=0,
            height=300,
        )
        self.probability_canvas.pack(fill="both", expand=True, pady=(5, 0))
        self.probability_canvas.bind(
            "<Configure>", lambda _event: self.redraw_probability_chart()
        )

        model_info = self.format_model_info()
        self.model_info_label = tk.Label(
            self.right_panel,
            text=model_info,
            bg=self.PANEL,
            fg=self.MUTED,
            font=("Consolas", 9),
            anchor="w",
            justify="left",
        )
        self.model_info_label.pack(fill="x", pady=(8, 0))

    def format_model_info(self) -> str:
        epoch = (
            str(self.metadata.checkpoint_epoch)
            if self.metadata.checkpoint_epoch is not None
            else "n/a"
        )
        validation = (
            f"{100.0 * self.metadata.validation_accuracy:.2f}%"
            if self.metadata.validation_accuracy is not None
            else "n/a"
        )
        return (
            f"Modelo: D-RAMZI Photonic KAN {self.metadata.architecture}\n"
            f"Checkpoint: {self.metadata.checkpoint_kind} | epoch {epoch} | "
            f"val {validation}\n"
            f"Bases fisicas: {self.metadata.n_bases} | "
            f"LUT: {self.metadata.lut_points} pontos\n"
            f"Dispositivo: {self.metadata.device_name}"
        )

    def bind_shortcuts(self) -> None:
        self.root.bind("<Control-l>", lambda _event: self.clear_canvas())
        self.root.bind("<Control-z>", lambda _event: self.undo())
        self.root.bind("<Control-o>", lambda _event: self.open_image())
        self.root.bind("<Control-v>", lambda _event: self.paste_from_clipboard())
        self.root.bind("<Return>", lambda _event: self.predict_now())
        self.root.bind("<Escape>", lambda _event: self.clear_canvas())

    def push_history(self) -> None:
        self.history.append(self.canvas_image.copy())
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def undo(self) -> None:
        if not self.history:
            self.status_text.set("Nao ha nada para desfazer.")
            return
        self.cancel_pending_prediction()
        self.canvas_image = self.history.pop()
        self.canvas_draw = ImageDraw.Draw(self.canvas_image)
        self.external_source = None
        self.refresh_canvas()
        self.schedule_prediction()

    def clear_canvas(self) -> None:
        self.cancel_pending_prediction()
        self.canvas_image = Image.new(
            "L", (self.canvas_size, self.canvas_size), color=0
        )
        self.canvas_draw = ImageDraw.Draw(self.canvas_image)
        self.external_source = None
        self.history.clear()
        self.last_prediction = None
        self.refresh_canvas()
        self.clear_result_panel()
        self.status_text.set("Quadro limpo. Desenhe um unico digito de 0 a 9.")
        self.countdown_text.set("")
        self.draw_canvas.focus_set()

    def clear_result_panel(self) -> None:
        self.predicted_digit_label.configure(text="-", fg=self.ACCENT)
        self.confidence_label.configure(text="Aguardando um digito")
        self.top3_label.configure(text="")
        self.preview_canvas.delete("all")
        self.preview_photo_reference = None
        self.draw_empty_probability_chart()

    def on_left_press(self, event: tk.Event) -> None:
        self.begin_stroke(event, forced_mode=None)

    def on_right_press(self, event: tk.Event) -> None:
        self.begin_stroke(event, forced_mode="eraser")

    def begin_stroke(self, event: tk.Event, forced_mode: Optional[str]) -> None:
        self.cancel_pending_prediction()
        self.push_history()
        self.external_source = None
        self.drawing = True
        self.last_x = int(np.clip(event.x, 0, self.canvas_size - 1))
        self.last_y = int(np.clip(event.y, 0, self.canvas_size - 1))
        mode = forced_mode or self.tool_mode.get()
        self.draw_dot(self.last_x, self.last_y, mode)
        self.refresh_canvas()
        self.schedule_prediction()

    def on_left_drag(self, event: tk.Event) -> None:
        self.continue_stroke(event, forced_mode=None)

    def on_right_drag(self, event: tk.Event) -> None:
        self.continue_stroke(event, forced_mode="eraser")

    def continue_stroke(self, event: tk.Event, forced_mode: Optional[str]) -> None:
        x = int(np.clip(event.x, 0, self.canvas_size - 1))
        y = int(np.clip(event.y, 0, self.canvas_size - 1))

        if self.last_x is None or self.last_y is None:
            self.last_x, self.last_y = x, y

        mode = forced_mode or self.tool_mode.get()
        self.draw_segment(self.last_x, self.last_y, x, y, mode)
        self.last_x, self.last_y = x, y
        self.external_source = None
        self.refresh_canvas()
        self.schedule_prediction()

    def on_release(self, _event: tk.Event) -> None:
        self.drawing = False
        self.last_x = None
        self.last_y = None
        self.schedule_prediction()

    def tool_color_and_width(self, mode: str) -> Tuple[int, int]:
        width = int(round(self.brush_size.get()))
        if mode == "eraser":
            return 0, max(width, int(round(1.35 * width)))
        return 255, width

    def draw_dot(self, x: int, y: int, mode: str) -> None:
        color, width = self.tool_color_and_width(mode)
        radius = max(1, width // 2)
        self.canvas_draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius), fill=color
        )

    def draw_segment(
        self, x0: int, y0: int, x1: int, y1: int, mode: str
    ) -> None:
        color, width = self.tool_color_and_width(mode)
        self.canvas_draw.line((x0, y0, x1, y1), fill=color, width=width)
        radius = max(1, width // 2)
        self.canvas_draw.ellipse(
            (x1 - radius, y1 - radius, x1 + radius, y1 + radius), fill=color
        )

    def refresh_canvas(self) -> None:
        display = self.canvas_image.convert("RGB")
        self.canvas_photo_reference = ImageTk.PhotoImage(display, master=self.draw_canvas)
        self.draw_canvas.delete("all")
        self.draw_canvas.create_image(0, 0, anchor="nw", image=self.canvas_photo_reference)

    def set_external_image(self, image: Image.Image, source_name: str) -> None:
        self.cancel_pending_prediction()
        self.push_history()
        image = ImageOps.exif_transpose(image).copy()
        self.external_source = image
        self.canvas_image = fit_foreground_to_canvas(image, self.canvas_size)
        self.canvas_draw = ImageDraw.Draw(self.canvas_image)
        self.refresh_canvas()
        self.status_text.set(
            f"{source_name} carregada. Aguardando {self.idle_ms / 1000.0:.1f} s..."
        )
        self.schedule_prediction()

    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Escolha uma foto com um unico digito",
            filetypes=(
                ("Imagens", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                ("Todos os arquivos", "*.*"),
            ),
        )
        if not path:
            return
        try:
            with Image.open(path) as image:
                self.set_external_image(image.copy(), Path(path).name)
        except Exception as exc:
            messagebox.showerror("Erro ao abrir imagem", str(exc), parent=self.root)

    def paste_from_clipboard(self) -> None:
        try:
            content = ImageGrab.grabclipboard()
        except Exception as exc:
            self.status_text.set(f"Nao foi possivel ler a area de transferencia: {exc}")
            return

        if isinstance(content, Image.Image):
            self.set_external_image(content, "Imagem da area de transferencia")
            return

        if isinstance(content, list) and content:
            first_path = Path(content[0])
            if first_path.exists():
                try:
                    with Image.open(first_path) as image:
                        self.set_external_image(image.copy(), first_path.name)
                    return
                except Exception:
                    pass

        self.status_text.set("A area de transferencia nao contem uma imagem.")

    def open_camera(self) -> None:
        if not CV2_AVAILABLE:
            messagebox.showinfo(
                "Camera indisponivel",
                "Instale o suporte opcional com:\n\npip install opencv-python",
                parent=self.root,
            )
            return

        try:
            CameraDialog(
                self.root,
                camera_index=self.camera_index,
                on_capture=lambda image: self.set_external_image(
                    image, "Foto da camera"
                ),
            )
        except Exception as exc:
            messagebox.showerror("Erro de camera", str(exc), parent=self.root)

    def cancel_pending_prediction(self) -> None:
        if self.pending_prediction_id is not None:
            try:
                self.root.after_cancel(self.pending_prediction_id)
            except tk.TclError:
                pass
            self.pending_prediction_id = None

        if self.countdown_id is not None:
            try:
                self.root.after_cancel(self.countdown_id)
            except tk.TclError:
                pass
            self.countdown_id = None

        self.prediction_deadline = None
        self.countdown_text.set("")

    def schedule_prediction(self) -> None:
        self.cancel_pending_prediction()
        self.prediction_deadline = time.perf_counter() + self.idle_ms / 1000.0
        self.pending_prediction_id = self.root.after(
            self.idle_ms, self.auto_predict
        )
        self.update_countdown()

    def update_countdown(self) -> None:
        if self.prediction_deadline is None:
            return
        remaining = self.prediction_deadline - time.perf_counter()
        if remaining <= 0:
            self.countdown_text.set("Analisando...")
            return
        self.countdown_text.set(f"Predicao automatica em {remaining:.1f} s")
        self.countdown_id = self.root.after(100, self.update_countdown)

    def auto_predict(self) -> None:
        self.pending_prediction_id = None
        self.countdown_id = None
        self.prediction_deadline = None
        self.predict_now()

    def current_source(self) -> Image.Image:
        if self.external_source is not None:
            return self.external_source
        return self.canvas_image

    def predict_now(self) -> None:
        self.cancel_pending_prediction()
        self.status_text.set("Executando inferencia...")
        self.countdown_text.set("")
        self.root.update_idletasks()

        try:
            prediction = self.recognizer.predict(self.current_source())
        except EmptyDigitError as exc:
            self.status_text.set(str(exc))
            self.clear_result_panel()
            return
        except Exception as exc:
            self.status_text.set("Falha durante a inferencia.")
            messagebox.showerror("Erro de inferencia", str(exc), parent=self.root)
            return

        self.last_prediction = prediction
        self.update_result_panel(prediction)
        self.status_text.set(
            f"Predicao concluida em {prediction.inference_ms:.2f} ms. "
            "Continue desenhando para executar novamente."
        )

    def update_result_panel(self, prediction: Prediction) -> None:
        self.predicted_digit_label.configure(text=str(prediction.digit), fg=self.SUCCESS)
        self.confidence_label.configure(
            text=(
                f"Eu acho que e: {prediction.digit}\n"
                f"Score softmax: {100.0 * prediction.confidence:.2f}%"
            )
        )

        top_indices = np.argsort(prediction.probabilities)[::-1][:3]
        top_lines = ["Top 3 alternativas:"]
        for rank, index in enumerate(top_indices, start=1):
            top_lines.append(
                f"{rank}. {int(index)}   {100.0 * prediction.probabilities[index]:6.2f}%"
            )
        self.top3_label.configure(text="\n".join(top_lines))

        preview = prediction.input_preview.resize((224, 224), nearest_resampling())
        self.preview_photo_reference = ImageTk.PhotoImage(preview, master=self.preview_canvas)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(0, 0, anchor="nw", image=self.preview_photo_reference)
        self.redraw_probability_chart()

    def draw_empty_probability_chart(self) -> None:
        self.probability_canvas.delete("all")
        self.probability_canvas.create_text(
            20,
            25,
            anchor="nw",
            text="As dez saidas aparecerao aqui.",
            fill=self.MUTED,
            font=("Segoe UI", 11),
        )

    def redraw_probability_chart(self) -> None:
        if self.last_prediction is None:
            self.draw_empty_probability_chart()
            return

        canvas = self.probability_canvas
        canvas.delete("all")
        width = max(360, canvas.winfo_width())
        height = max(260, canvas.winfo_height())

        left = 42
        right = 66
        top = 10
        bottom = 10
        row_height = (height - top - bottom) / 10.0
        max_bar_width = max(30, width - left - right)

        probabilities = self.last_prediction.probabilities
        predicted = self.last_prediction.digit

        for digit in range(10):
            y0 = top + digit * row_height + 3
            y1 = top + (digit + 1) * row_height - 3
            probability = float(probabilities[digit])
            x1 = left + max_bar_width * probability

            canvas.create_text(
                left - 12,
                (y0 + y1) / 2,
                text=str(digit),
                fill=self.TEXT,
                font=("Segoe UI", 10, "bold"),
                anchor="e",
            )
            canvas.create_rectangle(
                left,
                y0,
                left + max_bar_width,
                y1,
                fill="#3a3f48",
                outline="",
            )
            color = self.SUCCESS if digit == predicted else self.ACCENT
            canvas.create_rectangle(
                left,
                y0,
                x1,
                y1,
                fill=color,
                outline="",
            )
            canvas.create_text(
                width - 8,
                (y0 + y1) / 2,
                text=f"{100.0 * probability:6.2f}%",
                fill=self.TEXT,
                font=("Consolas", 9),
                anchor="e",
            )

    def save_preprocessed_input(self) -> None:
        try:
            _tensor, preview = preprocess_to_mnist(self.current_source())
        except EmptyDigitError as exc:
            self.status_text.set(str(exc))
            return

        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Salvar a entrada 28x28",
            defaultextension=".png",
            initialfile="mnist_input_28x28.png",
            filetypes=(("PNG", "*.png"), ("Todos os arquivos", "*.*")),
        )
        if not path:
            return

        preview.save(path)
        self.status_text.set(f"Entrada 28x28 salva em: {path}")


# ============================================================================
# 7. COMMAND-LINE AND STARTUP
# ============================================================================


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive MNIST GUI for a trained D-RAMZI Photonic KAN."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("dramzi_photonic_kan_results"),
        help="Folder containing configuration.json, physical_basis/, and MNIST/.",
    )
    parser.add_argument(
        "--checkpoint",
        choices=("best", "final"),
        default="best",
        help="Use best_validation_model.pt or final_model.pt.",
    )
    parser.add_argument(
        "--idle-ms",
        type=int,
        default=2000,
        help="Milliseconds without drawing before automatic inference.",
    )
    parser.add_argument(
        "--canvas-size",
        type=int,
        default=520,
        help="Internal drawing-canvas size in pixels.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV camera index.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference even when CUDA is available.",
    )
    return parser.parse_args()


def enable_windows_dpi_awareness() -> None:
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def resolve_results_root(requested: Path) -> Optional[Path]:
    requested = requested.expanduser()
    expected_lut = requested / "physical_basis" / "lut_curves.npy"
    expected_mnist = requested / "MNIST"
    if requested.exists() and expected_lut.exists() and expected_mnist.exists():
        return requested.resolve()

    selector = tk.Tk()
    selector.withdraw()
    selector.update_idletasks()
    selected = filedialog.askdirectory(
        parent=selector,
        title="Selecione a pasta dramzi_photonic_kan_results",
        initialdir=str(requested.parent if requested.parent.exists() else Path.cwd()),
    )
    selector.destroy()

    if not selected:
        return None
    return Path(selected).resolve()


def main() -> None:
    enable_windows_dpi_awareness()
    args = parse_arguments()

    results_root = resolve_results_root(args.results_root)
    if results_root is None:
        print("No results folder selected.")
        return

    print("=" * 78)
    print("D-RAMZI PHOTONIC KAN - INTERACTIVE MNIST GUI")
    print("=" * 78)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Results root: {results_root}")

    try:
        model, device, metadata = load_trained_model(
            results_root=results_root,
            checkpoint_kind=args.checkpoint,
            force_cpu=args.cpu,
        )
    except Exception as exc:
        error_root = tk.Tk()
        error_root.withdraw()
        messagebox.showerror("Erro ao carregar o modelo", str(exc), parent=error_root)
        error_root.destroy()
        raise

    print(f"Device: {device}")
    print(f"Device name: {metadata.device_name}")
    print(f"Checkpoint: {metadata.checkpoint_path}")
    print(f"Architecture: {metadata.architecture}")
    print(f"D-RAMZI bases: {metadata.n_bases}")
    print(f"LUT points: {metadata.lut_points}")

    root = tk.Tk()
    recognizer = DigitRecognizer(model, device)
    DRAMZIDigitApp(
        root=root,
        recognizer=recognizer,
        metadata=metadata,
        idle_ms=args.idle_ms,
        canvas_size=args.canvas_size,
        camera_index=args.camera_index,
    )
    root.mainloop()


if __name__ == "__main__":
    main()

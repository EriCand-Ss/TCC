"""Differentiable, edge-wise D-RAMZI Photonic KAN for PyTorch.

Each connection owns one physical D-RAMZI response with nine trainable
controls: two MRR phase offsets, two MRR amplitude transmissions, four MZI
phase settings, and one SOA gain. A layer evaluates one univariate response per
edge and aggregates incoming responses at each output node.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn


# Device and training constants used by the supplied physical model.
LAMBDA0 = 1550e-9
NEFF0 = 2.3537
NG = 2.3537
FSR_NM = 20e-9
RING_LENGTH = 2.0 * LAMBDA0**2 / (NG * FSR_NM)

WVG_LENGTHS = (
    4.0 * LAMBDA0**2 / (NG * FSR_NM),
    50e-6 + 2.0 * LAMBDA0**2 / (NG * FSR_NM),
    50e-6,
    4.0 * LAMBDA0**2 / (NG * FSR_NM),
    50e-6,
    50e-6 + 2.0 * LAMBDA0**2 / (NG * FSR_NM),
)
ETA_FC = 1e-6
GAMMA_FC = 1e10
MU = 1e5
TAU_FC = 5e-9
XI = 1e40
DETUNING = 2.0 * math.pi * 2e9
P_MAX = 0.2e-3
SOA_GAIN_MIN = 1.0
SOA_GAIN_MAX = 20.0
MRR_ATTENUATION_MIN = 0.70
MRR_ATTENUATION_MAX = 0.99
N_TRAINABLE_PARAMETERS_PER_EDGE = 9

R_COUPLER = 1.0 / math.sqrt(2.0)
T_COUPLER = math.sqrt(1.0 - R_COUPLER**2)
SQRT2 = math.sqrt(2.0)


def steady_state_carrier_density(
    input_power: np.ndarray | float,
    detuning: float = DETUNING,
    newton_iterations: int = 14,
) -> np.ndarray | float:
    """Vectorized stationary solution of the supplied carrier-rate model."""
    power = np.asarray(input_power, dtype=float)
    scalar_input = power.ndim == 0
    power = np.atleast_1d(np.maximum(power, 0.0))

    coefficient = TAU_FC * XI * MU**4 * power**2
    base_denominator = GAMMA_FC**2 + detuning**2
    upper = coefficient / base_denominator**2
    density = upper.copy()

    for _ in range(newton_iterations):
        shifted_detuning = detuning + ETA_FC * density
        denominator = GAMMA_FC**2 + shifted_detuning**2
        residual = density - coefficient / denominator**2
        derivative = 1.0 + 4.0 * coefficient * ETA_FC * shifted_detuning / denominator**3
        proposal = density - residual / derivative
        density = np.clip(proposal, 0.0, upper)

    density[power <= 0.0] = 0.0
    return float(density[0]) if scalar_input else density


def make_carrier_density_lut(
    n_points: int = 4097,
    max_power: float = P_MAX * SOA_GAIN_MAX,
) -> np.ndarray:
    """Precompute the stationary carrier response for differentiable lookup."""
    if n_points < 2:
        raise ValueError("n_points must be at least 2.")
    power_grid = np.linspace(0.0, float(max_power), int(n_points), dtype=np.float64)
    return np.asarray(steady_state_carrier_density(power_grid), dtype=np.float32)


class CarrierDensityLUT(nn.Module):
    """Piecewise-linear, differentiable lookup of stationary carrier density."""

    def __init__(self, values: Sequence[float] | np.ndarray, max_power: float) -> None:
        super().__init__()
        table = torch.as_tensor(np.asarray(values, dtype=np.float32).copy())
        if table.ndim != 1 or table.numel() < 2:
            raise ValueError("carrier LUT must be a one-dimensional array of length >= 2.")
        self.register_buffer("table", table)
        self.max_power = float(max_power)

    def forward(self, power: torch.Tensor) -> torch.Tensor:
        power = torch.clamp(power, min=0.0, max=self.max_power)
        position = power * ((self.table.numel() - 1) / self.max_power)
        lower = torch.floor(position).to(torch.long)
        lower = lower.clamp(0, self.table.numel() - 2)
        upper = lower + 1
        fraction = position - lower.to(position.dtype)
        low_value = self.table[lower]
        high_value = self.table[upper]
        return low_value + fraction * (high_value - low_value)


def _waveguide_transfer(length: float) -> complex:
    beta = 2.0 * math.pi * NEFF0 / LAMBDA0
    return complex(np.exp(-1j * beta * length))


_H_WVG1, _H_WVG2, _H_WVG3, _H_WVG4, _H_WVG5, _H_WVG6 = (
    _waveguide_transfer(length) for length in WVG_LENGTHS
)


def _mrr_transfer(
    carrier_density: torch.Tensor,
    ring_phase: torch.Tensor,
    attenuation: torch.Tensor,
) -> torch.Tensor:
    """Complex through-port transfer of one tunable nonlinear MRR."""
    effective_index = NEFF0 - ETA_FC * carrier_density
    beta = (2.0 * math.pi / LAMBDA0) * effective_index
    round_trip = torch.exp(-1j * (beta * RING_LENGTH + ring_phase))
    numerator = R_COUPLER - attenuation * round_trip
    denominator = 1.0 - R_COUPLER * attenuation * round_trip
    return numerator / denominator


def _dramzi_edge_response(
    normalized_input: torch.Tensor,
    phases: torch.Tensor,
    attenuation_logits: torch.Tensor,
    gain_logits: torch.Tensor,
    carrier_lut: CarrierDensityLUT,
) -> torch.Tensor:
    """Return normalized optical power for a batch of D-RAMZI edges.

    Shapes:
        normalized_input: [batch, input]
        phases: [output_chunk, input, 6]
        attenuation_logits: [output_chunk, input, 2]
        gain_logits: [output_chunk, input]
        result: [batch, output_chunk, input]
    """
    x = torch.clamp(normalized_input, 0.0, 1.0).unsqueeze(1)
    input_power = x * P_MAX
    input_field = torch.sqrt(input_power.clamp_min(1e-24)).to(torch.complex64)

    attenuation = MRR_ATTENUATION_MIN + (
        MRR_ATTENUATION_MAX - MRR_ATTENUATION_MIN
    ) * torch.sigmoid(attenuation_logits)
    gain = SOA_GAIN_MIN + (SOA_GAIN_MAX - SOA_GAIN_MIN) * torch.sigmoid(gain_logits)

    # First RAMZI: nonlinear MRR followed by an MZI.
    nc1 = carrier_lut(input_power)
    h_mrr1 = _mrr_transfer(
        nc1,
        phases[None, :, :, 0],
        attenuation[None, :, :, 0],
    )
    arm_top = h_mrr1 * input_field / SQRT2
    arm_bottom = _H_WVG1 * input_field / SQRT2
    arm_bottom = arm_bottom * torch.exp(1j * phases[None, :, :, 2])

    after_top = R_COUPLER * arm_top - 1j * T_COUPLER * arm_bottom
    after_bottom = -1j * T_COUPLER * arm_top + R_COUPLER * arm_bottom
    arm_top = _H_WVG2 * after_top
    arm_bottom = _H_WVG3 * after_bottom * torch.exp(1j * phases[None, :, :, 3])
    field_after_ramzi1 = (arm_top + arm_bottom) / SQRT2

    # Interstage SOA and second RAMZI.
    field_into_ramzi2 = torch.sqrt(gain[None, :, :]).to(torch.complex64) * field_after_ramzi1
    power_into_ramzi2 = field_into_ramzi2.real.square() + field_into_ramzi2.imag.square()
    nc2 = carrier_lut(power_into_ramzi2)
    h_mrr2 = _mrr_transfer(
        nc2,
        phases[None, :, :, 1],
        attenuation[None, :, :, 1],
    )
    arm_top = h_mrr2 * field_into_ramzi2 / SQRT2
    arm_bottom = _H_WVG4 * field_into_ramzi2 / SQRT2
    arm_bottom = arm_bottom * torch.exp(1j * phases[None, :, :, 4])

    after_top = R_COUPLER * arm_top - 1j * T_COUPLER * arm_bottom
    after_bottom = -1j * T_COUPLER * arm_top + R_COUPLER * arm_bottom
    arm_top = _H_WVG5 * after_top
    arm_bottom = _H_WVG6 * after_bottom * torch.exp(1j * phases[None, :, :, 5])
    output_field = (arm_top + arm_bottom) / SQRT2

    output_power = output_field.real.square() + output_field.imag.square()
    # Express the output in units of the D-RAMZI input power range.
    # The SOA can make this ratio exceed one, as expected physically.
    return output_power / P_MAX


class DRAMZIKANLayer(nn.Module):
    """A KAN layer with one independently trainable D-RAMZI on every edge.

    For output node j, the layer computes
        y_j = sum_i phi_{j,i}(x_i) + bias_j.
    Each phi is the normalized optical power response of one physical edge.
    The hidden-layer output is normalized at the layer boundary before it is
    encoded as the next layer's optical input.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        output_chunk_size: int = 8,
    ) -> None:
        super().__init__()
        if in_features < 1 or out_features < 1:
            raise ValueError("in_features and out_features must be positive.")
        if output_chunk_size < 1:
            raise ValueError("output_chunk_size must be positive.")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.output_chunk_size = int(output_chunk_size)

        # Six phases: MRR resonance phases (2) + MZI phases (4).
        self.phases = nn.Parameter(torch.empty(out_features, in_features, 6))
        # Two MRR amplitude transmissions, bounded to a passive range.
        self.attenuation_logits = nn.Parameter(torch.empty(out_features, in_features, 2))
        # One SOA gain per D-RAMZI edge.
        self.gain_logits = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.reset_parameters()

    @staticmethod
    def _logit(probability: float) -> float:
        return math.log(probability / (1.0 - probability))

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.phases.uniform_(0.0, 2.0 * math.pi)
            alpha_fraction = (0.80 - MRR_ATTENUATION_MIN) / (
                MRR_ATTENUATION_MAX - MRR_ATTENUATION_MIN
            )
            self.attenuation_logits.fill_(self._logit(alpha_fraction))
            gain_fraction = (4.0 - SOA_GAIN_MIN) / (SOA_GAIN_MAX - SOA_GAIN_MIN)
            self.gain_logits.fill_(self._logit(gain_fraction))
            self.bias.zero_()

    def forward(
        self,
        x: torch.Tensor,
        carrier_lut: CarrierDensityLUT,
    ) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError(
                f"Expected input [batch, {self.in_features}], got {tuple(x.shape)}."
            )
        x = x.to(dtype=torch.float32).clamp(0.0, 1.0)
        output_chunks = []

        for start in range(0, self.out_features, self.output_chunk_size):
            stop = min(start + self.output_chunk_size, self.out_features)
            edge_values = _dramzi_edge_response(
                x,
                self.phases[start:stop],
                self.attenuation_logits[start:stop],
                self.gain_logits[start:stop],
                carrier_lut,
            )
            node_values = edge_values.sum(dim=-1)
            output_chunks.append(node_values)

        return torch.cat(output_chunks, dim=1) + self.bias

    def response_curve(
        self,
        x_values: torch.Tensor,
        output_index: int,
        input_index: int,
        carrier_lut: CarrierDensityLUT,
    ) -> torch.Tensor:
        """Evaluate one learned edge over a vector of normalized inputs."""
        if not (0 <= output_index < self.out_features):
            raise IndexError("output_index is out of range.")
        if not (0 <= input_index < self.in_features):
            raise IndexError("input_index is out of range.")
        x = x_values.reshape(-1, 1).to(dtype=torch.float32)
        return _dramzi_edge_response(
            x,
            self.phases[output_index : output_index + 1, input_index : input_index + 1],
            self.attenuation_logits[
                output_index : output_index + 1, input_index : input_index + 1
            ],
            self.gain_logits[output_index : output_index + 1, input_index : input_index + 1],
            carrier_lut,
        ).reshape(-1)


class DRAMZIPhotonicKAN(nn.Module):
    """Two-layer [784, 64, 10] Photonic KAN for digit classification."""

    def __init__(
        self,
        carrier_density_lut: Sequence[float] | np.ndarray,
        carrier_lut_max_power: float = P_MAX * SOA_GAIN_MAX,
        hidden_features: int = 64,
        n_classes: int = 10,
        output_chunk_size: int = 8,
        logit_scale: float | None = None,
    ) -> None:
        super().__init__()
        self.carrier_lut = CarrierDensityLUT(
            carrier_density_lut,
            max_power=carrier_lut_max_power,
        )
        self.layer1 = DRAMZIKANLayer(
            784,
            hidden_features,
            output_chunk_size=output_chunk_size,
        )
        self.layer2 = DRAMZIKANLayer(
            hidden_features,
            n_classes,
            output_chunk_size=output_chunk_size,
        )
        self.hidden_norm = nn.LayerNorm(hidden_features, elementwise_affine=False)
        self.logit_scale = (
            1.0 / math.sqrt(hidden_features)
            if logit_scale is None
            else float(logit_scale)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = images.flatten(start_dim=1).to(dtype=torch.float32)
        hidden_sum = self.layer1(x, self.carrier_lut)
        hidden = torch.sigmoid(self.hidden_norm(hidden_sum))
        logits = self.layer2(hidden, self.carrier_lut)
        return self.logit_scale * logits

    def count_edge_parameters(self) -> int:
        return N_TRAINABLE_PARAMETERS_PER_EDGE * (
            self.layer1.in_features * self.layer1.out_features
            + self.layer2.in_features * self.layer2.out_features
        )

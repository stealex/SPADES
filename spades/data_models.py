"""Typed containers used for SPADES data exchange."""

from dataclasses import dataclass
import numpy as np


@dataclass
class ScatteringWavefunctions:
    """Container for scattering-wavefunction datasets."""

    k_values: list[int]
    energy_grid: np.ndarray
    radial_grid: np.ndarray
    inner_phase_values: dict
    coulomb_phase_values: dict
    p_values: dict
    q_values: dict


@dataclass
class BoundWavefunctions:
    """Container for bound-wavefunction datasets."""

    radial_grid: np.ndarray
    binding_energies: dict
    p_values: dict
    q_values: dict

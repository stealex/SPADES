from abc import ABC, abstractmethod
from typing import Callable
import numpy as np
from scipy import integrate, interpolate

from spades.fermi_functions import FermiFunctions


class SpectrumBase(ABC):
    def __init__(self, sp_type: int, q_value: float, energy_points: np.ndarray) -> None:
        super().__init__()
        self.type = sp_type
        self.q_value = q_value
        self.energy_points = energy_points
        self.spectrum_values = {}

    @abstractmethod
    def compute_spectrum(self, eta_total: Callable | None):
        pass

    @abstractmethod
    def compute_psf(self, spectrum_integral):
        pass

    def integrate_spectrum(self, spectrum):
        psf_spl = interpolate.CubicSpline(
            self.energy_points, spectrum)
        result = integrate.quad(
            psf_spl, self.energy_points[0], self.energy_points[-1])

        if isinstance(result, tuple):
            psf = result[0]
        else:
            raise ValueError("PSF integration did not succeed")

        return psf

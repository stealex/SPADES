from abc import ABC, abstractmethod
from typing import Callable
import numpy as np
from scipy import integrate, interpolate

from spades.fermi_functions import FermiFunctions


class SpectrumBase(ABC):
    def __init__(self, total_ke: float, ei_ef: float) -> None:
        super().__init__()
        self.total_ke = total_ke
        self.ei_ef = ei_ef
        self.psfs = {}
        self.spectrum_values = {}
        self.spectrum_2D_values = {}
        self.spectrum_integrals = {}
        self.energy_points = None
        self.e1_grid_2D = None
        self.e2_grid_2D = None

    @abstractmethod
    def compute_spectrum(self, sp_type: int):
        pass

    def compute_2D_spectrum(self, sp_type: int):
        pass

    @abstractmethod
    def compute_psf(self):
        pass

    @abstractmethod
    def integrate_spectrum(self):
        pass


class BetaSpectrumBase(SpectrumBase):
    def __init__(self, total_ke: float, ei_ef: float, fermi_functions: FermiFunctions) -> None:
        super().__init__(total_ke, ei_ef)
        self.fermi_functions = fermi_functions

    @abstractmethod
    def compute_spectrum(self, sp_type: int):
        pass

    def compute_2D_spectrum(self, sp_type: int):
        pass

    @abstractmethod
    def compute_psf(self):
        pass

    @abstractmethod
    def integrate_spectrum(self):
        pass

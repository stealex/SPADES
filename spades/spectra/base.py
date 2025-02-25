from abc import ABC, abstractmethod
from typing import Callable
import numpy as np
from scipy import integrate, interpolate

from spades.fermi_functions import FermiFunctions


class SpectrumBase(ABC):
    def __init__(self, q_value: float) -> None:
        super().__init__()
        self.q_value = q_value

    @abstractmethod
    def compute_spectrum(self, sp_type: int):
        pass

    @abstractmethod
    def compute_psf(self):
        pass

    @abstractmethod
    def integrate_spectrum(self):
        pass
        return


class BetaSpectrumBase(SpectrumBase):
    def __init__(self, q_value: float, fermi_functions: FermiFunctions) -> None:
        super().__init__(q_value)
        self.q_value = q_value
        self.energy_points = None
        self.spectrum_values = {}
        self.fermi_functions = fermi_functions

    @abstractmethod
    def compute_spectrum(self, sp_type: int):
        pass

    @abstractmethod
    def compute_psf(self):
        pass

    @abstractmethod
    def integrate_spectrum(self):
        pass
        return

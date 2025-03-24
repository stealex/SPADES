from abc import ABC, abstractmethod
from typing import Callable
import numpy as np
from scipy import integrate, interpolate

from spades.fermi_functions import FermiFunctions


class SpectrumBase(ABC):
    """
    Abstract base class for spectra.
    """

    def __init__(self, total_ke: float, ei_ef: float) -> None:
        """_summary_

        Args:
            total_ke (float): total kinetic energy available in the process.
            ei_ef (float): energy difference between initial and final NUCLEAR levels.
        """
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
        """Abstract method for computation of spectra

        Args:
            sp_type (int): type of spectrum (e.g. ph.SINGLESPECTRUM, ph.SUMSPECTRUM, ph.ANGULARSPECTRUM)
        """
        pass

    @abstractmethod
    def compute_2D_spectrum(self, sp_type: int):
        """Abstract method for computation of 2D spectra

        Args:
            sp_type (int): type of spectrum (e.g. ph.SINGLESPECTRUM, ph.SUMSPECTRUM, ph.ANGULARSPECTRUM)
        """
        pass

    @abstractmethod
    def compute_psf(self):
        """Abstract method for the computation of PSF from spectrum integrals
        """
        pass

    @abstractmethod
    def integrate_spectrum(self):
        """Abstract method that integrates spectra
        """
        pass


class BetaSpectrumBase(SpectrumBase):
    """
    Abstract class for spectra involving emission.
    """

    def __init__(self, total_ke: float, ei_ef: float, fermi_functions: FermiFunctions) -> None:
        """
        Args:
            total_ke (float): total kinetic energy available in the process
            ei_ef (float): energy difference between initial and final NUCLEAR levels
            fermi_functions(FermiFunctions): Fermi functions to be used in the computation
        """
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

"""Abstract interfaces for spectra and phase-space-factor computations."""

from abc import ABC, abstractmethod
from typing import Callable
import numpy as np
from scipy import integrate, interpolate
from spades import ph

from spades.fermi_functions import FermiFunctions


class SpectrumBase(ABC):
    """Base interface shared by all spectra implementations."""

    def __init__(self, total_ke: float, ei_ef: float) -> None:
        """Initialize common energy scales and output containers.

        Args:
            total_ke (float): Total kinetic energy available in the process.
            ei_ef (float): Energy difference between initial and final nuclear levels.
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
    def compute_spectrum(self, sp_type: ph.SpectrumTypes):
        """Compute a 1D spectrum for the requested spectrum type.

        Args:
            sp_type (SpectrumTypes): Spectrum type selector.
        """
        pass

    @abstractmethod
    def compute_2D_spectrum(self, sp_type: ph.SpectrumTypes):
        """Compute a 2D spectrum for the requested spectrum type.

        Args:
            sp_type (ph.SpectrumTypes): Spectrum type selector.
        """
        pass

    @abstractmethod
    def compute_psf(self):
        """Compute phase-space factors from integrated spectra."""
        pass

    @abstractmethod
    def integrate_spectrum(self):
        """Integrate previously computed spectrum values."""
        pass


class BetaSpectrumBase(SpectrumBase):
    """Base class for channels with outgoing beta particles."""

    def __init__(self, total_ke: float, ei_ef: float, fermi_functions: FermiFunctions) -> None:
        """Initialize beta-spectrum settings.

        Args:
            total_ke (float): Total kinetic energy available in the process.
            ei_ef (float): Energy difference between initial and final nuclear levels.
            fermi_functions(FermiFunctions): Fermi-function backend for Coulomb effects.
        """
        super().__init__(total_ke, ei_ef)
        self.fermi_functions = fermi_functions

    @abstractmethod
    def compute_spectrum(self, sp_type: ph.SpectrumTypes):
        """Compute a 1D beta spectrum for ``sp_type``.

        Parameters
        ----------
        sp_type:
            Spectrum type selector.
        """
        pass

    @abstractmethod
    def compute_2D_spectrum(self, sp_type: ph.SpectrumTypes):
        """Compute a 2D beta spectrum for ``sp_type``.

        Parameters
        ----------
        sp_type:
            Spectrum type selector.
        """
        pass

    @abstractmethod
    def compute_psf(self):
        """Compute PSFs from integrated spectra."""
        pass

    @abstractmethod
    def integrate_spectrum(self):
        """Integrate and normalize previously computed spectra."""
        pass

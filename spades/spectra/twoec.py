"""Spectra/PSF building blocks for double-electron-capture channels."""

from abc import abstractmethod
import numpy as np
from scipy import integrate
from spades.fermi_functions import FermiFunctions
from spades.spectra.closure_helpers import neutrino_integrand_closure_standard_00, neutrino_integrand_closure_standard_02
from spades.spectra.taylor_helpers import integral_order
from spades.spectra.base import SpectrumBase
from spades.wavefunctions import BoundHandler
from spades import ph
from numba import njit


def neutrino_integrand_closure(enu_1, e_electron_1, e_electron_2, enu_2, enei, transition: ph.TransitionTypes):
    """Dispatch the closure neutrino kernel for 2EC by transition type.

    Parameters
    ----------
    enu_1, e_electron_1, e_electron_2, enu_2, enei:
        Energy terms entering the neutrino kernel.
    transition:
        Nuclear transition selector.

    Returns
    -------
    float
        Kernel value for the requested transition.
    """
    if transition == ph.TransitionTypes.ZEROPLUS_TO_TWOPLUS:
        return neutrino_integrand_closure_standard_02(enu_1, e_electron_1, e_electron_2, enu_2, enei)
    else:
        return neutrino_integrand_closure_standard_00(enu_1, e_electron_1, e_electron_2, enu_2, enei)


class TwoECSpectrum(SpectrumBase):
    """Base class for shell-resolved 2EC spectrum-integral models."""

    def __init__(self, total_ke: float, ei_ef: float, bound_handler: BoundHandler, nuclear_radius: float) -> None:
        """Allocate shell-pair containers for 2EC integrals and PSFs.

        Parameters
        ----------
        total_ke:
            Total kinetic energy available in the process.
        ei_ef:
            Nuclear-level energy difference.
        bound_handler:
            Bound-state wavefunction handler used for capture probabilities.
        nuclear_radius:
            Nuclear radius in fm.
        """
        super().__init__(total_ke, ei_ef)
        self.bound_handler = bound_handler
        self.nuclear_radius = nuclear_radius
        self.constant_in_front = 1.0

        # TODO: better naming
        self.spectrum_integrals = {}

        for n1 in self.bound_handler.p_grid:
            self.spectrum_integrals[n1] = {}
            for k1 in self.bound_handler.p_grid[n1]:
                if k1 != -1:
                    continue
                self.spectrum_integrals[n1][k1] = {}
                for n2 in self.bound_handler.p_grid:
                    if n2 < n1:
                        continue
                    self.spectrum_integrals[n1][k1][n2] = {}
                    for k2 in self.bound_handler.p_grid[n2]:
                        if k2 != -1:
                            continue
                        self.spectrum_integrals[n1][k1][n2][k2] = {}

    @abstractmethod
    def compute_spectrum(self, sp_type: int):
        """Compute shell-pair spectrum integrals for 2EC.

        Parameters
        ----------
        sp_type:
            Spectrum type selector (kept for interface compatibility).
        """
        pass

    def integrate_spectrum(self):
        """Not used for 2EC classes because shell integrals are computed directly."""
        raise NotImplementedError()

    def compute_psf(self):
        """Convert shell-pair integrals into shell-pair PSFs."""
        for n1 in self.spectrum_integrals:
            self.psfs[n1] = {-1: {}}
            for n2 in self.spectrum_integrals[n1][-1]:
                psf_mev = self.spectrum_integrals[n1][-1][n2][-1] * \
                    self.constant_in_front
                psf_years = psf_mev/(ph.hbar*np.log(2.))/(ph.year**(-1))

                self.psfs[n1][-1][n2] = {-1: psf_years}

    def compute_2D_spectrum(self, sp_type: ph.SpectrumTypes):
        """2D spectra are not implemented for 2EC classes.

        Parameters
        ----------
        sp_type:
            Requested spectrum type.
        """
        raise NotImplementedError()


class TwoECSpectrumClosure(TwoECSpectrum):
    """Closure-approximation implementation for 2nu double electron capture."""

    def __init__(self, total_ke: float, ei_ef: float, bound_handler: BoundHandler, nuclear_radius: float, enei: float, transition_type: ph.TransitionTypes) -> None:
        """Initialize closure constants and transition for 2EC.

        Parameters
        ----------
        total_ke, ei_ef:
            Process energy scales.
        bound_handler:
            Bound-state wavefunction handler.
        nuclear_radius:
            Nuclear radius in fm.
        enei:
            Closure parameter ``<E_N> - E_I``.
        transition_type:
            Nuclear transition selector.
        """
        super().__init__(total_ke, ei_ef, bound_handler, nuclear_radius)
        self.enei = enei
        self.atilde = self.enei+0.5*(self.total_ke-2*ph.electron_mass)
        self.constant_in_front = 2*(self.atilde**2.0)*((ph.fermi_coupling_constant*ph.v_ud)**4) /\
            (48.*(np.pi**3.0)) * (ph.electron_mass**4.0)
        self.transition_type = transition_type

    def compute_spectrum(self, sp_type: ph.SpectrumTypes = ph.SpectrumTypes.SINGLESPECTRUM):
        """Compute shell-pair spectrum integrals by neutrino-energy integration.

        Parameters
        ----------
        sp_type:
            Kept for interface compatibility.
        """
        for n1 in self.bound_handler.p_grid:
            prob1 = self.bound_handler.probability_in_sphere(
                self.nuclear_radius, n1, -1)
            for n2 in self.bound_handler.p_grid:
                if n2 < n1:
                    continue
                prob2 = self.bound_handler.probability_in_sphere(
                    self.nuclear_radius, n2, -1)

                eb1 = np.abs(self.bound_handler.be[n1][-1])
                eb2 = np.abs(self.bound_handler.be[n2][-1])
                integration_end = self.total_ke-eb1-eb2
                if integration_end < 0:
                    self.spectrum_integrals[n1][-1][n2][-1] = 0.
                    continue
                tmp_result = integrate.quad(
                    func=lambda x: neutrino_integrand_closure(
                        x,
                        -(ph.electron_mass - eb1),
                        -(ph.electron_mass - eb2),
                        self.total_ke-x-eb1-eb2,
                        self.enei,
                        self.transition_type),
                    a=0.,
                    b=self.total_ke-eb1-eb2
                )

                if not isinstance(tmp_result, tuple):
                    raise ValueError("Spectrum integration did not succeed")

                res = tmp_result[0]*prob1*prob2

                self.spectrum_integrals[n1][-1][n2][-1] = res


class TwoECSpectrumTaylor(TwoECSpectrum):
    """Taylor-expansion implementation for double-electron-capture integrals."""

    def __init__(self, total_ke: float, ei_ef: float, bound_handler: BoundHandler, nuclear_radius: float, order: ph.TaylorOrders, transition_type: ph.TransitionTypes) -> None:
        """Initialize Taylor order and transition for 2EC integrals.

        Parameters
        ----------
        total_ke, ei_ef:
            Process energy scales.
        bound_handler:
            Bound-state wavefunction handler.
        nuclear_radius:
            Nuclear radius in fm.
        order:
            Taylor expansion order.
        transition_type:
            Nuclear transition selector.
        """
        super().__init__(total_ke, ei_ef, bound_handler, nuclear_radius)
        self.order = order
        self.transition_type = transition_type

        self.constant_in_front = ((ph.fermi_coupling_constant*ph.v_ud)**4) /\
            (2.*(np.pi**3.0)) * (ph.electron_mass**4.0)

    def compute_spectrum(self, sp_type: ph.SpectrumTypes = ph.SpectrumTypes.SINGLESPECTRUM):
        """Compute shell-pair integrals from analytic Taylor kernels.

        Parameters
        ----------
        sp_type:
            Kept for interface compatibility.
        """
        for n1 in self.bound_handler.p_grid:
            prob1 = self.bound_handler.probability_in_sphere(
                self.nuclear_radius, n1, -1)

            for n2 in self.bound_handler.p_grid:
                if n2 < n1:
                    continue
                prob2 = self.bound_handler.probability_in_sphere(
                    self.nuclear_radius, n2, -1)

                eb1 = np.abs(self.bound_handler.be[n1][-1])
                eb2 = np.abs(self.bound_handler.be[n2][-1])

                integ_end = self.total_ke-eb1-eb2
                if integ_end < 0:
                    self.spectrum_integrals[n1][-1][n2][-1] = 0.
                    continue
                tmp_result = integral_order(eb1,
                                            eb2,
                                            self.total_ke,
                                            self.order,
                                            self.transition_type)

                res = tmp_result*prob1*prob2
                self.spectrum_integrals[n1][-1][n2][-1] = res

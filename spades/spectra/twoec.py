from abc import abstractmethod
from unittest import result
import numpy as np
from scipy import integrate
from spades.fermi_functions import FermiFunctions
from spades.math_stuff import kn
from spades.spectra.base import SpectrumBase
from spades.spectra_old import neutrino_integrand_standard
from spades.wavefunctions import BoundHandler
from spades import ph
from numba import njit


class TwoECSpectrum(SpectrumBase):
    def __init__(self, q_value: float, bound_handler: BoundHandler, nuclear_radius: float) -> None:
        super().__init__(q_value)
        self.bound_handler = bound_handler
        self.nuclear_radius = nuclear_radius

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
        pass

    def integrate_spectrum(self):
        raise NotImplementedError()

    @abstractmethod
    def compute_psf(self):
        pass


@njit
def kn_ecec(eb1: float, enu: float, w0: float, enei: float):
    return 1./(-(ph.electron_mass-eb1)+enu+enei) + 1./(ph.electron_mass-eb1 - enu+w0+enei)


class TwoECSpectrumClosure(TwoECSpectrum):
    def __init__(self, q_value: float, bound_handler: BoundHandler, nuclear_radius: float, enei: float) -> None:
        super().__init__(q_value, bound_handler, nuclear_radius)
        self.enei = enei
        self.atilde = self.enei+0.5*(self.q_value-2*ph.electron_mass)
        self.constant_in_front = 2*(self.atilde**2.0)*((ph.fermi_coupling_constant*ph.v_ud)**4) /\
            (48.*(np.pi**3.0)) * (ph.electron_mass**4.0)

    def compute_spectrum(self, sp_type: int = ph.SINGLESPECTRUM):
        print(f"q_value = {self.q_value}")
        for n1 in self.bound_handler.p_grid:
            prob1 = self.bound_handler.probability_in_sphere(
                self.nuclear_radius, n1, -1)
            print(prob1)
            for n2 in self.bound_handler.p_grid:
                if n2 < n1:
                    continue
                prob2 = self.bound_handler.probability_in_sphere(
                    self.nuclear_radius, n2, -1)

                eb1 = np.abs(self.bound_handler.be[n1][-1])
                eb2 = np.abs(self.bound_handler.be[n2][-1])
                integration_end = self.q_value-eb1-eb2
                if integration_end < 0:
                    self.spectrum_integrals[n1][-1][n2][-1] = 0.
                    continue
                tmp_result = integrate.quad(
                    func=lambda x: neutrino_integrand_standard(
                        x, -(ph.electron_mass-eb1), -(ph.electron_mass-eb2), self.q_value-2*ph.electron_mass, self.enei-ph.electron_mass),
                    a=0.,
                    b=self.q_value-eb1-eb2
                )

                if not isinstance(tmp_result, tuple):
                    raise ValueError("Spectrum integration did not succeed")

                res = tmp_result[0]*prob1*prob2

                self.spectrum_integrals[n1][-1][n2][-1] = res

    def compute_psf(self):
        psfs = {}
        for n1 in self.spectrum_integrals:
            psfs[n1] = {-1: {}}
            for n2 in self.spectrum_integrals[n1][-1]:
                psf_mev = self.spectrum_integrals[n1][-1][n2][-1] * \
                    self.constant_in_front
                psf_years = psf_mev/(ph.hbar*np.log(2.))/(ph.year**(-1))

                psfs[n1][-1][n2] = {-1: psf_years}

        return psfs

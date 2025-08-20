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
    if transition == ph.TransitionTypes.ZEROPLUS_TO_TWOPLUS:
        return neutrino_integrand_closure_standard_02(enu_1, e_electron_1, e_electron_2, enu_2, enei)
    else:
        return neutrino_integrand_closure_standard_00(enu_1, e_electron_1, e_electron_2, enu_2, enei)


class TwoECSpectrum(SpectrumBase):
    def __init__(self, total_ke: float, ei_ef: float, bound_handler: BoundHandler, nuclear_radius: float) -> None:
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
        pass

    def integrate_spectrum(self):
        raise NotImplementedError()

    def compute_psf(self):
        for n1 in self.spectrum_integrals:
            self.psfs[n1] = {-1: {}}
            for n2 in self.spectrum_integrals[n1][-1]:
                psf_mev = self.spectrum_integrals[n1][-1][n2][-1] * \
                    self.constant_in_front
                psf_years = psf_mev/(ph.hbar*np.log(2.))/(ph.year**(-1))

                self.psfs[n1][-1][n2] = {-1: psf_years}

    def compute_2D_spectrum(self, sp_type: ph.SpectrumTypes):
        raise NotImplementedError()


class TwoECSpectrumClosure(TwoECSpectrum):
    def __init__(self, total_ke: float, ei_ef: float, bound_handler: BoundHandler, nuclear_radius: float, enei: float, transition_type: ph.TransitionTypes) -> None:
        super().__init__(total_ke, ei_ef, bound_handler, nuclear_radius)
        self.enei = enei
        self.atilde = self.enei+0.5*(self.total_ke-2*ph.electron_mass)
        self.constant_in_front = 2*(self.atilde**2.0)*((ph.fermi_coupling_constant*ph.v_ud)**4) /\
            (48.*(np.pi**3.0)) * (ph.electron_mass**4.0)
        self.transition_type = transition_type

    def compute_spectrum(self, sp_type: ph.SpectrumTypes = ph.SpectrumTypes.SINGLESPECTRUM):
        print(f"total_ke = {self.total_ke}")
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
    def __init__(self, total_ke: float, ei_ef: float, bound_handler: BoundHandler, nuclear_radius: float, order: ph.TaylorOrders, transition_type: ph.TransitionTypes) -> None:
        super().__init__(total_ke, ei_ef, bound_handler, nuclear_radius)
        self.order = order
        self.transition_type = transition_type

        self.constant_in_front = ((ph.fermi_coupling_constant*ph.v_ud)**4) /\
            (2.*(np.pi**3.0)) * (ph.electron_mass**4.0)

    def compute_spectrum(self, sp_type: ph.SpectrumTypes = ph.SpectrumTypes.SINGLESPECTRUM):
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

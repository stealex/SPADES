from abc import ABC, abstractmethod

import numpy as np
from scipy import interpolate
from scipy.interpolate import CubicSpline
from spades.spectra.base import BetaSpectrumBase
from spades.fermi_functions import FermiFunctions
from spades.wavefunctions import BoundHandler
from spades import ph

from abc import abstractmethod
from functools import lru_cache
from typing import Callable
import numpy as np
from numpy import ndarray
from tqdm import tqdm
from scipy import integrate
from spades import fermi_functions, ph
from spades.fermi_functions import FermiFunctions
from spades.wavefunctions import BoundHandler
from spades.spectra.closure_helpers import neutrino_integrand_closure_standard_00, neutrino_integrand_closure_standard_02


def neutrino_integrand_closure(enu_1, e_positron, e_electron, enu_2, enei, transition: ph.TransitionTypes):
    if transition == ph.TransitionTypes.ZEROPLUS_TO_TWOPLUS:
        return neutrino_integrand_closure_standard_02(enu_1, e_positron, e_electron, enu_2, enei)
    else:
        return neutrino_integrand_closure_standard_00(enu_1, e_positron, e_electron, enu_2, enei)


class ECBetaSpectrumBase(BetaSpectrumBase):
    """Base class for the computation of 2nu EC betaPlus spectra.
    """

    def __init__(self, total_ke: float, ei_ef: float, fermi_functions: FermiFunctions, bound_handler: BoundHandler, nuclear_radius: float, **kwargs) -> None:
        """Creates also the energy grids (different for each shell).

        Args:
            total_ke (float): total kinetic energy available in the process
            ei_ef (float): energy difference between initial and final NUCLEAR levels
            fermi_functions(FermiFunctions): Fermi functions to be used in the computation
            bound_handler (BoundHandler): Handler containing bound wavefunctions
            nuclear_radius (float): radius of nucleus (fm).
        """
        super().__init__(total_ke, ei_ef, fermi_functions)
        self.bound_handler = bound_handler
        self.g_func = {}
        self.f_func = {}
        self.nuclear_radius = nuclear_radius

        self.energy_grid_type = kwargs.get("energy_grid_type", "lin")
        self.min_ke = kwargs.get("min_ke", 1E-4)
        self.n_ke_points = kwargs.get("n_ke_points", 100)
        self.energy_points = {}

        for n in self.bound_handler.p_grid:
            self.spectrum_values[n] = {}
            self.energy_points[n] = {}
            for k in self.bound_handler.p_grid[n]:
                if k != -1:
                    continue
                # build energy points
                eb = - np.abs(self.bound_handler.be[n][k])
                if self.energy_grid_type == "lin":
                    self.energy_points[n][k] = np.linspace(
                        self.min_ke,
                        self.total_ke+eb-self.min_ke,
                        self.n_ke_points
                    )
                elif self.energy_grid_type == "log":
                    self.energy_points[n][k] = np.logspace(
                        np.log10(self.min_ke),
                        np.log10(self.total_ke+eb-self.min_ke),
                        self.n_ke_points
                    )
                else:
                    raise NotImplementedError(
                        f"Energy grid type {self.energy_grid_type} not implemented")

                self.spectrum_values[n][k] = np.zeros_like(
                    self.energy_points[n][k])

    @abstractmethod
    def compute_spectrum(self, sp_type: ph.SpectrumTypes = ph.SpectrumTypes.SINGLESPECTRUM):
        pass

    @abstractmethod
    def integrate_spectrum(self):
        pass

    @abstractmethod
    def compute_psf(self):
        pass

    def compute_2D_spectrum(self, sp_type: ph.SpectrumTypes):
        raise NotImplementedError()


class ClosureSpectrum(ECBetaSpectrumBase):
    """Base Closure spectrum for 2nu ec betaPlus spectra.
    """

    def __init__(self, total_ke: float, ei_ef: float, fermi_functions: FermiFunctions, bound_handler: BoundHandler, nuclear_radius: float, enei: float, **kwargs) -> None:
        """
        Args:
            total_ke (float): total kinetic energy available in the process
            ei_ef (float): energy difference between initial and final NUCLEAR levels
            fermi_functions(FermiFunctions): Fermi functions to be used in the computation
            bound_handler (BoundHandler): Handler containing bound wavefunctions
            nuclear_radius (float): radius of nucleus (fm).
            enei (float): <E_N> - E_I
        """
        super().__init__(total_ke, ei_ef, fermi_functions,
                         bound_handler, nuclear_radius, **kwargs)
        self.enei = enei
        self.atilde = self.enei + 0.5*self.ei_ef
        self.constant_in_front = 1.0

    @abstractmethod
    def compute_spectrum(self, sp_type: int):
        pass

    def integrate_spectrum(self):
        """Integrates each positron spectrum resulting from a given (n,k) capture.
        """
        for n in self.spectrum_values:
            self.spectrum_integrals[n] = {}
            for k in self.spectrum_values[n]:
                if k != -1:
                    continue
                print(f"Doing for {n} {k}")
                interp_func = interpolate.CubicSpline(
                    self.energy_points[n][k], self.spectrum_values[n][k]
                )

                result = integrate.quad(
                    interp_func,
                    self.energy_points[n][k][0],
                    self.energy_points[n][k][-1]
                )
                if isinstance(result, tuple):
                    self.spectrum_integrals[n][k] = result[0]
                else:
                    raise ValueError("Spectrum integration did not succeed")

                # normalize the resulting spectrum
                self.spectrum_values[n][k] = self.spectrum_values[n][k]/result[0]

    def compute_psf(self):
        """Computes the PSF for each (n,k) capture from integrals
        """
        for n in self.spectrum_integrals:
            self.psfs[n] = {}
            for k in self.spectrum_integrals[n]:
                psf_mev = self.spectrum_integrals[n][k]*self.constant_in_front
                psf_years = psf_mev/(ph.hbar*np.log(2.))/(ph.year**(-1))
                self.psfs[n][k] = psf_years


class ClosureSpectrum2nu(ClosureSpectrum):
    """Concrete class for the computation of closure spectrum in 2nu EC betaPlus decay.
    """

    def __init__(self, total_ke: float, ei_ef: float, fermi_functions: FermiFunctions, bound_handler: BoundHandler, nuclear_radius: float, enei: float,
                 transition_type: ph.TransitionTypes, **kwargs) -> None:
        super().__init__(total_ke, ei_ef, fermi_functions,
                         bound_handler, nuclear_radius, enei, **kwargs)
        self.constant_in_front = 2*(self.atilde**2.0)*((ph.fermi_coupling_constant*ph.v_ud)**4) /\
            (48.*(np.pi**5.0)) * ph.electron_mass
        self.transition_type = transition_type

    def compute_spectrum(self, sp_type: ph.SpectrumTypes = ph.SpectrumTypes.SINGLESPECTRUM):
        """For each (n,k) shell it computes the positron spectrum by integrating on the neutrino energy.

        Args:
            sp_type (ph.SpectrumTypes, optional): not used. Defaults to ph.SpectrumTypes.SINGLESPECTRUM.

        Raises:
            ValueError: _description_
        """
        # we have one spectrum for each shell
        for i_n in tqdm(range(len(self.bound_handler.config.n_values)),
                        desc="\t"*2 +
                        f"- {ph.SPECTRUM_TYPES_NICE[ph.SpectrumTypes.SINGLESPECTRUM]}",
                        ncols=100):
            n = self.bound_handler.config.n_values[i_n]
            spectrum_current = np.zeros_like(self.energy_points[n][-1])
            eb = np.abs(self.bound_handler.be[n][-1])
            prob = self.bound_handler.probability_in_sphere(
                self.nuclear_radius, n, -1)

            for i_e in range(len(self.energy_points[n][-1])-1):
                ep = self.energy_points[n][-1][i_e]
                result = integrate.quad(
                    func=lambda x: neutrino_integrand_closure(
                        x,
                        ep+ph.electron_mass,
                        eb,
                        self.total_ke-ep-x-eb, self.enei,
                        self.transition_type),
                    a=0.,
                    b=self.total_ke - eb - ep
                )
                if isinstance(result, tuple):
                    spectrum_current[i_e] = result[0]
                else:
                    raise ValueError("Spectrum integration did not succeed")

                ff_ec_beta = self.fermi_functions.ff_ecbeta_eval(ep)
                spectrum_current[i_e] = spectrum_current[i_e] *\
                    ff_ec_beta*prob*np.sqrt(ep*(ep+2.0*ph.electron_mass)) *\
                    (ep+ph.electron_mass)

            spectrum_current[-1] = 0.

            self.spectrum_values[n][-1] = spectrum_current
        # print(f"spectrum_valeus = {self.spectrum_values}")


class ECBetaSpectrumBase0nu(BetaSpectrumBase):
    """Base class for the computation of 2nu EC betaPlus spectra.
    """

    def __init__(self, total_ke: float, ei_ef: float, fermi_functions: FermiFunctions, bound_handler: BoundHandler, nuclear_radius: float, **kwargs) -> None:
        """
        Args:
            total_ke (float): total kinetic energy available in the process
            ei_ef (float): energy difference between initial and final NUCLEAR levels
            fermi_functions(FermiFunctions): Fermi functions to be used in the computation
            bound_handler (BoundHandler): Handler containing bound wavefunctions
            nuclear_radius (float): radius of nucleus (fm).
        """
        super().__init__(total_ke, ei_ef, fermi_functions)
        self.bound_handler = bound_handler
        self.g_func = {}
        self.f_func = {}
        self.nuclear_radius = nuclear_radius
        print(kwargs)

        for n in self.bound_handler.p_grid:
            self.spectrum_integrals[n] = {}
            for k in self.bound_handler.p_grid[n]:
                if k != -1:
                    continue
                self.spectrum_integrals[n][k] = 0.

    @abstractmethod
    def compute_spectrum(self, sp_type: ph.SpectrumTypes = ph.SpectrumTypes.SINGLESPECTRUM):
        pass

    def integrate_spectrum(self):
        raise NotImplementedError()

    @abstractmethod
    def compute_psf(self):
        pass

    def compute_2D_spectrum(self, sp_type: ph.SpectrumTypes):
        raise NotImplementedError()


class ClosureSpectrum0nu(ECBetaSpectrumBase0nu):
    """Base Closure spectrum for 0nu ec betaPlus spectra.
    """

    def __init__(self, total_ke: float, ei_ef: float, fermi_functions: FermiFunctions, bound_handler: BoundHandler, nuclear_radius: float, enei: float, **kwargs) -> None:
        """
        Args:
            total_ke (float): total kinetic energy available in the process
            ei_ef (float): energy difference between initial and final NUCLEAR levels
            fermi_functions(FermiFunctions): Fermi functions to be used in the computation
            bound_handler (BoundHandler): Handler containing bound wavefunctions
            nuclear_radius (float): radius of nucleus (fm).
            enei (float): <E_N> - E_I
        """
        super().__init__(total_ke, ei_ef, fermi_functions,
                         bound_handler, nuclear_radius, **kwargs)
        self.enei = enei
        self.atilde = self.enei + 0.5*self.ei_ef
        self.constant_in_front = 1.0

    @abstractmethod
    def compute_spectrum(self, sp_type: int):
        pass

    def compute_psf(self):
        """Computes the PSF for each (n,k) capture from integrals
        """
        for n in self.spectrum_integrals:
            self.psfs[n] = {}
            for k in self.spectrum_integrals[n]:
                psf_mev = self.spectrum_integrals[n][k]*self.constant_in_front
                psf_years = psf_mev/(ph.hbar*np.log(2.))/(ph.year**(-1))
                self.psfs[n][k] = psf_years


class ClosureSpectrum0nu_LNE(ClosureSpectrum0nu):
    """Concrete class for the computation of closure spectrum in 2nu EC betaPlus decay.
    """

    def __init__(self, total_ke: float, ei_ef: float, fermi_functions: FermiFunctions, bound_handler: BoundHandler, nuclear_radius: float, enei: float, **kwargs) -> None:
        super().__init__(total_ke, ei_ef, fermi_functions,
                         bound_handler, nuclear_radius, enei, **kwargs)
        self.constant_in_front = ((ph.fermi_coupling_constant*ph.v_ud)**4) / \
            (8*(np.pi**3)*(nuclear_radius**2.0)) * \
            (ph.electron_mass**5.0)*(ph.hc**2.0)

    def compute_spectrum(self, sp_type: ph.SpectrumTypes = ph.SpectrumTypes.SINGLESPECTRUM):
        """For each (n,k) shell it computes the spectrum integral 

        Args:
            sp_type (ph.SpectrumTypes, optional): not used. Defaults to ph.SpectrumTypes.SINGLESPECTRUM.
        """
        for i_n in tqdm(range(len(self.bound_handler.config.n_values)),
                        desc="\t"*2 +
                        f"- {ph.SPECTRUM_TYPES_NICE[ph.SpectrumTypes.SINGLESPECTRUM]}",
                        ncols=100):
            n = self.bound_handler.config.n_values[i_n]
            eb = np.abs(self.bound_handler.be[n][-1])
            ep = self.total_ke+eb
            pp = np.sqrt(ep*(ep+2.0*ph.electron_mass))
            ff_ec_beta = self.fermi_functions.ff_ecbeta_eval(ep)
            prob = self.bound_handler.probability_in_sphere(
                self.nuclear_radius, n, -1)
            self.spectrum_integrals[n][-1] = ff_ec_beta*prob*pp*ep

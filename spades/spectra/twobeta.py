from abc import abstractmethod

import jedi
from spades.fermi_functions import FermiFunctions
from spades.spectra.base import BetaSpectrumBase
from abc import abstractmethod
from functools import lru_cache
from typing import Callable
from scipy import integrate, interpolate
from tqdm import tqdm
from spades.fermi_functions import FermiFunctions
from spades import ph
import numpy as np
from numba import njit
from spades.math_stuff import kn, ln, neutrino_integrand_closure_standard_00


@njit
def neutrino_integrand_closure_standard_02(enu1: float, e1: float, e2: float, enu2: float, enei: float):
    k = kn(e1, e2, enu1, enu2, enei)
    l = ln(e1, e2, enu1, enu2, enei)
    return 3.0*(k*k-l*l)*(enu1**2.0)*(enu2**2.0)


@njit
def neutrino_integrand_closure_angular_00(enu1: float, e1: float, e2: float, enu2: float, enei: float):
    k = kn(e1, e2, enu1, enu2, enei)
    l = ln(e1, e2, enu1, enu2, enei)
    return 1./3.*(2*k*k + 2*l*l + 5*k*l)*(enu1**2.0)*(enu2**2.0)


@lru_cache(maxsize=None)
def standard_electron_integrant_2nubb(e1, e2, fermi_func: Callable):
    return fermi_func(e1)*fermi_func(e2) * \
        (e1+ph.electron_mass) * (e2+ph.electron_mass) * \
        np.sqrt(e1*(e1+2.0*ph.electron_mass)) * \
        np.sqrt(e2*(e2+2.0*ph.electron_mass))


def spectrum_integrant_2nubb(enu, e2, e1, total_ke, sp_type: int, emin, enei, full_func: Callable,  transition: int):
    """Spectrum integrat for 2nu beta beta decay.

    Args:
        enu (_type_): energy of (anti-) neutrino
        e2 (_type_): kinetic energy of second massive lepton
        e1 (_type_): kinetic energy of first massive letpon
        total_ke (_type_): q-value of the decay
        sp_type (int): spectrum type
        emin (_type_): minimum kinetic energy used in integration
        enei (_type_): E_{N} - E_{I}
        full_func (Callable): full Fermi function
        transition (int): transition type

    Raises:
        NotImplementedError: _description_
        NotImplementedError: _description_
        NotImplementedError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    # compute total energies
    et1 = e1+ph.electron_mass
    et2 = e2+ph.electron_mass
    enu2 = total_ke - e1 - e2 - enu
    if sp_type == ph.SINGLESPECTRUM:
        if (transition == ph.ZEROPLUS_TO_ZEROPLUS):
            return standard_electron_integrant_2nubb(e1, e2, full_func) *\
                neutrino_integrand_closure_standard_00(
                    enu, et1, et2, enu2, enei)
        elif (transition == ph.ZEROPLUS_TO_TWOPLUS):
            return standard_electron_integrant_2nubb(e1, e2, full_func) *\
                neutrino_integrand_closure_standard_02(
                    enu, et1, et2, enu2, enei)
        else:
            raise NotImplementedError()
    elif sp_type == ph.SUMMEDSPECTRUM:
        t = e1
        v = e2
        ee2 = t*v/total_ke
        ee1 = t - ee2

        et1 = ee1+ph.electron_mass
        et2 = ee2+ph.electron_mass
        enu2 = total_ke + 2*ph.electron_mass - et1 - et2 - enu

        if (ee1 < emin) or (ee2 < emin):
            return 0.
        if transition == ph.ZEROPLUS_TO_ZEROPLUS:
            ret_val = t/total_ke*standard_electron_integrant_2nubb(ee1, ee2, full_func) *\
                neutrino_integrand_closure_standard_00(
                    enu, et1, et2, enu2, enei)
        elif transition == ph.ZEROPLUS_TO_TWOPLUS:
            ret_val = t/total_ke*standard_electron_integrant_2nubb(ee1, ee2, full_func) *\
                neutrino_integrand_closure_standard_02(
                    enu, et1, et2, enu2, enei)
        else:
            raise NotImplementedError()
        return ret_val
    elif sp_type == ph.ANGULARSPECTRUM:
        if transition == ph.ZEROPLUS_TO_ZEROPLUS:
            return -1.0*standard_electron_integrant_2nubb(e1, e2, full_func) *\
                neutrino_integrand_closure_angular_00(
                    enu, et1, et2, enu2, enei)
        else:
            raise NotImplementedError()
    else:
        raise ValueError("Unknown spectrum type")


def range_enu(e2, e1, total_ke, sp_type, emin, enei, full_func, transition):
    if sp_type == ph.SINGLESPECTRUM:
        return [0., total_ke-e1-e2]
    elif sp_type == ph.SUMMEDSPECTRUM:
        t = e1
        v = e2
        return [0., total_ke-t]
    elif sp_type == ph.ANGULARSPECTRUM:
        return [0., total_ke-e1-e2]


def range_e2(e1, total_ke, sp_type, emin, enei,  full_func, transition):
    if sp_type == ph.SINGLESPECTRUM:
        return [emin, total_ke-e1]
    elif sp_type == ph.SUMMEDSPECTRUM:
        return [emin, total_ke-emin]
    elif sp_type == ph.ANGULARSPECTRUM:
        return [emin, total_ke-e1]


class TwoBetaSpectrumBase(BetaSpectrumBase):
    def __init__(self, total_ke: float, ei_ef: float, fermi_functions: FermiFunctions, **kwargs) -> None:
        super().__init__(total_ke, ei_ef, fermi_functions)
        # create energy points structure
        self.energy_grid_type = kwargs.get("energy_grid_type", "lin")
        self.min_ke = kwargs.get("min_ke", 1E-4)
        self.n_ke_points = kwargs.get("n_ke_points", 100)
        if (self.energy_grid_type == "lin"):
            self.energy_points = np.linspace(self.min_ke,
                                             self.total_ke-self.min_ke,
                                             self.n_ke_points)
        elif (self.energy_grid_type == "log"):
            self.energy_points = np.logspace(
                np.log10(self.min_ke),
                np.log10(self.total_ke-self.min_ke),
                self.n_ke_points
            )
        else:
            raise NotImplementedError()

    @abstractmethod
    def compute_spectrum(self, sp_type: int):
        pass

    @abstractmethod
    def compute_2D_spectrum(self, sp_type: int):
        pass

    @abstractmethod
    def compute_psf(self):
        pass

    @abstractmethod
    def integrate_spectrum(self):
        pass


class ClosureSpectrumBase(TwoBetaSpectrumBase):
    def __init__(self, total_ke: float, ei_ef: float, enei: float, fermi_functions: FermiFunctions, **kwargs) -> None:
        super().__init__(total_ke, ei_ef, fermi_functions, **kwargs)
        self.enei = enei
        self.atilde = self.enei + 0.5*ei_ef
        self.constant_in_front = 1.0

    @abstractmethod
    def compute_spectrum(self, sp_type: int):
        pass

    @abstractmethod
    def compute_2D_spectrum(self, sp_type: int):
        pass

    def integrate_spectrum(self):
        self.spectrum_integrals = {}
        for key in self.spectrum_values:
            interp_func = interpolate.CubicSpline(
                self.energy_points, self.spectrum_values[key]
            )

            result = integrate.qmc_quad(
                interp_func,
                self.energy_points[0],
                self.energy_points[-1]
            )
            if isinstance(result, tuple):
                self.spectrum_integrals[key] = result[0]
            else:
                raise ValueError("Spectrum integration did not succeed")

            self.spectrum_values[key] = self.spectrum_values[key]/result[0]
            if key in self.spectrum_2D_values:
                self.spectrum_2D_values[key] = self.spectrum_2D_values[key]/result[0]

    def compute_psf(self):
        for key in self.spectrum_values:
            psf_mev = self.spectrum_integrals[key]*self.constant_in_front
            psf_years = psf_mev/(ph.hbar*np.log(2.))/(ph.year**(-1))
            self.psfs[key] = psf_years


class ClosureSpectrum2nu(ClosureSpectrumBase):
    def __init__(self, total_ke: float, ei_ef: float, enei: float, fermi_functions: FermiFunctions, eta_total: Callable | None,
                 transition: int, **kwargs) -> None:
        super().__init__(total_ke, ei_ef, enei, fermi_functions, **kwargs)
        self.transition = transition
        self.constant_in_front = ((self.atilde/ph.electron_mass)**2.0)*((ph.fermi_coupling_constant*ph.v_ud)**4) /\
            (96*(np.pi**7))
        if (eta_total is None):
            self.eta_total = lambda x: 1.0
        else:
            self.eta_total = eta_total

        self.e1_grid_2D = kwargs.get("e1_grid_2D", None)
        self.e2_grid_2D = kwargs.get("e2_grid_2D", None)

    # @lru_cache(maxsize=None)
    def full_func(self, x, sp_type):
        if sp_type == ph.ANGULARSPECTRUM:
            return self.fermi_functions.ff1_eval(x)*self.eta_total(x)
        else:
            return self.fermi_functions.ff0_eval(x)*self.eta_total(x)

    def compute_spectrum(self, sp_type: int):
        self.spectrum_values[sp_type] = np.zeros_like(self.energy_points)
        for i_e in tqdm(range(len(self.energy_points)-1),
                        desc="\t"*2+f"- {ph.SPECTRUM_TYPES_NICE[sp_type]}",
                        ncols=100):
            e1 = self.energy_points[i_e]
            result = integrate.nquad(
                spectrum_integrant_2nubb,
                ranges=[range_enu, range_e2],
                args=(e1,
                      self.total_ke,
                      sp_type,
                      self.energy_points[0],
                      self.enei,
                      lambda x: self.full_func(x, sp_type),
                      self.transition
                      ),
                opts={"epsabs": 1E-13}
            )
            if isinstance(result, tuple):
                self.spectrum_values[sp_type][i_e] = result[0]
            else:
                raise ValueError("Spectrum integration did not succeed")
        self.spectrum_values[sp_type][-1] = 0.

    def compute_2D_spectrum(self, sp_type: int):
        self.spectrum_2D_values[sp_type] = np.zeros_like(self.e1_grid_2D)

        for ie in tqdm(range(len(self.e1_grid_2D)),
                       desc="\t"*2 +
                       f"- 2D {ph.SPECTRUM_TYPES_NICE[sp_type]}",
                       ncols=100):
            for je in range(len(self.e2_grid_2D)):
                e1 = self.e1_grid_2D[ie, je]
                e2 = self.e2_grid_2D[ie, je]

                if (e1+e2 <= self.total_ke):
                    result = integrate.quad(
                        func=spectrum_integrant_2nubb,
                        a=0.,
                        b=self.total_ke-e1-e2,
                        args=(e2, e1, self.total_ke, sp_type, self.min_ke, self.enei,
                              lambda x: self.full_func(x, sp_type), self.transition)
                    )
                    if isinstance(result, tuple):
                        self.spectrum_2D_values[sp_type][ie][je] = result[0]
                    else:
                        raise ValueError(
                            "Spectrum integration did not succeed")
                else:
                    self.spectrum_2D_values[sp_type][ie, je] = np.nan


@lru_cache(maxsize=None)
def standard_electron_integrant_0nubb(e1, total_ke, fermi_func: Callable):
    e2 = total_ke - e1
    return fermi_func(e1)*fermi_func(e2) * \
        (e1+ph.electron_mass) * (e2+ph.electron_mass) * \
        np.sqrt(e1*(e1+2.0*ph.electron_mass)) * \
        np.sqrt(e2*(e2+2.0*ph.electron_mass))


class ClosureSpectrum0nu_LNE(ClosureSpectrumBase):
    def __init__(self, total_ke: float, ei_ef: float, nuclear_radius: float, fermi_functions: FermiFunctions, eta_total: Callable | None, **kwargs) -> None:
        super().__init__(total_ke, ei_ef, 0., fermi_functions, **kwargs)
        self.constant_in_front = ((ph.fermi_coupling_constant*ph.v_ud)**4) / \
            (32.*(np.pi**5)*(nuclear_radius**2.)) * \
            (ph.electron_mass**2.0)*(ph.hc**2.0)
        if (eta_total is None):
            self.eta_total = lambda x: 1.0
        else:
            self.eta_total = eta_total

    @lru_cache(maxsize=None)
    def full_func(self, x, sp_type):
        if sp_type == ph.ANGULARSPECTRUM:
            return self.fermi_functions.ff1_eval(x)*self.eta_total(x)
        else:
            return self.fermi_functions.ff0_eval(x)*self.eta_total(x)

    def compute_spectrum(self, sp_type: int):
        self.spectrum_values[sp_type] = np.zeros_like(self.energy_points)
        for i_e in tqdm(range(len(self.energy_points)-1),
                        desc="\t"*2+f"- {ph.SPECTRUM_TYPES_NICE[sp_type]}",
                        ncols=100):
            e1 = self.energy_points[i_e]
            self.spectrum_values[sp_type][i_e] = standard_electron_integrant_0nubb(
                e1, self.total_ke, lambda x: self.full_func(x, sp_type))

        self.spectrum_values[sp_type][-1] = 0.
        return self.spectrum_values

    def compute_2D_spectrum(self, sp_type: int, e1_grid: np.ndarray, e2_grid: np.ndarray):
        raise NotImplementedError

from abc import abstractmethod
from ast import Call
from functools import lru_cache
from typing import Callable
from numpy import ndarray
from scipy import integrate
from tqdm import tqdm
from spades.fermi_functions import FermiFunctions
from spades.spectra.base import SpectrumBase
from spades import ph
import numpy as np
from numba import njit


@njit
def kn(e1: float, enu: float, q_value: float, enei: float):
    return 1./(e1+enu+enei+ph.electron_mass) + 1./(enei+q_value-e1-enu+ph.electron_mass)


@njit
def ln(e2: float, enu: float, q_value: float, enei: float):
    return 1./(enei+q_value-e2-enu+ph.electron_mass) + 1./(e2+enu+enei+ph.electron_mass)


@njit
def neutrino_integrand_standard(enu: float, e1: float, e2: float, q_value: float, enei: float):
    k = kn(e1, enu, q_value, enei)
    l = ln(e2, enu, q_value, enei)
    return (k*k+l*l+k*l)*(enu**2.0)*((q_value-e1-e2-enu)**2.0)


@njit
def neutrino_integrand_angular(enu: float, e1: float, e2: float, q_value: float, enei: float):
    k = kn(e1, enu, q_value, enei)
    l = ln(e2, enu, q_value, enei)
    return 1./3.*(2*k*k + 2*l*l + 5*k*l)*(enu**2.0)*((q_value-e1-e2-enu)**2.0)


def standard_electron_integrant(e1, e2, fermi_func: Callable):
    return fermi_func(e1)*fermi_func(e2) * \
        (e1+ph.electron_mass) * (e2+ph.electron_mass) * \
        np.sqrt(e1*(e1+2.0*ph.electron_mass)) * \
        np.sqrt(e2*(e2+2.0*ph.electron_mass))


class ClosureSpectrumBase(SpectrumBase):
    def __init__(self, sp_type: int, q_value: float, energy_points: ndarray, enei: float) -> None:
        super().__init__(sp_type, q_value, energy_points)
        self.enei = enei
        self.atilde = self.enei + 0.5*(self.q_value+2.0*ph.electron_mass)
        self.constant_in_front = 1.0

    @abstractmethod
    def compute_spectrum(self):
        pass

    def compute_psf(self, spectrum_integral: float):
        psf_mev = spectrum_integral*self.constant_in_front
        psf_years = psf_mev/(ph.hbar*np.log(2.))/(ph.year**(-1))
        return psf_years


class ClosureSpectrum2nuBB(ClosureSpectrumBase):
    def __init__(self, sp_type: int, q_value: float, energy_points: ndarray, enei: float, fermi_function: Callable) -> None:
        super().__init__(sp_type, q_value, energy_points, enei)
        self.constant_in_front = ((self.atilde/ph.electron_mass)**2.0)*((ph.fermi_coupling_constant*ph.v_ud)**4) / \
            (96*(np.pi**7))
        self.fermi_function = fermi_function

    @abstractmethod
    def compute_spectrum(self):
        pass


class ClosureSpectrum2nuBBSingle(ClosureSpectrum2nuBB):
    def __init__(self, q_value: float, energy_points: ndarray, enei: float, fermi_function: Callable) -> None:
        super().__init__(ph.SINGLESPECTRUM, q_value, energy_points, enei, fermi_function)

    def compute_spectrum(self, eta_total: Callable | None = None):
        self.spectrum_values["0"] = np.zeros_like(self.energy_points)

        if not (eta_total is None):
            @lru_cache(maxsize=None)
            def full_func(e1): return self.fermi_function(
                e1)*(1.0+eta_total(e1))
        else:
            @lru_cache(maxsize=None)
            def full_func(e1): return self.fermi_function(e1)

        def integrant(enu, e2, e1, q_value, enei, emin):
            return standard_electron_integrant(e1, e2, full_func) * neutrino_integrand_standard(enu, e1, e2, q_value, enei)

        def range_enu(e2, e1, q_value, enei, emin):
            return [0., q_value-e1-e2]

        def range_e2(e1, q_value, enei, emin):
            return [emin, q_value-e1]

        for i_e in tqdm(range(len(self.energy_points)-1),
                        desc="\t"*2+f"- {ph.SPECTRUM_TYPES_NICE[self.type]}",
                        ncols=100):
            e1 = self.energy_points[i_e]
            result = integrate.nquad(
                integrant,
                ranges=[range_enu, range_e2],
                args=(e1, self.q_value, self.enei, self.energy_points[0])
            )
            if isinstance(result, tuple):
                self.spectrum_values["0"][i_e] = result[0]
            else:
                raise ValueError("Spectrum integration did not succeed")

        print(self.spectrum_values["0"])
        return self.spectrum_values


class ClosureSpectrum2nuBBSum(ClosureSpectrum2nuBB):
    def __init__(self, q_value: float, energy_points: ndarray, enei: float, fermi_function: Callable) -> None:
        super().__init__(ph.SUMMEDSPECTRUM, q_value, energy_points, enei, fermi_function)

    def compute_spectrum(self, eta_total: Callable | None = None):
        self.spectrum_values["0"] = np.zeros_like(self.energy_points)

        if not (eta_total is None):
            @lru_cache(maxsize=None)
            def full_func(e1): return self.fermi_function(
                e1)*(1.0+eta_total(e1))
        else:
            @lru_cache(maxsize=None)
            def full_func(e1): return self.fermi_function(e1)

        def integrant(enu, v, t, q_value, enei, emin):
            e2 = t*v/q_value
            e1 = t - e2
            if (e1 < emin) or (e2 < emin):
                return 0.
            ret_val = t/q_value*standard_electron_integrant(e1, e2, full_func) *\
                neutrino_integrand_standard(
                    enu, e1, e2, q_value, enei)
            return ret_val

        def range_enu(v, t, q_value, enei, emin):
            return [0., q_value-t]

        def range_e2(t, q_value, enei, emin):
            return [emin, q_value-emin]

        for i_e in tqdm(range(len(self.energy_points)-1),
                        desc="\t"*2+f"- {ph.SPECTRUM_TYPES_NICE[self.type]}",
                        ncols=100):
            e1 = self.energy_points[i_e]
            result = integrate.nquad(
                integrant,
                ranges=[range_enu, range_e2],
                args=(e1, self.q_value, self.enei, self.energy_points[0])
            )
            if isinstance(result, tuple):
                self.spectrum_values["0"][i_e] = result[0]
            else:
                raise ValueError("Spectrum integration did not succeed")

        print(self.spectrum_values["0"])
        return self.spectrum_values


class ClosureSpectrum2nuBBAngular(ClosureSpectrum2nuBB):
    def __init__(self, q_value: float, energy_points: ndarray, enei: float, fermi_function: Callable) -> None:
        super().__init__(ph.SUMMEDSPECTRUM, q_value, energy_points, enei, fermi_function)

    def compute_spectrum(self, eta_total: Callable | None = None):
        self.spectrum_values["0"] = np.zeros_like(self.energy_points)

        if not (eta_total is None):
            @lru_cache(maxsize=None)
            def full_func(e1): return self.fermi_function(
                e1)*(1.0+eta_total(e1))
        else:
            @lru_cache(maxsize=None)
            def full_func(e1): return self.fermi_function(e1)

        def integrant(enu, e2, e1, q_value, enei, emin):
            return -1.0*standard_electron_integrant(e1, e2, full_func) *\
                neutrino_integrand_angular(enu, e1, e2, q_value, enei)

        def range_enu(e2, e1, q_value, enei, emin):
            return [0., q_value-e1-e2]

        def range_e2(e1, q_value, enei, emin):
            return [emin, q_value-e1]

        for i_e in tqdm(range(len(self.energy_points)-1),
                        desc="\t"*2+f"- {ph.SPECTRUM_TYPES_NICE[self.type]}",
                        ncols=100):
            e1 = self.energy_points[i_e]
            result = integrate.nquad(
                integrant,
                ranges=[range_enu, range_e2],
                args=(e1, self.q_value, self.enei, self.energy_points[0])
            )
            if isinstance(result, tuple):
                self.spectrum_values["0"][i_e] = result[0]
            else:
                raise ValueError("Spectrum integration did not succeed")

        print(self.spectrum_values["0"])
        return self.spectrum_values


def create_closure_spectrum(sp_type: int, q_value: float, energy_points: ndarray, enei: float, fermi_functions: FermiFunctions):
    if sp_type == ph.SINGLESPECTRUM:
        return ClosureSpectrum2nuBBSingle(q_value, energy_points, enei, fermi_functions.ff0_eval)
    elif sp_type == ph.SUMMEDSPECTRUM:
        return ClosureSpectrum2nuBBSum(q_value, energy_points, enei, fermi_functions.ff0_eval)
    else:
        raise NotImplementedError

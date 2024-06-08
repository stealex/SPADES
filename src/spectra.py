import numpy as np
from numba import njit
from scipy import integrate
from . import ph
from typing import Callable
from scipy import interpolate
from tqdm import tqdm
from functools import lru_cache


class spectra_config:
    def __init__(self, method: str, wavefunction_evaluation: str, nuclear_radius: str | float, types: list[str],
                 energy_grid_type: str, fermi_functions: list[str], corrections: list[str] | None = None, q_value: float | str | None = None, min_ke: float | str | None = None, n_ke_points: int | str | None = None) -> None:
        self.method = ph.SPECTRUM_METHODS[method]
        self.wavefunction_evaluation = ph.WAVEFUNCTIONEVALUATION[wavefunction_evaluation]
        self.nuclear_radius = nuclear_radius
        self.energy_grid_type = energy_grid_type

        self.types = []
        for t in types:
            self.types.append(ph.SPECTRUM_TYPES[t])

        self.corrections = []
        if type(corrections) == list:
            for c in corrections:
                self.corrections.append(ph.CORRECTIONS[c])

        self.fermi_functions = []
        for f in fermi_functions:
            self.fermi_functions.append(ph.FERMIFUNCTIONS[f])

        self.q_value = q_value
        self.min_ke = min_ke
        self.n_ke_points = n_ke_points


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


class spectrum:
    def __init__(self, q_value: float, energy_points: np.ndarray) -> None:
        self.q_value = q_value
        self.energy_points = energy_points
        self.constant_in_front = 1.0

    def compute_spectrum(self, spectrum_type: int, fermi_func: Callable):
        pass

    def standard_electron_integrant(self, e1, e2, fermi_func: Callable):
        return fermi_func(e1)*fermi_func(e2) * \
            (e1+ph.electron_mass) * (e2+ph.electron_mass) * \
            np.sqrt(e1*(e1+2.0*ph.electron_mass)) * \
            np.sqrt(e2*(e2+2.0*ph.electron_mass))

    def integrate_spectrum(self, spectrum):
        psf_spl = interpolate.CubicSpline(
            self.energy_points, spectrum)
        result = integrate.quad(
            psf_spl, self.energy_points[0], self.energy_points[-1])

        if isinstance(result, tuple):
            psf = result[0]
        else:
            raise ValueError("PSF integration did not succeed")

        return psf

    def compute_psf(self, spectrum):
        pass


class closure_spectrum(spectrum):
    def __init__(self, q_value: float, energy_points: np.ndarray, enei: float) -> None:
        super().__init__(q_value, energy_points)
        self.enei = enei
        self.atilde = self.enei + 0.5*(self.q_value+2.0*ph.electron_mass)
        self.constant_in_front = ((self.atilde/ph.electron_mass)**2.0)*((ph.fermi_coupling_constant*ph.v_ud)**4) / \
            (96*(np.pi**7))

    def compute_spectrum(self, spectrum_type: int, fermi_func: Callable, eta_total: Callable | None = None):
        spectrum = np.zeros_like(self.energy_points)
        if not (eta_total is None):
            @ lru_cache(maxsize=None)
            def full_func(e1): return fermi_func(e1)*(1.0+eta_total(e1))
        else:
            @ lru_cache(maxsize=None)
            def full_func(e1): return fermi_func(e1)

        if spectrum_type == ph.SINGLESPECTRUM:
            def integrant_single(enu, e2, e1, q_value, enei, emin): return self.standard_electron_integrant(e1, e2, full_func) *\
                neutrino_integrand_standard(enu, e1, e2, q_value, enei)
            integrant = integrant_single

            def range_enu_single(e2, e1, q_value, enei, emin):
                return [0., q_value-e1-e2-emin]
            range_enu = range_enu_single

            def range_e2_single(e1, q_value, enei, emin):
                return [emin, q_value-e1-emin]
            range_e2 = range_e2_single

        elif spectrum_type == ph.ANGULARSPECTRUM:
            def integrant_angular(enu, e2, e1, q_value, enei, emin): return -1.0*self.standard_electron_integrant(e1, e2, full_func) *\
                neutrino_integrand_angular(enu, e1, e2, q_value, enei)
            integrant = integrant_angular

            def range_enu_angular(e2, e1, q_value, enei, emin):
                return [0., q_value-e1-e2-emin]
            range_enu = range_enu_angular

            def range_e2_angular(e1, q_value, enei, emin):
                return [emin, q_value-e1-emin]
            range_e2 = range_e2_angular

        elif spectrum_type == ph.SUMMEDSPECTRUM:
            def integrant_sum(enu, v, t, q_value, enei, emin):
                e2 = t*v/q_value
                e1 = t - e2
                if (e1 < emin) or (e2 < emin):
                    return 0.
                ret_val = t/q_value*self.standard_electron_integrant(e1, e2, full_func) *\
                    neutrino_integrand_standard(
                        enu, e1, e2, q_value, enei)
                return ret_val
            integrant = integrant_sum

            def range_enu_sum(v, t, q_value, enei, emin):
                return [0., q_value-t-emin]
            range_enu = range_enu_sum

            def range_e2_sum(t, q_value, enei, emin):
                return [emin, q_value-emin]
            range_e2 = range_e2_sum

        sp_type_nice = list(
            filter(lambda x: ph.SPECTRUM_TYPES[x] == spectrum_type, ph.SPECTRUM_TYPES))[0]

        for i_e in tqdm(range(len(self.energy_points)-1),
                        desc="\t"*2+f"- {sp_type_nice}",
                        ncols=100):
            e1 = self.energy_points[i_e]
            result = integrate.nquad(
                integrant,
                ranges=[range_enu, range_e2],
                args=(e1, self.q_value, self.enei, self.energy_points[0]),
            )
            if isinstance(result, tuple):
                spectrum[i_e] = result[0]
            else:
                raise ValueError("Spectrum integration did not succeed")

        spectrum[-1] = 0.
        return spectrum

    def compute_psf(self, spec):
        psf_mev = spec*self.constant_in_front
        psf_years = psf_mev/(ph.hbar*np.log(2.))/(ph.year**(-1))
        return psf_years


# Notation as in Simkovic et al Phys. Rev. C 97, 034315 (2018)
# @njit
# def eps_k(e1: float, e2: float, enu1: float, q_value: float):
#     enu2 = q_value-e1-e2-enu1
#     return 0.5*(e2+enu2-e1-enu1)


# @njit
# def eps_l(e1: float, e2: float, enu1: float, q_value: float):
#     enu2 = q_value-e1-e2-enu1
#     return 0.5*(e1+enu2-e2-enu1)


# @njit
# def a_0(e1: float, e2: float, enu1: float, q_value: float):
#     return 1.


# @njit
# def a_2(e1: float, e2: float, enu1: float, q_value: float):
#     return (eps_k(e1, e2, enu1, q_value)**2.0 + eps_l(e1, e2, enu1, q_value)**2.0)/((2.0*ph.electron_mass)**2.0)


# @njit
# def a_22(e1: float, e2: float, enu1: float, q_value: float):
#     return (eps_k(e1, e2, enu1, q_value)**2.0 * eps_l(e1, e2, enu1, q_value)**2.0)/((2.0*ph.electron_mass)**4.0)


# @njit
# def a_4(e1: float, e2: float, enu1: float, q_value: float):
#     return (eps_k(e1, e2, enu1, q_value)**4.0 + eps_l(e1, e2, enu1, q_value)**4.0)/((2.0*ph.electron_mass)**4.0)


# class taylor_expansion_spectrum(spectrum):
#     def __init__(self, q_value: float, energy_points: np.ndarray, fermi_func: Callable) -> None:
#         super().__init__(q_value, energy_points, fermi_func)
#         self.a2nu = {"0": a_0, "2": a_2, "22": a_22, "4": a_4}

#     def standard_electron_integrant(self, e1, e2):
#         return self.fermi_func(e1)*self.fermi_func(e2) * \
#             (e1+ph.electron_mass) * (e2+ph.electron_mass) * \
#             np.sqrt(e1*(e1+2.0*ph.electron_mass)) * \
#             np.sqrt(e2*(e2+2.0*ph.electron_mass))

#     def compute_spectrum(self, spectrum_type: int):
#         pass

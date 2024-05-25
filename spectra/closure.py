import numpy as np
from numba import njit
from scipy import integrate
from utils import ph
from typing import Callable
from scipy import interpolate


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

    def compute_spectrum(self, spectrum_type: int, fermi_func: Callable):
        spectrum = np.zeros_like(self.energy_points)

        if spectrum_type == ph.SINGLESPECTRUM:
            print("Single spectrum")

            def integrant(enu, e2, e1, q_value, enei, emin): return self.standard_electron_integrant(e1, e2, fermi_func) *\
                neutrino_integrand_standard(enu, e1, e2, q_value, enei)

            def range_enu(e2, e1, q_value, enei, emin):
                return [0., q_value-e1-e2]

            def range_e2(e1, q_value, enei, emin):
                return [emin, q_value-e1]

        elif spectrum_type == ph.ANGULARSPECTRUM:
            print("Angular spectrum")

            def integrant(enu, e2, e1, q_value, enei, emin): return -1.0*self.standard_electron_integrant(e1, e2, fermi_func) *\
                neutrino_integrand_angular(enu, e1, e2, q_value, enei)

            def range_enu(e2, e1, q_value, enei, emin):
                return [0., q_value-e1-e2]

            def range_e2(e1, q_value, enei, emin):
                return [emin, q_value-e1]

        elif spectrum_type == ph.SUMMEDSPECTRUM:
            print("Summed spectrum")

            def integrant(enu, v, t, q_value, enei, emin):
                e2 = t*v/q_value
                e1 = t - e2
                if (e1 < emin) or (e2 < emin):
                    return 0.
                ret_val = t/q_value*self.standard_electron_integrant(e1, e2, fermi_func) *\
                    neutrino_integrand_standard(
                        enu, e1, e2, q_value, enei)
                return ret_val

            def range_enu(v, t, q_value, enei, emin):
                return [0., q_value-t-emin]

            def range_e2(t, q_value, enei, emin):
                return [emin, q_value-emin]

        for i_e in range(len(self.energy_points)):
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

    def compute_psf(self, spectrum):
        spec = super().integrate_spectrum(spectrum)
        psf_mev = spec*self.constant_in_front
        psf_years = psf_mev/(ph.hbar*np.log(2.))/(ph.year**(-1))
        return psf_years


# Notation as in Simkovic et al Phys. Rev. C 97, 034315 (2018)
@njit
def eps_k(e1: float, e2: float, enu1: float, q_value: float):
    enu2 = q_value-e1-e2-enu1
    return 0.5*(e2+enu2-e1-enu1)


@njit
def eps_l(e1: float, e2: float, enu1: float, q_value: float):
    enu2 = q_value-e1-e2-enu1
    return 0.5*(e1+enu2-e2-enu1)


@njit
def a_0(e1: float, e2: float, enu1: float, q_value: float):
    return 1.


@njit
def a_2(e1: float, e2: float, enu1: float, q_value: float):
    return (eps_k(e1, e2, enu1, q_value)**2.0 + eps_l(e1, e2, enu1, q_value)**2.0)/((2.0*ph.electron_mass)**2.0)


@njit
def a_22(e1: float, e2: float, enu1: float, q_value: float):
    return (eps_k(e1, e2, enu1, q_value)**2.0 * eps_l(e1, e2, enu1, q_value)**2.0)/((2.0*ph.electron_mass)**4.0)


@njit
def a_4(e1: float, e2: float, enu1: float, q_value: float):
    return (eps_k(e1, e2, enu1, q_value)**4.0 + eps_l(e1, e2, enu1, q_value)**4.0)/((2.0*ph.electron_mass)**4.0)


class taylor_expansion_spectrum(spectrum):
    def __init__(self, q_value: float, energy_points: np.ndarray, fermi_func: Callable) -> None:
        super().__init__(q_value, energy_points, fermi_func)
        self.a2nu = {"0": a_0, "2": a_2, "22": a_22, "4": a_4}

    def standard_electron_integrant(self, e1, e2):
        return self.fermi_func(e1)*self.fermi_func(e2) * \
            (e1+ph.electron_mass) * (e2+ph.electron_mass) * \
            np.sqrt(e1*(e1+2.0*ph.electron_mass)) * \
            np.sqrt(e2*(e2+2.0*ph.electron_mass))

    def compute_spectrum(self, spectrum_type: int):
        pass

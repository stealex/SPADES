from ast import Raise, arg
import numpy as np
from numba import njit
from scipy import integrate
from . import ph
from typing import Callable
from scipy import interpolate
from tqdm import tqdm
from functools import lru_cache
import matplotlib.pyplot as plt


def normalize_spectra(values, integral):
    if (type(values) is np.ndarray) and (type(integral) is float):
        return values/integral
    if (type(values) is dict) and (type(integral) is dict):
        ret_val = {}
        for key in values:
            ret_val[key] = values[key]/integral[key]
        return ret_val
    else:
        raise NotImplementedError(f"Don't know what to do with type(spectra) = {
                                  type(values)} and type(integral) = {type(integral)}")


class Spectra2DConfig:
    def __init__(self, n_points_log: int, n_points_lin: int, e_max_log: float):
        self.n_points_log = n_points_log
        self.n_points_lin = n_points_lin
        self.e_max_log = e_max_log


class SpectraConfig:
    def __init__(self, method: str, wavefunction_evaluation: str, nuclear_radius: str | float, types: list[str],
                 energy_grid_type: str, fermi_functions: list[str], q_value: float, min_ke: float, n_ke_points: int, ke_step: float, corrections: list[str] | None = None,
                 orders: list[str] | None = None) -> None:
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

        self.orders = orders
        self.q_value = q_value
        self.min_ke = min_ke
        self.n_ke_points = n_ke_points
        self.ke_step = ke_step


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


class Spectrum:
    def __init__(self, q_value: float, energy_points: np.ndarray) -> None:
        self.q_value = q_value
        self.energy_points = energy_points
        self.constant_in_front = 1.0

    def compute_spectrum(self, spectrum_type: int, fermi_func: Callable):
        pass

    def standard_electron_integrant(self, e1, e2, fermi_func: Callable):
        pass

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


class Spectrum2nubb(Spectrum):
    def __init__(self, q_value: float, energy_points: np.ndarray) -> None:
        super().__init__(q_value, energy_points)

    def compute_spectrum(self, spectrum_type: int, fermi_func: Callable):
        pass

    def standard_electron_integrant(self, e1, e2, fermi_func: Callable):
        return fermi_func(e1)*fermi_func(e2) * \
            (e1+ph.electron_mass) * (e2+ph.electron_mass) * \
            np.sqrt(e1*(e1+2.0*ph.electron_mass)) * \
            np.sqrt(e2*(e2+2.0*ph.electron_mass))

    def compute_psf(self, spectrum):
        pass


class ClosureSpectrum2nubb(Spectrum2nubb):
    def __init__(self, q_value: float, energy_points: np.ndarray, enei: float) -> None:
        super().__init__(q_value, energy_points)
        self.enei = enei
        self.atilde = self.enei + 0.5*(self.q_value+2.0*ph.electron_mass)
        self.constant_in_front = ((self.atilde/ph.electron_mass)**2.0)*((ph.fermi_coupling_constant*ph.v_ud)**4) / \
            (96*(np.pi**7))

    def compute_spectrum(self, spectrum_type: int, fermi_func: Callable, eta_total: Callable | None = None):
        spectrum = np.zeros_like(self.energy_points)
        if not (eta_total is None):
            @lru_cache(maxsize=None)
            def full_func(e1): return fermi_func(e1)*(1.0+eta_total(e1))
        else:
            @lru_cache(maxsize=None)
            def full_func(e1): return fermi_func(e1)

        if spectrum_type == ph.SINGLESPECTRUM:
            def integrant_single(enu, e2, e1, q_value, enei, emin): return self.standard_electron_integrant(e1, e2, full_func) *\
                neutrino_integrand_standard(enu, e1, e2, q_value, enei)
            integrant = integrant_single

            def range_enu_single(e2, e1, q_value, enei, emin):
                return [0., q_value-e1-e2]
            range_enu = range_enu_single

            def range_e2_single(e1, q_value, enei, emin):
                return [emin, q_value-e1]
            range_e2 = range_e2_single

        elif spectrum_type == ph.ANGULARSPECTRUM:
            def integrant_angular(enu, e2, e1, q_value, enei, emin): return -1.0*self.standard_electron_integrant(e1, e2, full_func) *\
                neutrino_integrand_angular(enu, e1, e2, q_value, enei)
            integrant = integrant_angular

            def range_enu_angular(e2, e1, q_value, enei, emin):
                return [0., q_value-e1-e2]
            range_enu = range_enu_angular

            def range_e2_angular(e1, q_value, enei, emin):
                return [emin, q_value-e1]
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
                return [0., q_value-t]
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

    def compute_2d_spectra(self, spectrum_type: int, fermi_func: Callable, e1_final: np.ndarray, e2_final: np.ndarray, eta_total: Callable | None = None):
        # the change of variable is done on dG/de1de2
        # e1 = eta1 + emin
        # e2 = eta2*(q_value - eta1 - 2*emin) + emin
        if not (eta_total is None):
            @lru_cache(maxsize=None)
            def full_func(e1): return fermi_func(e1)*(1.0+eta_total(e1))
        else:
            @lru_cache(maxsize=None)
            def full_func(e1): return fermi_func(e1)

        if spectrum_type == ph.SINGLESPECTRUM:
            # def integrant_single(enu, eta2, eta1, q_value, enei, emin):
            #     e1 = eta1 + emin
            #     e2 = eta2*(q_value-eta1-2*emin) + emin

            #     return self.standard_electron_integrant(e1, e2, full_func) *\
            #         neutrino_integrand_standard(enu, e1, e2, q_value, enei)
            def integrant_single(enu, e2, e1, q_value, enei, emin):
                return self.standard_electron_integrant(e1, e2, full_func) *\
                    neutrino_integrand_standard(enu, e1, e2, q_value, enei)
            integrant = integrant_single
        if spectrum_type == ph.ANGULARSPECTRUM:
            # def integrant_angular(enu, eta2, eta1, q_value, enei, emin):
            #     e1 = eta1 + emin
            #     e2 = eta2*(q_value-eta1-2*emin) + emin

            #     return -1*self.standard_electron_integrant(e1, e2, full_func) *\
            #         neutrino_integrand_angular(enu, e1, e2, q_value, enei)
            def integrant_angular(enu, e2, e1, q_value, enei, emin):
                return -1*self.standard_electron_integrant(e1, e2, full_func) *\
                    neutrino_integrand_angular(enu, e1, e2, q_value, enei)
            integrant = integrant_angular

        # eta1_grid = self.energy_points-self.energy_points[0]
        # eta2_grid = np.linspace(0, 1, len(self.energy_points))
        # e1_grid = self.energy_points
        # e2_grid = []
        spectrum_2d = np.zeros((len(e1_final), len(e2_final)))
        # emin = self.energy_points[0]
        # spectrum_2d = np.zeros((len(eta1_grid), len(eta2_grid)))
        # spectrum_2d = np.zeros((len(e1_grid), len(e2_grid)))
        for ie in tqdm(range(len(e1_final)),
                       desc="\t"*2 +
                       f"- 2D {ph.SPECTRUM_TYPES_NICE[spectrum_type]}",
                       ncols=100):
            # e1 = eta1_grid[ie]+self.energy_points[0]
            for je in range(len(e2_final)):
                e1 = e1_final[ie, je]
                e2 = e2_final[ie, je]
                if (e1+e2 <= self.q_value):
                    result = integrate.quad(
                        integrant,
                        0., self.q_value-e1-e2,
                        args=(e2, e1, self.q_value, self.enei,
                              self.energy_points[0])
                    )
                    if isinstance(result, tuple):
                        spectrum_2d[ie, je] = result[0]
                else:
                    spectrum_2d[ie, je] = np.nan
            # e2_grid.append(np.arange(
            #     e1_grid[0], self.q_value-e1, self.energy_points[1]-self.energy_points[0]))
            # spectrum_tmp = np.zeros_like(e2_grid[-1])
            # for je in range(len(e2_grid[ie])):
            #     # e2 = eta2_grid[je]*(self.q_value-eta1_grid[ie] -
            #     #                     2*self.energy_points[0])+self.energy_points[0]
            #     e2 = e2_grid[ie][je]
            #     if self.q_value-e1-e2 < 0.:
            #         if abs(self.q_value-e1-e2) > 1e-10:
            #             raise ValueError("Something went wrong in the integration.\n"
            #                              "The energy grid is most probably wrong.")
            #         else:
            #             # spectrum_2d[ie, je] = 0.
            #             spectrum_tmp[je] = 0.
            #             continue

                # result = integrate.quad(
                #     integrant,
                #     0., self.q_value-e1-e2,
                #     # args=(eta2_grid[je], eta1_grid[ie], self.q_value,
                #     #       self.enei, self.energy_points[0]),
                #     args=(e2, e1, self.q_value, self.enei,
                #           self.energy_points[0])
                # )
                # if isinstance(result, tuple):
                #     # if result[0] >= 0. else 0.
                #     # spectrum_2d[ie, je] = result[0]
                #     spectrum_tmp[je] = result[0]
                # else:
                #     raise ValueError("Spectrum integration did not succeed")
            # spectrum_2d.append(spectrum_tmp)

        # spectrum_2d[-1, :] = 0.
        # return (eta1_grid, eta2_grid, spectrum_2d)

        # print(energy_grid_final)
        return spectrum_2d

    def compute_psf(self, spec):
        psf_mev = spec*self.constant_in_front
        psf_years = psf_mev/(ph.hbar*np.log(2.))/(ph.year**(-1))
        return psf_years


# Notation as in Nitescu  Universe 2021, 7(5), 147; https://doi.org/10.3390/universe7050147
@njit
def small_a(e1: float, e2: float, q_value: float):
    return q_value - e1 - e2


@njit
def small_b(e1: float, e2: float):
    return e1 - e2


@njit
def integral_order_0(e1: float, e2: float, q_value: float):
    return 1./30. * small_a(e1, e2, q_value)**5.0


@njit
def integral_order_2(e1: float, e2: float, q_value: float):
    a = small_a(e1, e2, q_value)
    b = small_b(e1, e2)
    return 1./(1680.*ph.electron_mass**2.0) *\
        a**5 * (a**2 + 7.*b**2)


@njit
def integral_order_22(e1: float, e2: float, q_value: float):
    a = small_a(e1, e2, q_value)
    b = small_b(e1, e2)
    return 1./(161280.*ph.electron_mass**4.0) *\
        a**5 * (a**4 - 6*(a**2) * (b**2) + 21.*b**4)


@njit
def integral_order_4(e1: float, e2: float, q_value: float):
    a = small_a(e1, e2, q_value)
    b = small_b(e1, e2)
    return 1./(80640.*ph.electron_mass**4) *\
        a**5 * (a**4 + 18.*(a**2)*(b**2) + 21*b**4)


@njit
def integral_order(e1: float, e2: float, q_value: float, order: str):
    if order == "0":
        return integral_order_0(e1, e2, q_value)
    elif order == "2":
        return integral_order_2(e1, e2, q_value)
    elif order == "22":
        return integral_order_22(e1, e2, q_value)
    elif order == "4":
        return integral_order_4(e1, e2, q_value)


class TaylorSpectrum2nubb(Spectrum2nubb):
    def __init__(self, q_value: float, energy_points: np.ndarray, orders: list[str] | None) -> None:
        super().__init__(q_value, energy_points)
        if orders is None:
            orders = ["0"]

        for ord in orders:
            if ord not in ["0", "2", "22", "4"]:
                raise ValueError(f"Unknown order {ord}")

        self.orders = orders
        self.constant_in_front = ((ph.fermi_coupling_constant*ph.v_ud)**4) / \
            (8.*(np.pi**7)*np.log(2.)*ph.electron_mass**2.0)

    def compute_spectrum(self, spectrum_type: int, fermi_func: Callable, eta_total: Callable | None = None):
        spectrum = np.zeros_like(self.energy_points)

        if not (eta_total is None):
            @lru_cache(maxsize=None)
            def full_func(e1): return fermi_func(e1)*(1.0+eta_total(e1))
        else:
            @lru_cache(maxsize=None)
            def full_func(e1): return fermi_func(e1)

        spectra = {}
        for ord in self.orders:
            spectra[ord] = np.zeros_like(self.energy_points)

            if spectrum_type == ph.SINGLESPECTRUM:
                def integrant_single(e2, e1, q_value, emin):
                    return self.standard_electron_integrant(e1, e2, full_func) *\
                        integral_order(e1, e2, q_value, ord)
                integrant = integrant_single

                def range_e2_single(e1, q_value, emin):
                    return [emin, q_value-e1]
                range_e2 = range_e2_single

            elif spectrum_type == ph.ANGULARSPECTRUM:
                def integrant_angular(e2, e1, q_value, emin):
                    return -1.0*self.standard_electron_integrant(e1, e2, full_func) *\
                        integral_order(e1, e2, q_value, ord)
                integrant = integrant_angular

                def range_e2_angular(e1, q_value, emin):
                    return [emin, q_value-e1]

                range_e2 = range_e2_angular

            elif spectrum_type == ph.SUMMEDSPECTRUM:
                def integrant_sum(enu, v, t, q_value, emin):
                    e2 = t*v/q_value
                    e1 = t - e2

                    if (e1 < emin) or (e2 < emin):
                        return 0.

                    ret_val = t/q_value * \
                        self.standard_electron_integrant(e1, e2, full_func) *\
                        integral_order(e1, e2, q_value, ord)
                    return ret_val
                integrant = integrant_sum

                def range_e2_sum(t, q_value, emin):
                    return [emin, q_value-emin]
                range_e2 = range_e2_sum

            sp_type_nice = list(
                filter(lambda x: ph.SPECTRUM_TYPES[x] == spectrum_type, ph.SPECTRUM_TYPES))[0]
            for i_e in tqdm(range(len(self.energy_points) - 1),
                            desc="\t"*2+f"- {sp_type_nice} {ord}",
                            ncols=100
                            ):
                e1 = self.energy_points[i_e]
                result = integrate.nquad(
                    integrant,
                    ranges=[range_e2],
                    args=(e1, self.q_value, self.energy_points[0])
                )

                if isinstance(result, tuple):
                    spectra[ord][i_e] = result[0]
                else:
                    raise ValueError("Spectrum integration did not succeed")

            spectra[ord][-1] = 0.
        return spectra

    def integrate_spectrum(self, spectrum):
        spectrum_integral = {}
        for ord in spectrum:
            spectrum_integral[ord] = super().integrate_spectrum(spectrum[ord])
        return spectrum_integral

    def compute_psf(self, spec):
        psf_years = {}
        print(ph.hbar)
        for key in spec:
            psf_mev = spec[key]*self.constant_in_front
            psf_years[key] = psf_mev/ph.hbar/(ph.year**(-1))
        return psf_years

    def compute_2d_spectra(self, spectrum_type: int, fermi_func: Callable, e1_final: np.ndarray,
                           e2_final: np.ndarray, eta_total: Callable | None = None):

        if not (eta_total is None):
            @lru_cache(maxsize=None)
            def full_func(e1): return fermi_func(e1)*(1.0+eta_total(e1))
        else:
            @lru_cache(maxsize=None)
            def full_func(e1): return fermi_func(e1)

        spectra_2d = {}
        fact = 1.
        if spectrum_type == ph.ANGULARSPECTRUM:
            fact = -1.0

        for ord in self.orders:
            spectra_2d[ord] = np.zeros((len(e1_final), len(e2_final)))

            for ie in tqdm(range(len(e1_final)),
                           desc="\t"*2 +
                           f"- 2D {ph.SPECTRUM_TYPES_NICE[spectrum_type]}",
                           ncols=100):

                for je in range(len(e2_final)):
                    e1 = e1_final[ie, je]
                    e2 = e2_final[ie, je]

                    if (e1+e2 <= self.q_value):
                        spectra_2d[ord][ie, je] = fact * self.standard_electron_integrant(e1, e2, full_func) *\
                            integral_order(e1, e2, self.q_value, ord)
                    else:
                        spectra_2d[ord][ie, je] = np.nan

        return spectra_2d


class Spectrum0nubb(Spectrum):
    def __init__(self, q_value: float, energy_points: np.ndarray, nuclear_radius: float) -> None:
        super().__init__(q_value, energy_points)
        self.nuclear_radius = nuclear_radius
        self.constant_in_front = ((ph.fermi_coupling_constant*ph.v_ud)**4) / \
            (32.*(np.pi**5)*(nuclear_radius**2.)) * \
            (ph.electron_mass**2.0)*(ph.hc**2.0)

    def standard_electron_integrant(self, e1, fermi_func: Callable):
        p1 = np.sqrt(e1*(e1+2.0*ph.electron_mass))
        e2 = self.q_value-e1
        p2 = np.sqrt(e2*(e2+2.0*ph.electron_mass))

        return fermi_func(e1)*fermi_func(e2) * \
            (e1+ph.electron_mass) * (e2+ph.electron_mass) * \
            p1*p2

    def compute_spectrum(self, spectrum_type: int, i_spectrum: int, fermi_func: Callable, eta_total: Callable | None = None):
        spectrum = np.zeros_like(self.energy_points)

        if not (eta_total is None):
            @lru_cache(maxsize=None)
            def full_func(e1): return fermi_func(e1)*(1.0+eta_total(e1))
        else:
            @lru_cache(maxsize=None)
            def full_func(e1): return fermi_func(e1)

        if spectrum_type == ph.SINGLESPECTRUM:
            if (i_spectrum == 1):
                def spectrum_func(e1):
                    return self.standard_electron_integrant(e1, full_func)
            else:
                # throw not-implmeneted error
                raise NotImplementedError

            spectrum_function = spectrum_func
        elif spectrum_type == ph.ANGULARSPECTRUM:
            if (i_spectrum == 1):
                def spectrum_func(e1):
                    return -1.0*self.standard_electron_integrant(e1, full_func)
            else:
                # throw not-implmeneted error
                raise NotImplementedError

            spectrum_function = spectrum_func

        for i_e in tqdm(range(len(self.energy_points)-1),
                        desc="\t"*2 +
                        f"- {ph.SPECTRUM_TYPES_NICE[spectrum_type]}",
                        ncols=100):
            e1 = self.energy_points[i_e]
            spectrum[i_e] = spectrum_function(e1)

        spectrum[-1] = 0.
        return spectrum

    def compute_psf(self, spec):
        psf_mev = spec*self.constant_in_front
        psf_years = psf_mev/(ph.hbar*np.log(2.))/(ph.year**(-1))
        return psf_years

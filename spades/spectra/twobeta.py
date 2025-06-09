from abc import abstractmethod

from spades.fermi_functions import FermiFunctions
from spades.spectra.base import BetaSpectrumBase
from abc import abstractmethod
from functools import lru_cache
from typing import Callable
from scipy import integrate, interpolate
from tqdm import tqdm
from spades import ph
import numpy as np
from spades.spectra.closure_helpers import neutrino_integrand_closure_standard_00, neutrino_integrand_closure_standard_02, neutrino_integrand_closure_angular_00, neutrino_integrand_closure_angular_02
from spades.spectra.taylor_helpers import integral_order


@lru_cache(maxsize=None)
def standard_electron_integrant_2nubb(e1, e2, fermi_func: Callable) -> float:
    """Computes the electron (positron) integrant for "Standard" (i.e. G) psfs/spectra

    Args:
        e1 (_type_): kinetic energy of first electron (positron)
        e2 (_type_): kinetic energy of second electron (positron)
        fermi_func (Callable): fermi function callable with signature f(e)

    Returns:
        float: value of integrant
    """
    return fermi_func(e1)*fermi_func(e2) * \
        (e1+ph.electron_mass) * (e2+ph.electron_mass) * \
        np.sqrt(e1*(e1+2.0*ph.electron_mass)) * \
        np.sqrt(e2*(e2+2.0*ph.electron_mass))


def spectrum_integrant_closure_2nubb(enu: float, e2: float, e1: float, total_ke: float,
                                     sp_type: ph.SpectrumTypes, emin: float, enei: float, full_func: Callable,
                                     transition: ph.TransitionTypes) -> float:
    """Full Spectrum integrant in closure approximation

    Args:
        enu (float): energy of first (anti-)neutrino. 
        e2 (float): kinetic energy of second electron (positron)
        e1 (float): kinetic energy of first electron (positron)
        total_ke (float): total kinetic energy available in the process
        sp_type (SpectrumTypes): type of spectrum
        emin (float): starting point of integration on electron (positron) energy
        enei (float): <E_N> - E_I
        full_func (Callable): Fermi function, possibly including corrections. Signature f(e)
        transition (TransitionTypes): type of transition

    Raises:
        NotImplementedError: if the (spectrum type,transition type) is unknown

    Returns:
        float: integrant value
    """

    # compute total energies
    et1 = e1+ph.electron_mass
    et2 = e2+ph.electron_mass
    enu2 = total_ke - e1 - e2 - enu
    if sp_type == ph.SpectrumTypes.SINGLESPECTRUM:
        if (transition == ph.TransitionTypes.ZEROPLUS_TO_ZEROPLUS) or (transition == ph.TransitionTypes.ZEROPLUS_TO_ZEROTWOPLUS):
            return standard_electron_integrant_2nubb(e1, e2, full_func) *\
                neutrino_integrand_closure_standard_00(
                    enu, et1, et2, enu2, enei)
        elif (transition == ph.TransitionTypes.ZEROPLUS_TO_TWOPLUS):
            return standard_electron_integrant_2nubb(e1, e2, full_func) *\
                neutrino_integrand_closure_standard_02(
                    enu, et1, et2, enu2, enei)
        else:
            raise NotImplementedError()
    elif sp_type == ph.SpectrumTypes.SUMMEDSPECTRUM:
        t = e1
        v = e2
        ee2 = t*v/total_ke
        ee1 = t - ee2

        et1 = ee1+ph.electron_mass
        et2 = ee2+ph.electron_mass
        enu2 = total_ke + 2*ph.electron_mass - et1 - et2 - enu

        if (ee1 < emin) or (ee2 < emin):
            return 0.
        if (transition == ph.TransitionTypes.ZEROPLUS_TO_ZEROPLUS) or (transition == ph.TransitionTypes.ZEROPLUS_TO_ZEROTWOPLUS):
            ret_val = t/total_ke*standard_electron_integrant_2nubb(ee1, ee2, full_func) *\
                neutrino_integrand_closure_standard_00(
                    enu, et1, et2, enu2, enei)
        elif transition == ph.TransitionTypes.ZEROPLUS_TO_TWOPLUS:
            ret_val = t/total_ke*standard_electron_integrant_2nubb(ee1, ee2, full_func) *\
                neutrino_integrand_closure_standard_02(
                    enu, et1, et2, enu2, enei)
        else:
            raise NotImplementedError()
        return ret_val
    elif sp_type == ph.SpectrumTypes.ANGULARSPECTRUM:
        if (transition == ph.TransitionTypes.ZEROPLUS_TO_ZEROPLUS) or (transition == ph.TransitionTypes.ZEROPLUS_TO_ZEROTWOPLUS):
            return -1.0*standard_electron_integrant_2nubb(e1, e2, full_func) *\
                neutrino_integrand_closure_angular_00(
                    enu, et1, et2, enu2, enei)
        else:
            return -1.0*standard_electron_integrant_2nubb(e1, e2, full_func) *\
                neutrino_integrand_closure_angular_02(
                    enu, et1, et2, enu2, enei)

    else:
        raise NotImplementedError("Unknown spectrum type")


def range_enu(e2: float, e1: float, total_ke: float, sp_type: ph.SpectrumTypes, emin: float, enei: float, full_func: Callable, transition: ph.TransitionTypes):
    """Function to be used as range of integration over (anti-)neutrino energies.

    Args:
        e2 (float): Kinetic energy of second electron (positron)
        e1 (float): Kinetic energy of first electron (positron)
        total_ke (float): Total kinetic energy available in the process
        sp_type (SpectrumType): Spectrum type. 
        emin (float): Not used. It's here for compatibility with scipy.quad
        enei (float): Not used. It's here for compatibility with scipy.quad
        full_func (Callable): Not used. It's here for compatibility with scipy.quad
        transition (TransitionTypes): Not used. It's here for compatibility with scipy.quad

    Returns:
        [enu_min, enu_max]: range of integration suitable for use with scipy.quad
    """
    if sp_type == ph.SpectrumTypes.SINGLESPECTRUM:
        return [0., total_ke-e1-e2]
    elif sp_type == ph.SpectrumTypes.SUMMEDSPECTRUM:
        t = e1
        v = e2
        return [0., total_ke-t]
    elif sp_type == ph.SpectrumTypes.ANGULARSPECTRUM:
        return [0., total_ke-e1-e2]


def range_e2(e1: float, total_ke: float, sp_type: ph.SpectrumTypes, emin: float, enei: float,  full_func: Callable, transition: ph.TransitionTypes):
    """Function to be used as range of integration over second electron (positron) energies.

    Args:
        e1 (float): Kinetic energy of first electron (positron)
        total_ke (float): Total kinetic energy available in the process
        sp_type (SpectrumType): Spectrum type 
        emin (float): Lowest kinetic energy used in integration over electron energies
        enei (float): Not used. It's here for compatibility with scipy.quad
        full_func (Callable): Not used. It's here for compatibility with scipy.quad
        transition (TransitionTypes): Not used. It's here for compatibility with scipy.quad

    Returns:
        [emin, emax]: range of kinetic energy for second electron (positron)
    """
    if sp_type == ph.SpectrumTypes.SINGLESPECTRUM:
        return [emin, total_ke-e1]
    elif sp_type == ph.SpectrumTypes.SUMMEDSPECTRUM:
        return [emin, total_ke-emin]
    elif sp_type == ph.SpectrumTypes.ANGULARSPECTRUM:
        return [emin, total_ke-e1]


class TwoBetaSpectrumBase(BetaSpectrumBase):
    """Base class for the 2betaPlus and 2betaMinus spectra computation.
    """

    def __init__(self, total_ke: float, ei_ef: float, fermi_functions: FermiFunctions, **kwargs) -> None:
        """
        Args:
            total_ke (float): total kinetic energy available in the rpcess
            ei_ef (float): difference between initial and final NUCLEAR energy levels
            fermi_functions (FermiFunctions): fermi functions to be used in the defintion of spectra

        Keyword args:
            energy_grid_type(str): can be lin or log. Defaults to lin
            min_ke(float): minimum kinetic energy (MeV) for electrons (positrons). Defaults to 1E-4
            n_ke_points(int): number of points along kinetic energy grid. Defaults to 100.
        Raises:
            NotImplementedError: in case of unknown energy grid type
        """
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


class TaylorSpectrumBase(TwoBetaSpectrumBase):
    def __init__(self, total_ke: float, ei_ef: float, fermi_functions: FermiFunctions, taylor_order: ph.TaylorOrders, **kwargs) -> None:
        super().__init__(total_ke, ei_ef, fermi_functions, **kwargs)
        self.taylor_order = taylor_order
        self.constant_in_front = 1.0

    @abstractmethod
    def compute_spectrum(self, sp_type: ph.SpectrumTypes):
        pass

    @abstractmethod
    def compute_2D_spectrum(self, sp_type: ph.SpectrumTypes):
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
            psf_mev = self.spectrum_integrals[key] * self.constant_in_front
            psf_years = psf_mev/(ph.hbar*np.log(2.))/(ph.year**(-1))
            self.psfs[key] = psf_years


def spectrum_integrant_taylor_2nubb(e2, e1, total_ke: float, sp_type: ph.SpectrumTypes, transition_type: ph.TransitionTypes, order: ph.TaylorOrders, full_func: Callable, emin: float):
    if sp_type == ph.SpectrumTypes.SINGLESPECTRUM:
        return standard_electron_integrant_2nubb(e1, e2, full_func)*integral_order(e1, e2, total_ke, order, transition_type)
    elif sp_type == ph.SpectrumTypes.SUMMEDSPECTRUM:
        t = e1
        v = e2
        ee2 = t*v/total_ke
        ee1 = t - ee2

        if (ee1 < emin) or (ee2 < emin):
            return 0.

        return t/total_ke * standard_electron_integrant_2nubb(ee1, ee2, full_func) * integral_order(ee1, ee2, total_ke, order, transition_type)
    elif sp_type == ph.SpectrumTypes.ANGULARSPECTRUM:
        return -1.0*standard_electron_integrant_2nubb(e1, e2, full_func)*integral_order(e1, e2, total_ke, order, transition_type)
    else:
        raise NotImplementedError


class TaylorSpectrum2nu(TaylorSpectrumBase):
    def __init__(self, total_ke: float, ei_ef: float, fermi_functions: FermiFunctions, taylor_order: ph.TaylorOrders, eta_total: Callable | None, transition, **kwargs) -> None:
        super().__init__(total_ke, ei_ef, fermi_functions, taylor_order, **kwargs)
        self.transition = transition
        if (transition == ph.TransitionTypes.ZEROPLUS_TO_TWOPLUS):
            pass
        else:
            self.constant_in_front = ((ph.fermi_coupling_constant*ph.v_ud)**4) / \
                (8.*(np.pi**7)*ph.electron_mass**2.0)

        if (eta_total is None):
            self.eta_total = lambda x: 1.0
        else:
            self.eta_total = eta_total

    def full_func(self, x, sp_type):
        if sp_type == ph.SpectrumTypes.ANGULARSPECTRUM:
            return self.fermi_functions.ff1_eval(x)*self.eta_total(x)
        else:
            return self.fermi_functions.ff0_eval(x)*self.eta_total(x)

    def compute_spectrum(self, sp_type: ph.SpectrumTypes):
        self.spectrum_values[sp_type] = np.zeros_like(self.energy_points)
        for i_e in tqdm(range(len(self.energy_points)-1),
                        desc="\t"*2 +
                        f"- {ph.SPECTRUM_TYPES_NICE[sp_type]}",
                        ncols=100):
            e1 = self.energy_points[i_e]
            range_list = range_e2(e1, self.total_ke, sp_type,
                                  self.energy_points[0], 0, self.full_func, self.transition)
            result = integrate.quad(
                spectrum_integrant_taylor_2nubb,
                a=range_list[0],
                b=range_list[1],
                args=(e1,
                      self.total_ke,
                      sp_type,
                      self.transition,
                      self.taylor_order,
                      lambda x: self.full_func(x, sp_type),
                      self.energy_points[0]
                      ),
            )
            # print(result)
            if isinstance(result, tuple):
                self.spectrum_values[sp_type][i_e] = result[0]
            else:
                raise ValueError("Spectrum integration did not succeed")
        self.spectrum_values[sp_type][-1] = 0.

    def compute_2D_spectrum(self, sp_type: ph.SpectrumTypes):
        raise NotImplementedError()


class ClosureSpectrumBase(TwoBetaSpectrumBase):
    """Base Closure spectrum class
    """

    def __init__(self, total_ke: float, ei_ef: float, enei: float, fermi_functions: FermiFunctions, *args, **kwargs) -> None:
        """
        Args:
            total_ke (float): total kinetic energy available in the rpcess
            ei_ef (float): difference between initial and final NUCLEAR energy levels
            enei (float): <E_N> - E_I
            fermi_functions (FermiFunctions): fermi functions to be used in the defintion of spectra
        Keyword args:
            additional arguments passed to base class TwooBetaSpectrumBase
        """
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
    """Concrete implementation for closure spectrum for 2nu 2beta (plus or minus) decays.
    """

    def __init__(self, total_ke: float, ei_ef: float, enei: float, fermi_functions: FermiFunctions, eta_total: Callable | None,
                 transition: int, **kwargs) -> None:
        """
        Args:
            total_ke (float): total kinetic energy available in the rpcess
            ei_ef (float): difference between initial and final NUCLEAR energy levels
            enei (float): <E_N> - E_I
            eta_total (Callable | None): exchange correction function
            transition (int): transition type
        Keyword args:
            additional arguments passed to ClosureSpectrumBase
        """
        super().__init__(total_ke, ei_ef, enei, fermi_functions, **kwargs)
        self.transition = transition
        if (transition == ph.TransitionTypes.ZEROPLUS_TO_TWOPLUS):
            self.constant_in_front = ((self.atilde/ph.electron_mass)**6.0)*((ph.fermi_coupling_constant*ph.v_ud)**4) /\
                (96.*(np.pi**7))
        else:
            self.constant_in_front = ((self.atilde/ph.electron_mass)**2.0)*((ph.fermi_coupling_constant*ph.v_ud)**4) /\
                (96.*(np.pi**7))
        if (eta_total is None):
            self.eta_total = lambda x: 1.0
        else:
            self.eta_total = eta_total

        self.e1_grid_2D = kwargs.get("e1_grid_2D", None)
        self.e2_grid_2D = kwargs.get("e2_grid_2D", None)

    # @lru_cache(maxsize=None)
    def full_func(self, x, sp_type):
        if sp_type == ph.SpectrumTypes.ANGULARSPECTRUM:
            return self.fermi_functions.ff1_eval(x)*self.eta_total(x)
        else:
            return self.fermi_functions.ff0_eval(x)*self.eta_total(x)

    def compute_spectrum(self, sp_type: ph.SpectrumTypes):
        self.spectrum_values[sp_type] = np.zeros_like(self.energy_points)
        for i_e in tqdm(range(len(self.energy_points)-1),
                        desc="\t"*2 +
                        f"- {ph.SPECTRUM_TYPES_NICE[sp_type]}",
                        ncols=100):
            e1 = self.energy_points[i_e]
            result = integrate.nquad(
                spectrum_integrant_closure_2nubb,
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

    def compute_2D_spectrum(self, sp_type: ph.SpectrumTypes):
        self.spectrum_2D_values[sp_type] = np.zeros_like(self.e1_grid_2D)
        if (self.e1_grid_2D is None) or (self.e2_grid_2D is None):
            raise ValueError("2D energy grid not initialized properly")

        for ie in tqdm(range(len(self.e1_grid_2D)),
                       desc="\t"*2 +
                       f"- 2D {ph.SPECTRUM_TYPES_NICE[sp_type]}",
                       ncols=100):
            for je in range(len(self.e2_grid_2D)):
                e1 = self.e1_grid_2D[ie, je]
                e2 = self.e2_grid_2D[ie, je]

                if (e1+e2 <= self.total_ke):
                    result = integrate.quad(
                        func=spectrum_integrant_closure_2nubb,
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
    def full_func(self, x, sp_type, tr_type: ph.TransitionTypes = ph.TransitionTypes.ZEROPLUS_TO_ZEROPLUS):
        if tr_type == ph.TransitionTypes.ZEROPLUS_TO_TWOPLUS:
            raise NotImplementedError()

        if sp_type == ph.SpectrumTypes.ANGULARSPECTRUM:
            return self.fermi_functions.ff1_eval(x)*self.eta_total(x)
        else:
            return self.fermi_functions.ff0_eval(x)*self.eta_total(x)

    def compute_spectrum(self, sp_type: ph.SpectrumTypes):
        self.spectrum_values[sp_type] = np.zeros_like(self.energy_points)
        fact = 1.0
        if (sp_type == ph.SpectrumTypes.ANGULARSPECTRUM):
            fact = -1.0
        for i_e in tqdm(range(len(self.energy_points)-1),
                        desc="\t"*2+f"- {ph.SPECTRUM_TYPES_NICE[sp_type]}",
                        ncols=100):
            e1 = self.energy_points[i_e]
            self.spectrum_values[sp_type][i_e] = fact*standard_electron_integrant_0nubb(
                e1, self.total_ke, lambda x: self.full_func(x, sp_type))

        self.spectrum_values[sp_type][-1] = 0.
        return self.spectrum_values

    def compute_2D_spectrum(self, sp_type: int, e1_grid: np.ndarray, e2_grid: np.ndarray):
        raise NotImplementedError

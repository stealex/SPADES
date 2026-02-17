"""Spectra and PSF models for two-beta decay channels."""

from abc import abstractmethod

from spades.fermi_functions import FermiFunctions
from spades.spectra.base import BetaSpectrumBase
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
    """Evaluate the electron/positron kinematic integrand for ``G``-type observables.

    Parameters
    ----------
    e1:
        Kinetic energy of the first lepton (MeV in internal units).
    e2:
        Kinetic energy of the second lepton (MeV in internal units).
    fermi_func:
        Callable that returns the Fermi-function factor at a given kinetic energy.

    Returns
    -------
    float
        Value of the phase-space kinematic prefactor including Fermi functions.
    """
    return fermi_func(e1)*fermi_func(e2) * \
        (e1+ph.electron_mass) * (e2+ph.electron_mass) * \
        np.sqrt(e1*(e1+2.0*ph.electron_mass)) * \
        np.sqrt(e2*(e2+2.0*ph.electron_mass))


def spectrum_integrant_closure_2nubb(enu: float, e2: float, e1: float, total_ke: float,
                                     sp_type: ph.SpectrumTypes, emin: float, enei: float, full_func: Callable,
                                     transition: ph.TransitionTypes) -> float:
    """Closure-approximation integrand for 2nu two-beta spectra.

    Parameters
    ----------
    enu:
        Energy of the first neutrino/antineutrino.
    e2:
        Kinetic energy of the second charged lepton.
    e1:
        Kinetic energy of the first charged lepton (or summed variable for ``SUMMEDSPECTRUM`` path).
    total_ke:
        Total kinetic energy available to leptons.
    sp_type:
        Target spectrum type.
    emin:
        Minimum charged-lepton kinetic energy used in integrations.
    enei:
        Closure parameter ``<E_N> - E_I``.
    full_func:
        Callable including Fermi and optional correction factors.
    transition:
        Nuclear transition type.

    Returns
    -------
    float
        Value of the full differential integrand.
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
    """Return integration bounds for neutrino energy in closure integrations.

    Parameters
    ----------
    e2, e1, total_ke, sp_type, emin, enei, full_func, transition:
        Arguments required by :func:`scipy.integrate.nquad` range callbacks.

    Returns
    -------
    list[float]
        Two-element list ``[lower, upper]`` with neutrino-energy bounds.
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
    """Return integration bounds for the second charged lepton kinetic energy.

    Parameters
    ----------
    e1:
        Kinetic energy of the first charged lepton.
    total_ke:
        Total available kinetic energy.
    sp_type:
        Spectrum type.
    emin:
        Lower kinetic-energy cutoff used by SPADES integrations.
    enei, full_func, transition:
        Unused placeholders required by :func:`scipy.integrate.nquad`.

    Returns
    -------
    list[float]
        Two-element list ``[lower, upper]`` for the second-energy integral.
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
        """Initialize common 2-beta spectrum settings and 1D kinetic-energy grid.

        Parameters
        ----------
        total_ke:
            Total kinetic energy available to leptons.
        ei_ef:
            Nuclear-level energy difference ``E_i - E_f``.
        fermi_functions:
            Fermi-function backend used in spectrum kernels.
        **kwargs:
            Optional controls:
            ``energy_grid_type`` (``"lin"`` or ``"log"``), ``min_ke``, ``n_ke_points``.
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
        """Compute one 1D spectrum for spectrum type ``sp_type``.

        Parameters
        ----------
        sp_type:
            Spectrum type selector.
        """
        pass

    @abstractmethod
    def compute_2D_spectrum(self, sp_type: int):
        """Compute one 2D spectrum for spectrum type ``sp_type``.

        Parameters
        ----------
        sp_type:
            Spectrum type selector.
        """
        pass

    @abstractmethod
    def compute_psf(self):
        """Compute PSFs from integrated spectra."""
        pass

    @abstractmethod
    def integrate_spectrum(self):
        """Integrate and normalize computed spectra."""
        pass


class TaylorSpectrumBase(TwoBetaSpectrumBase):
    """Base implementation for Taylor-expansion 2nu spectra."""

    def __init__(self, total_ke: float, ei_ef: float, fermi_functions: FermiFunctions, taylor_order: ph.TaylorOrders, **kwargs) -> None:
        """Initialize Taylor-order selection and base normalization constant.

        Parameters
        ----------
        total_ke, ei_ef:
            Process energy scales.
        fermi_functions:
            Fermi-function backend.
        taylor_order:
            Requested Taylor order.
        **kwargs:
            Forwarded grid settings.
        """
        super().__init__(total_ke, ei_ef, fermi_functions, **kwargs)
        self.taylor_order = taylor_order
        self.constant_in_front = 1.0

    @abstractmethod
    def compute_spectrum(self, sp_type: ph.SpectrumTypes):
        """Compute one 1D Taylor spectrum for spectrum type ``sp_type``.

        Parameters
        ----------
        sp_type:
            Spectrum type selector.
        """
        pass

    @abstractmethod
    def compute_2D_spectrum(self, sp_type: ph.SpectrumTypes):
        """Compute one 2D Taylor spectrum for spectrum type ``sp_type``.

        Parameters
        ----------
        sp_type:
            Spectrum type selector.
        """
        pass

    def integrate_spectrum(self):
        """Integrate and normalize each computed 1D spectrum."""
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
        """Convert integrated spectra to PSFs in user time units."""
        for key in self.spectrum_values:
            psf_mev = self.spectrum_integrals[key] * self.constant_in_front
            psf_years = psf_mev/(ph.hbar*np.log(2.))/(ph.year**(-1))
            self.psfs[key] = psf_years


def spectrum_integrant_taylor_2nubb(e2, e1, total_ke: float, sp_type: ph.SpectrumTypes, transition_type: ph.TransitionTypes, order: ph.TaylorOrders, full_func: Callable, emin: float):
    """Taylor-expansion integrand used for 2nu two-beta spectra.

    Parameters
    ----------
    e2, e1:
        Lepton kinetic-energy variables used by the corresponding spectrum definition.
    total_ke:
        Total available kinetic energy.
    sp_type:
        Requested spectrum type.
    transition_type:
        Nuclear transition type.
    order:
        Taylor expansion order.
    full_func:
        Fermi/correction factor callable.
    emin:
        Lower kinetic-energy cutoff.

    Returns
    -------
    float
        Integrand value for the requested spectrum type.
    """
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
    """Taylor-expansion implementation for 2nu two-beta spectra."""

    def __init__(self, total_ke: float, ei_ef: float, fermi_functions: FermiFunctions, taylor_order: ph.TaylorOrders, eta_total: Callable | None, transition, **kwargs) -> None:
        """Build 2nu Taylor spectra object with optional exchange correction factor.

        Parameters
        ----------
        total_ke, ei_ef:
            Process energy scales.
        fermi_functions:
            Fermi-function backend.
        taylor_order:
            Requested Taylor order.
        eta_total:
            Optional correction-factor callable.
        transition:
            Nuclear transition selector.
        **kwargs:
            Forwarded grid settings.
        """
        super().__init__(total_ke, ei_ef, fermi_functions, taylor_order, **kwargs)
        self.transition = transition
        if (transition == ph.TransitionTypes.ZEROPLUS_TO_TWOPLUS):
            self.constant_in_front = ((ph.fermi_coupling_constant*ph.v_ud)**4) / \
                (8.*(np.pi**7)*ph.electron_mass**2.0)
        else:
            self.constant_in_front = ((ph.fermi_coupling_constant*ph.v_ud)**4) / \
                (8.*(np.pi**7)*ph.electron_mass**2.0)

        if (eta_total is None):
            self.eta_total = lambda x: 1.0
        else:
            self.eta_total = eta_total

    def full_func(self, x, sp_type):
        """Return energy-dependent Fermi/correction factor for a given spectrum type.

        Parameters
        ----------
        x:
            Kinetic energy.
        sp_type:
            Spectrum type selector.
        """
        if sp_type == ph.SpectrumTypes.ANGULARSPECTRUM:
            return self.fermi_functions.ff1_eval(x)*self.eta_total(x)
        else:
            return self.fermi_functions.ff0_eval(x)*self.eta_total(x)

    def compute_spectrum(self, sp_type: ph.SpectrumTypes):
        """Compute the selected 1D Taylor spectrum over the configured energy grid.

        Parameters
        ----------
        sp_type:
            Spectrum type selector.
        """
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

            if isinstance(result, tuple):
                self.spectrum_values[sp_type][i_e] = result[0]
            else:
                raise ValueError("Spectrum integration did not succeed")
        self.spectrum_values[sp_type][-1] = 0.

    def compute_2D_spectrum(self, sp_type: ph.SpectrumTypes):
        """2D spectra are not implemented for Taylor 2nu in this class.

        Parameters
        ----------
        sp_type:
            Requested spectrum type.
        """
        raise NotImplementedError()


class ClosureSpectrumBase(TwoBetaSpectrumBase):
    """Base Closure spectrum class
    """

    def __init__(self, total_ke: float, ei_ef: float, enei: float, fermi_functions: FermiFunctions, *args, **kwargs) -> None:
        """Initialize closure-approximation settings.

        Parameters
        ----------
        total_ke:
            Total kinetic energy available to leptons.
        ei_ef:
            Nuclear-level energy difference ``E_i - E_f``.
        enei:
            Closure parameter ``<E_N> - E_I``.
        fermi_functions:
            Fermi-function backend.
        *args, **kwargs:
            Forwarded to :class:`TwoBetaSpectrumBase`.
        """
        super().__init__(total_ke, ei_ef, fermi_functions, **kwargs)
        self.enei = enei
        self.atilde = self.enei + 0.5*ei_ef
        self.constant_in_front = 1.0

    @abstractmethod
    def compute_spectrum(self, sp_type: int):
        """Compute one 1D closure spectrum for spectrum type ``sp_type``.

        Parameters
        ----------
        sp_type:
            Spectrum type selector.
        """
        pass

    @abstractmethod
    def compute_2D_spectrum(self, sp_type: int):
        """Compute one 2D closure spectrum for spectrum type ``sp_type``.

        Parameters
        ----------
        sp_type:
            Spectrum type selector.
        """
        pass

    def integrate_spectrum(self):
        """Integrate and normalize each computed closure spectrum."""
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
        """Convert integrated closure spectra to PSFs in user time units."""
        for key in self.spectrum_values:
            psf_mev = self.spectrum_integrals[key]*self.constant_in_front
            psf_years = psf_mev/(ph.hbar*np.log(2.))/(ph.year**(-1))
            self.psfs[key] = psf_years


class ClosureSpectrum2nu(ClosureSpectrumBase):
    """Concrete implementation for closure spectrum for 2nu 2beta (plus or minus) decays.
    """

    def __init__(self, total_ke: float, ei_ef: float, enei: float, fermi_functions: FermiFunctions, eta_total: Callable | None,
                 transition: int, **kwargs) -> None:
        """Construct closure 2nu two-beta spectrum model.

        Parameters
        ----------
        total_ke, ei_ef, enei:
            Energy scales for the process and closure approximation.
        fermi_functions:
            Fermi-function backend.
        eta_total:
            Optional exchange-correction factor callable.
        transition:
            Transition type from :class:`spades.ph.TransitionTypes`.
        **kwargs:
            Optional grid/2D settings forwarded to base classes.
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
        """Return Fermi and correction factor for one outgoing-lepton kinetic energy.

        Parameters
        ----------
        x:
            Kinetic energy.
        sp_type:
            Spectrum type selector.
        """
        if sp_type == ph.SpectrumTypes.ANGULARSPECTRUM:
            return self.fermi_functions.ff1_eval(x)*self.eta_total(x)
        else:
            return self.fermi_functions.ff0_eval(x)*self.eta_total(x)

    def compute_spectrum(self, sp_type: ph.SpectrumTypes):
        """Compute one 1D closure spectrum by nested neutrino/lepton integration.

        Parameters
        ----------
        sp_type:
            Spectrum type selector.
        """
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
        """Compute one 2D closure spectrum on pre-configured ``(e1, e2)`` meshgrids.

        Parameters
        ----------
        sp_type:
            Spectrum type selector.
        """
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
    """Evaluate the two-lepton kinematic prefactor for 0nu modes.

    Parameters
    ----------
    e1:
        Kinetic energy of the first lepton.
    total_ke:
        Total available kinetic energy.
    fermi_func:
        Callable returning the Fermi/correction factor at a given energy.

    Returns
    -------
    float
        Integrand value after enforcing ``e2 = total_ke - e1``.
    """
    e2 = total_ke - e1
    return fermi_func(e1)*fermi_func(e2) * \
        (e1+ph.electron_mass) * (e2+ph.electron_mass) * \
        np.sqrt(e1*(e1+2.0*ph.electron_mass)) * \
        np.sqrt(e2*(e2+2.0*ph.electron_mass))


class ClosureSpectrum0nu_LNE(ClosureSpectrumBase):
    """Closure model for neutrinoless 2-beta with light-neutrino exchange."""

    def __init__(self, total_ke: float, ei_ef: float, nuclear_radius: float, fermi_functions: FermiFunctions, eta_total: Callable | None, **kwargs) -> None:
        """Initialize LNE prefactor and optional exchange correction.

        Parameters
        ----------
        total_ke, ei_ef:
            Process energy scales.
        nuclear_radius:
            Nuclear radius in fm.
        fermi_functions:
            Fermi-function backend.
        eta_total:
            Optional correction-factor callable.
        **kwargs:
            Forwarded grid settings.
        """
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
        """Return Fermi and correction factor for a given kinetic energy and spectrum type.

        Parameters
        ----------
        x:
            Kinetic energy.
        sp_type:
            Spectrum type selector.
        tr_type:
            Transition selector.
        """
        if tr_type == ph.TransitionTypes.ZEROPLUS_TO_TWOPLUS:
            raise NotImplementedError()

        if sp_type == ph.SpectrumTypes.ANGULARSPECTRUM:
            return self.fermi_functions.ff1_eval(x)*self.eta_total(x)
        else:
            return self.fermi_functions.ff0_eval(x)*self.eta_total(x)

    def compute_spectrum(self, sp_type: ph.SpectrumTypes):
        """Compute 0nu closure spectrum over the configured 1D energy grid.

        Parameters
        ----------
        sp_type:
            Spectrum type selector.

        Returns
        -------
        dict
            Updated ``self.spectrum_values`` mapping.
        """
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
        """Not implemented for this 0nu closure class.

        Parameters
        ----------
        sp_type:
            Spectrum type selector.
        e1_grid, e2_grid:
            2D energy meshgrids.
        """
        raise NotImplementedError

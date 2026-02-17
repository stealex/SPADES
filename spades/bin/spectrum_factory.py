"""Factories for constructing spectrum objects by process and method."""

from __future__ import annotations

import logging
from typing import Callable

from spades import ph
from spades.dhfs import AtomicSystem
from spades.spectra.base import SpectrumBase
import spades.spectra.twobeta
import spades.spectra.ecbeta
import spades.spectra.twoec
from spades.wavefunctions import WaveFunctionsHandler
from spades import fermi_functions
from spades.config import RunConfig

logger = logging.getLogger(__name__)


def _resolve_enei_and_orders(input_config: RunConfig, final_atom: AtomicSystem) -> tuple[float | None, list[ph.TaylorOrders] | None]:
    """Resolve optional ``enei`` and Taylor orders from spectra method configuration.

    Parameters
    ----------
    input_config:
        Run configuration containing method options.
    final_atom:
        Final atomic system used for ``enei='auto'`` evaluation.

    Returns
    -------
    tuple[float | None, list[ph.TaylorOrders] | None]
        Resolved ``enei`` value and list of Taylor orders (or ``None`` when absent).
    """
    atilde = 1.12 * (final_atom.mass_number**0.5)
    enei = None
    orders = None

    if "enei" in input_config.spectra_config.method:
        enei_raw = input_config.spectra_config.method["enei"]
        if isinstance(enei_raw, float):
            enei = enei_raw
        elif isinstance(enei_raw, str):
            if enei_raw != "auto":
                raise ValueError("Cannot interpret enei option")
            enei = atilde - 0.5 * input_config.spectra_config.ei_ef
        else:
            raise ValueError("Cannot interpret enei option")

    if "orders" in input_config.spectra_config.method:
        orders_raw = input_config.spectra_config.method["orders"]
        if isinstance(orders_raw, list):
            orders = [ph.TAYLOR_ORDER_NAMES_MAP[str(key)] for key in orders_raw]
        elif isinstance(orders_raw, str):
            if orders_raw != "auto":
                raise ValueError("Cannot interpret orders option")
            if input_config.process.transition == ph.TransitionTypes.ZEROPLUS_TO_TWOPLUS:
                orders = [ph.TaylorOrders.TWOTWO, ph.TaylorOrders.SIX]
            else:
                orders = [
                    ph.TaylorOrders.ZERO,
                    ph.TaylorOrders.TWO,
                    ph.TaylorOrders.TWOTWO,
                    ph.TaylorOrders.FOUR,
                ]
        else:
            raise ValueError("Cannot interpret orders option")

    return enei, orders


def _create_twobeta_2nu(
    input_config: RunConfig,
    fermi_func: fermi_functions.FermiFunctions | None,
    eta_total: Callable | None,
    enei: float | None,
    orders: list[ph.TaylorOrders] | None,
    e1_grid_2D,
    e2_grid_2D,
) -> SpectrumBase | dict:
    """Create 2nu two-beta spectrum object(s) for Closure or Taylor methods.

    Parameters
    ----------
    input_config:
        Run configuration.
    fermi_func:
        Fermi-function backend.
    eta_total:
        Optional correction function.
    enei:
        Closure-energy parameter.
    orders:
        Taylor orders when Taylor method is selected.
    e1_grid_2D, e2_grid_2D:
        Optional 2D energy grids.

    Returns
    -------
    SpectrumBase | dict
        Single spectrum object or order-indexed dictionary.
    """
    if fermi_func is None:
        raise ValueError("Fermi functions are required for 2nu two-beta spectra.")
    method_name = input_config.spectra_config.method["name"]
    if method_name == "Closure":
        if enei is None:
            raise ValueError("Closure method requires 'enei' configuration.")
        logger.debug("Creating two-beta 2nu closure spectrum")
        return spades.spectra.twobeta.ClosureSpectrum2nu(
            total_ke=input_config.spectra_config.total_ke,
            ei_ef=input_config.spectra_config.ei_ef,
            enei=enei,
            fermi_functions=fermi_func,
            eta_total=eta_total,
            transition=input_config.process.transition,
            min_ke=input_config.spectra_config.min_ke,
            n_ke_points=input_config.spectra_config.n_ke_points,
            energy_grid_type=input_config.spectra_config.energy_grid_type,
            e1_grid_2D=e1_grid_2D,
            e2_grid_2D=e2_grid_2D,
        )
    if method_name == "Taylor":
        if not orders:
            raise ValueError("Taylor method requires 'orders' configuration.")
        logger.info("Creating two-beta 2nu Taylor spectra for %d orders", len(orders))
        spectra = {}
        for order in orders:
            spectra[order] = spades.spectra.twobeta.TaylorSpectrum2nu(
                total_ke=input_config.spectra_config.total_ke,
                ei_ef=input_config.spectra_config.ei_ef,
                fermi_functions=fermi_func,
                eta_total=eta_total,
                taylor_order=order,
                transition=input_config.process.transition,
                min_ke=input_config.spectra_config.min_ke,
                n_ke_points=input_config.spectra_config.n_ke_points,
                energy_grid_type=input_config.spectra_config.energy_grid_type,
                e1_grid_2D=e1_grid_2D,
                e2_grid_2D=e2_grid_2D,
            )
        return spectra
    raise NotImplementedError(f"Method '{method_name}' not implemented for 2nu two-beta.")


def _create_twobeta_0nu(
    input_config: RunConfig,
    fermi_func: fermi_functions.FermiFunctions | None,
    eta_total: Callable | None,
) -> SpectrumBase:
    """Create neutrinoless two-beta closure spectrum object.

    Parameters
    ----------
    input_config:
        Run configuration.
    fermi_func:
        Fermi-function backend.
    eta_total:
        Optional correction function.

    Returns
    -------
    SpectrumBase
        Configured neutrinoless two-beta spectrum object.
    """
    if fermi_func is None:
        raise ValueError("Fermi functions are required for neutrinoless two-beta spectra.")
    if input_config.spectra_config.method["name"] != "Closure":
        raise NotImplementedError("Only Closure method is implemented for 0nu two-beta.")
    if input_config.process.mechanism != ph.NeutrinoLessModes.LIGHT_NEUTRINO_EXCHANGE:
        raise NotImplementedError("Only LNE mechanism is implemented for 0nu two-beta.")
    return spades.spectra.twobeta.ClosureSpectrum0nu_LNE(
        total_ke=input_config.spectra_config.total_ke,
        ei_ef=input_config.spectra_config.ei_ef,
        nuclear_radius=input_config.spectra_config.nuclear_radius,
        fermi_functions=fermi_func,
        eta_total=eta_total,
        min_ke=input_config.spectra_config.min_ke,
        n_ke_points=input_config.spectra_config.n_ke_points,
        energy_grid_type=input_config.spectra_config.energy_grid_type,
    )


def _create_ecbeta_2nu(
    input_config: RunConfig,
    fermi_func: fermi_functions.FermiFunctions | None,
    wf_handler_init: WaveFunctionsHandler | None,
    enei: float | None,
    e1_grid_2D,
    e2_grid_2D,
) -> SpectrumBase:
    """Create 2nu ECbeta+ closure spectrum object.

    Parameters
    ----------
    input_config:
        Run configuration.
    fermi_func:
        Fermi-function backend.
    wf_handler_init:
        Initial-state wavefunction handler.
    enei:
        Closure-energy parameter.
    e1_grid_2D, e2_grid_2D:
        Optional 2D energy grids.

    Returns
    -------
    SpectrumBase
        Configured 2nu ECbeta+ spectrum object.
    """
    if wf_handler_init is None:
        raise ValueError("Received None for wf_handler_init. Cannot compute EC without it.")
    if fermi_func is None:
        raise ValueError("Fermi functions are required for 2nu ECbeta+ spectra.")
    if enei is None:
        raise ValueError("Closure method requires 'enei' configuration.")
    if input_config.spectra_config.method["name"] != "Closure":
        raise NotImplementedError("Only Closure method is implemented for 2nu ECbeta+.")
    return spades.spectra.ecbeta.ClosureSpectrum2nu(
        total_ke=input_config.spectra_config.total_ke,
        ei_ef=input_config.spectra_config.ei_ef,
        fermi_functions=fermi_func,
        bound_handler=wf_handler_init.bound_handler,
        nuclear_radius=input_config.spectra_config.nuclear_radius,
        enei=enei,
        transition_type=input_config.process.transition,
        min_ke=input_config.spectra_config.min_ke,
        n_ke_points=input_config.spectra_config.n_ke_points,
        energy_grid_type=input_config.spectra_config.energy_grid_type,
        e1_grid_2D=e1_grid_2D,
        e2_grid_2D=e2_grid_2D,
    )


def _create_ecbeta_0nu(
    input_config: RunConfig,
    fermi_func: fermi_functions.FermiFunctions | None,
    wf_handler_init: WaveFunctionsHandler | None,
    enei: float | None,
) -> SpectrumBase:
    """Create 0nu ECbeta+ closure spectrum object.

    Parameters
    ----------
    input_config:
        Run configuration.
    fermi_func:
        Fermi-function backend.
    wf_handler_init:
        Initial-state wavefunction handler.
    enei:
        Closure-energy parameter.

    Returns
    -------
    SpectrumBase
        Configured 0nu ECbeta+ spectrum object.
    """
    if wf_handler_init is None:
        raise ValueError("Received None for wf_handler_init. Cannot compute EC without it.")
    if fermi_func is None:
        raise ValueError("Fermi functions are required for 0nu ECbeta+ spectra.")
    if enei is None:
        raise ValueError("Closure method requires 'enei' configuration.")
    if input_config.spectra_config.method["name"] != "Closure":
        raise NotImplementedError("Only Closure method is implemented for 0nu ECbeta+.")
    return spades.spectra.ecbeta.ClosureSpectrum0nu_LNE(
        total_ke=input_config.spectra_config.total_ke,
        ei_ef=input_config.spectra_config.ei_ef,
        fermi_functions=fermi_func,
        bound_handler=wf_handler_init.bound_handler,
        nuclear_radius=input_config.spectra_config.nuclear_radius,
        enei=enei,
    )


def _create_twoec(
    input_config: RunConfig,
    wf_handler_init: WaveFunctionsHandler | None,
    enei: float | None,
    orders: list[ph.TaylorOrders] | None,
) -> SpectrumBase | dict:
    """Create 2EC spectrum object(s) for Closure or Taylor methods.

    Parameters
    ----------
    input_config:
        Run configuration.
    wf_handler_init:
        Initial-state wavefunction handler.
    enei:
        Closure-energy parameter.
    orders:
        Taylor orders when Taylor method is selected.

    Returns
    -------
    SpectrumBase | dict
        Single spectrum object or order-indexed dictionary.
    """
    if wf_handler_init is None:
        raise ValueError("Received None for wf_handler_init. Cannot compute 2EC without it.")
    method_name = input_config.spectra_config.method["name"]
    if method_name == "Closure":
        if enei is None:
            raise ValueError("Closure method requires 'enei' configuration.")
        return spades.spectra.twoec.TwoECSpectrumClosure(
            total_ke=input_config.spectra_config.total_ke,
            ei_ef=input_config.spectra_config.ei_ef,
            bound_handler=wf_handler_init.bound_handler,
            nuclear_radius=input_config.spectra_config.nuclear_radius,
            enei=enei,
            transition_type=input_config.process.transition,
        )
    if method_name == "Taylor":
        if not orders:
            raise ValueError("Taylor method requires 'orders' configuration.")
        spectra = {}
        for order in orders:
            spectra[order] = spades.spectra.twoec.TwoECSpectrumTaylor(
                total_ke=input_config.spectra_config.total_ke,
                ei_ef=input_config.spectra_config.ei_ef,
                bound_handler=wf_handler_init.bound_handler,
                nuclear_radius=input_config.spectra_config.nuclear_radius,
                transition_type=input_config.process.transition,
                order=order,
            )
        return spectra
    raise NotImplementedError(f"Method '{method_name}' not implemented for 2EC.")


def create_spectrum(
    input_config: RunConfig,
    fermi_func: fermi_functions.FermiFunctions | None,
    eta_total: Callable | None,
    final_atom: AtomicSystem,
    wf_handler_init: WaveFunctionsHandler | None,
    e1_grid_2D=None,
    e2_grid_2D=None,
) -> SpectrumBase | dict:
    """Create spectrum object(s) for the configured process and method.

    Parameters
    ----------
    input_config:
        Run configuration.
    fermi_func:
        Fermi-function backend, if required by the process.
    eta_total:
        Optional correction function.
    final_atom:
        Final atomic system.
    wf_handler_init:
        Initial-state wavefunction handler for capture channels.
    e1_grid_2D, e2_grid_2D:
        Optional 2D energy grids.

    Returns
    -------
    SpectrumBase | dict
        Constructed spectrum object(s).
    """
    enei, orders = _resolve_enei_and_orders(input_config, final_atom)

    process_type = input_config.process.type
    if process_type in [ph.ProcessTypes.TWONEUTRINO_TWOBMINUS, ph.ProcessTypes.TWONEUTRINO_TWOBPLUS]:
        return _create_twobeta_2nu(input_config, fermi_func, eta_total, enei, orders, e1_grid_2D, e2_grid_2D)
    if process_type in [ph.ProcessTypes.NEUTRINOLESS_TWOBMINUS, ph.ProcessTypes.NEUTRINOLESS_TWOBPLUS]:
        return _create_twobeta_0nu(input_config, fermi_func, eta_total)
    if process_type == ph.ProcessTypes.TWONEUTRINO_BPLUSEC:
        return _create_ecbeta_2nu(input_config, fermi_func, wf_handler_init, enei, e1_grid_2D, e2_grid_2D)
    if process_type == ph.ProcessTypes.NEUTRINOLESS_BPLUSEC:
        return _create_ecbeta_0nu(input_config, fermi_func, wf_handler_init, enei)
    if process_type == ph.ProcessTypes.TWONEUTRINO_TWOEC:
        return _create_twoec(input_config, wf_handler_init, enei, orders)
    raise NotImplementedError(f"Unsupported process type: {process_type}")

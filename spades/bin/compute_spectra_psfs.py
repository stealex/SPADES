#!/usr/bin/env python
"""CLI workflow to compute SPADES spectra and phase-space factors."""

from typing import Callable
from scipy import interpolate
import yaml
from argparse import ArgumentParser
import logging
import time
import os
import numpy as np

from spades import fermi_functions, math_stuff, ph, exchange
from spades.config import RunConfig
from spades.dhfs import AtomicSystem, create_ion
from spades.spectra.base import SpectrumBase
from spades.wavefunctions import WaveFunctionsHandler
from spades.spectra.spectrum_writer import SpectrumWriter
from spades.bin.spectrum_factory import create_spectrum as create_spectrum_from_factory

logger = logging.getLogger(__name__)


def parse_input(argv=None):
    """Parse CLI arguments and build resolved runtime objects.

    Returns
    -------
    tuple[RunConfig, AtomicSystem, AtomicSystem]
        Resolved run configuration, initial atom, and final atom.

    Parameters
    ----------
    argv:
        Optional CLI argument list. If ``None``, ``sys.argv`` is used.
    """
    logger.info("Parsing input arguments")
    parser = ArgumentParser(
        description="Compute spectra and PSFs for 2nubb decay")
    parser.add_argument("config_file",
                        help="path to yaml configuration file")
    parser.add_argument("--verbose",
                        help="verbosity level: 0 = lowest; 5 = highest",
                        type=int,
                        action="store",
                        choices=[0, 1, 2, 3, 4, 5],
                        default=0)
    parser.add_argument("--energy_unit",
                        help="any CLHEP-defined energy unit or 'electron_mass' or 'hartree_energy'",
                        type=str,
                        action="store",
                        default="MeV")
    parser.add_argument("--distance_unit",
                        help="any CLHEP-defined distance unit or 'bohr_radius'",
                        type=str,
                        action="store",
                        default="fm")
    parser.add_argument("--qvalues_file",
                        help="File storing Q-values in MeV. YAML file with ANuc: Qval entries",
                        type=str,
                        action="store",
                        default=ph.delta_m_files)
    parser.add_argument("--compute_2d_spectrum",
                        action="store_true",
                        default=False,
                        help="Compute 2D spectrum as well as 1D spectrum")

    args = parser.parse_args(argv)
    runtime_settings = ph.RuntimeSettings(
        verbose=args.verbose,
        distance_unit_name=args.distance_unit,
        energy_unit_name=args.energy_unit,
        qvalues_file=args.qvalues_file,
    )
    ph.apply_runtime_settings(runtime_settings)
    logging.basicConfig(level=10*(5-runtime_settings.verbose))

    with open(args.config_file, 'r') as f:
        conf = yaml.safe_load(f)

    input_config = RunConfig(conf)
    initial_atom, final_atom = create_atoms(input_config)
    input_config.resolve_nuclear_radius(initial_atom)
    input_config.resolve_ei_ef(initial_atom, final_atom)
    input_config.create_spectra_config()
    input_config.create_scattering_config()
    input_config.resolve_bound_config(initial_atom)
    input_config.create_bound_config()
    input_config.create_output_config()

    return (input_config, initial_atom, final_atom)


def create_atoms(input_config: RunConfig):
    """Build initial/final atomic systems according to selected process channel.

    Parameters
    ----------
    input_config:
        Resolved run configuration with process selection.

    Returns
    -------
    tuple[AtomicSystem, AtomicSystem]
        Initial and final atomic systems used throughout the workflow.
    """
    initial_atom = AtomicSystem(**input_config.initial_atom_dict)
    if ph.verbose >= 3:
        initial_atom.print()
    else:
        logger.info("Initial atom: %s", initial_atom.name_nice)

    proc = input_config.process.type
    if (proc in [ph.ProcessTypes.TWONEUTRINO_TWOBMINUS, ph.ProcessTypes.NEUTRINOLESS_TWOBMINUS,
                 ph.ProcessTypes.TWONEUTRINO_TWOBPLUS, ph.ProcessTypes.NEUTRINOLESS_TWOBPLUS]):
        final_atom = create_ion(
            initial_atom, z_nuc=initial_atom.Z+ph.PROCESS_IONISATION[proc])
    elif (proc in [ph.ProcessTypes.TWONEUTRINO_BPLUSEC, ph.ProcessTypes.NEUTRINOLESS_BPLUSEC]):
        interm = AtomicSystem(atomic_number=initial_atom.Z+ph.PROCESS_IONISATION[proc],
                              mass_number=initial_atom.mass_number,
                              electron_config="auto")
        final_atom = create_ion(
            interm, z_nuc=interm.Z-1
        )
    else:
        # Double electron capture
        final_atom = AtomicSystem(atomic_number=initial_atom.Z-2,
                                  mass_number=initial_atom.mass_number,
                                  electron_config="auto")

    if ph.verbose >= 3:
        final_atom.print()
    else:
        logger.info("Final atom: %s", final_atom.name_nice)
    return (initial_atom, final_atom)


def find_wavefunctions(input_config: RunConfig, initial_atom: AtomicSystem, final_atom: AtomicSystem):
    """Compute bound/scattering wavefunctions for initial and final atoms.

    Parameters
    ----------
    input_config:
        Run configuration including bound/scattering settings.
    initial_atom:
        Initial atomic system.
    final_atom:
        Final atomic system.

    Returns
    -------
    tuple[WaveFunctionsHandler | None, WaveFunctionsHandler | None]
        Handlers for initial and final systems, depending on enabled tasks.
    """
    wf_handler_init = None
    if input_config.bound_config != None:
        wf_handler_init = WaveFunctionsHandler(
            initial_atom, input_config.bound_config)
        wf_handler_init.find_all_wavefunctions()

    wf_handler_final = None
    if (input_config.scattering_config != None) and (final_atom != None) and (input_config.process.type != ph.ProcessTypes.TWONEUTRINO_TWOEC):
        wf_handler_final = WaveFunctionsHandler(
            final_atom, input_config.bound_config, input_config.scattering_config
        )
        wf_handler_final.find_all_wavefunctions()

    return (wf_handler_init, wf_handler_final)


def find_wave_functions(input_config: RunConfig, initial_atom: AtomicSystem, final_atom: AtomicSystem):
    """Backward-compatible alias for :func:`find_wavefunctions`.

    Parameters
    ----------
    input_config, initial_atom, final_atom:
        Forwarded unchanged to :func:`find_wavefunctions`.

    Returns
    -------
    tuple[WaveFunctionsHandler | None, WaveFunctionsHandler | None]
        Result returned by :func:`find_wavefunctions`.
    """
    return find_wavefunctions(input_config, initial_atom, final_atom)


def build_exchange_correction(wf_handler_init: WaveFunctionsHandler, wf_handler_final: WaveFunctionsHandler, nuclear_radius: float):
    """Apply exchange-driven orthogonalization to final-state scattering functions.

    Parameters
    ----------
    wf_handler_init:
        Initial-state handler with bound orbitals.
    wf_handler_final:
        Final-state handler with scattering states.
    nuclear_radius:
        Nuclear radius in fm.

    Returns
    -------
    exchange.ExchangeCorrection
        Exchange correction object used to transform scattering wavefunctions.
    """
    logger.info("Computing exchange correction")
    ex_corr = exchange.ExchangeCorrection(wf_handler_init,
                                          wf_handler_final,
                                          nuclear_radius)
    start_time = time.time()
    p_new, q_new = ex_corr.transform_scattering_wavefunctions()
    wf_handler_final.scattering_handler.p_grid = p_new
    wf_handler_final.scattering_handler.q_grid = q_new
    # ex_corr.compute_eta_total()
    stop_time = time.time()
    logger.info("Exchange correction took %.2f seconds", stop_time-start_time)
    return ex_corr


def build_energy_grids(input_config: RunConfig):
    """Construct 1D and optional transformed 2D energy grids.

    Parameters
    ----------
    input_config:
        Run configuration containing grid controls.

    Returns
    -------
    tuple[np.ndarray, np.ndarray | None, np.ndarray | None]
        1D kinetic-energy grid and optional 2D ``e1/e2`` meshgrids.
    """
    # always build 1D grid, we need it for integration
    if (input_config.spectra_config.energy_grid_type == "lin"):
        energy_grid_1D = np.linspace(input_config.spectra_config.min_ke,
                                     input_config.spectra_config.total_ke-input_config.spectra_config.min_ke,
                                     input_config.spectra_config.n_ke_points)
    elif (input_config.spectra_config.energy_grid_type == "log"):
        energy_grid_1D = np.logspace(
            np.log10(input_config.spectra_config.min_ke),
            np.log10(input_config.spectra_config.total_ke -
                     input_config.spectra_config.min_ke),
            input_config.spectra_config.n_ke_points
        )
    else:
        raise ValueError("Could not build 1D energy grid")

    e1_grid_2D = None
    e2_grid_2D = None
    if (input_config.spectra_config.compute_2d):
        e1_log = np.logspace(
            np.log10(input_config.spectra_config.min_ke),
            np.log10(input_config.spectra_config.e_max_log_2d),
            input_config.spectra_config.n_points_log_2d
        )
        e1_lin = np.linspace(
            input_config.spectra_config.e_max_log_2d,
            input_config.spectra_config.total_ke-input_config.spectra_config.min_ke,
            input_config.spectra_config.n_points_lin_2d
        )
        e1_grid_tmp = np.concatenate((e1_log, e1_lin[1:]))
        e2_grid_tmp = e1_grid_tmp.copy()
        e1_grid_2D, e2_grid_2D = np.meshgrid(
            e1_grid_tmp, e2_grid_tmp, indexing="ij")
    return energy_grid_1D, e1_grid_2D, e2_grid_2D


def create_fermi_functions(ff_type: int, input_config: RunConfig, wf_handler_final: WaveFunctionsHandler | None, final_atom: AtomicSystem | None,
                           energy_grid_1D: np.ndarray | None):
    """Instantiate a concrete Fermi-function backend from configuration.

    Parameters
    ----------
    ff_type:
        Fermi-function backend selector from :class:`spades.ph.FermiFunctionTypes`.
    input_config:
        Run configuration.
    wf_handler_final:
        Final-state wavefunction handler, required for numeric Fermi functions.
    final_atom:
        Final atom, required for analytical Fermi-function models.
    energy_grid_1D:
        Kinetic-energy grid used by spline-based point-like functions.

    Returns
    -------
    fermi_functions.FermiFunctions
        Concrete Fermi-function backend instance.
    """
    if ff_type == ph.FermiFunctionTypes.NUMERIC_FERMIFUNCTIONS:
        if wf_handler_final == None:
            raise ValueError(
                "No configuration for numeric fermi functions. Check input")
        else:
            # TODO: add density function
            return fermi_functions.Numeric(wf_handler_final.scattering_handler, input_config.spectra_config.nuclear_radius)
    elif ff_type == ph.FermiFunctionTypes.POINTLIKE_FERMIFUNCTIONS:
        if final_atom == None:
            raise ValueError(
                "Cannot build pointlike fermi functions without final atom")
        return fermi_functions.PointLike(final_atom.Z, input_config.spectra_config.nuclear_radius, energy_grid_1D)
    elif ff_type == ph.FermiFunctionTypes.CHARGEDSPHERE_FERMIFUNCTIONS:
        if final_atom == None:
            raise ValueError(
                "Cannot build charged sphere fermi functions without final atom"
            )
        return fermi_functions.ChargedSphere(final_atom.Z, input_config.spectra_config.nuclear_radius)
    else:
        raise ValueError(
            f"Cannot interpret fermi_functions option {ff_type}")


def compute_two_ec_psfs(input_config: RunConfig, wf_handler_init: WaveFunctionsHandler):
    """Placeholder for double-electron-capture PSF pipeline.

    Parameters
    ----------
    input_config:
        Run configuration.
    wf_handler_init:
        Initial-state wavefunction handler.
    """
    pass


def create_spectrum(input_config: RunConfig, fermi_functions: fermi_functions.FermiFunctions | None, eta_total: Callable | None,
                    final_atom: AtomicSystem, wf_handler_init: WaveFunctionsHandler | None, e1_grid_2D: np.ndarray | None = None, e2_grid_2D: np.ndarray | None = None) -> dict[str, SpectrumBase] | SpectrumBase:
    """Create configured spectrum object(s) for the selected decay channel.

    Parameters
    ----------
    input_config:
        Run configuration with process/method selections.
    fermi_functions:
        Selected Fermi-function backend, when required by process type.
    eta_total:
        Optional combined correction function.
    final_atom:
        Final atomic system.
    wf_handler_init:
        Initial-state wavefunction handler, required for capture channels.
    e1_grid_2D, e2_grid_2D:
        Optional 2D grids for computing differential 2D spectra.

    Returns
    -------
    SpectrumBase | dict
        One spectrum object or a dictionary of order-resolved spectrum objects.
    """
    return create_spectrum_from_factory(
        input_config=input_config,
        fermi_func=fermi_functions,
        eta_total=eta_total,
        final_atom=final_atom,
        wf_handler_init=wf_handler_init,
        e1_grid_2D=e1_grid_2D,
        e2_grid_2D=e2_grid_2D,
    )


def _run_spectrum_calculation(
    spectrum_obj: SpectrumBase,
    input_config: RunConfig,
) -> None:
    """Execute compute/integrate/psf lifecycle for a single spectrum object.

    Parameters
    ----------
    spectrum_obj:
        Spectrum object to run.
    input_config:
        Run configuration controlling selected spectrum types and options.
    """
    for sp_type in input_config.spectra_config.types:
        spectrum_obj.compute_spectrum(sp_type)
        if input_config.spectra_config.compute_2d:
            spectrum_obj.compute_2D_spectrum(sp_type)

    if input_config.process.type not in [
        ph.ProcessTypes.TWONEUTRINO_TWOEC,
        ph.ProcessTypes.NEUTRINOLESS_BPLUSEC,
    ]:
        spectrum_obj.integrate_spectrum()
    spectrum_obj.compute_psf()


def compute_spectra_and_psfs(input_config: RunConfig,
                             wf_handler_init: WaveFunctionsHandler | None,
                             wf_handler_final: WaveFunctionsHandler | None,
                             eta_total: Callable | None,
                             final_atom: AtomicSystem,
                             energy_grid_1D: np.ndarray,
                             e1_grid_2D: np.ndarray | None = None,
                             e2_grid_2D: np.ndarray | None = None):
    """Compute requested spectra and PSFs for all selected Fermi-function backends.

    Parameters
    ----------
    input_config:
        Resolved run configuration.
    wf_handler_init, wf_handler_final:
        Wavefunction handlers for initial/final systems.
    eta_total:
        Optional combined correction factor callable.
    final_atom:
        Final atomic system.
    energy_grid_1D:
        Kinetic-energy grid for 1D spectra/Fermi functions.
    e1_grid_2D, e2_grid_2D:
        Optional 2D grids for differential spectra.

    Returns
    -------
    dict
        Mapping from Fermi-function labels to spectrum object(s) with computed PSFs.
    """
    spectra = {}
    for ff_type in input_config.spectra_config.fermi_function_types:
        fermi_functions = None
        if input_config.process.type != ph.ProcessTypes.TWONEUTRINO_TWOEC:
            fermi_functions = create_fermi_functions(ff_type,
                                                     input_config,
                                                     wf_handler_final,
                                                     final_atom,
                                                     energy_grid_1D)

        spectrum = create_spectrum(input_config,
                                   fermi_functions,
                                   eta_total, final_atom,
                                   wf_handler_init,
                                   e1_grid_2D=e1_grid_2D,
                                   e2_grid_2D=e2_grid_2D)
        if isinstance(spectrum, SpectrumBase):
            _run_spectrum_calculation(spectrum, input_config)

            logger.info("Computed PSFs for %s", ph.FERMIFUNCTIONS_MAP_REV[ff_type])
            spectra[ph.FERMIFUNCTIONS_MAP_REV[ff_type]] = spectrum
        elif isinstance(spectrum, dict):
            spectra[ph.FERMIFUNCTIONS_MAP_REV[ff_type]] = {}
            for key in spectrum:
                _run_spectrum_calculation(spectrum[key], input_config)

                logger.info("Computed PSFs for %s order %s", ph.FERMIFUNCTIONS_MAP_REV[ff_type], key)
                spectra[ph.FERMIFUNCTIONS_MAP_REV[ff_type]] = spectrum
    return spectra


def build_corrections(input_config: RunConfig, wf_handler_init: WaveFunctionsHandler | None, wf_handler_final: WaveFunctionsHandler | None):
    """Build combined correction function applied to spectra kernels.

    Parameters
    ----------
    input_config:
        Run configuration with enabled corrections.
    wf_handler_init, wf_handler_final:
        Wavefunction handlers used by exchange/radiative corrections.

    Returns
    -------
    Callable | None
        Interpolated correction function ``eta_total(ke)`` or ``None`` if disabled.
    """
    eta_total = None
    e_values = None
    if (ph.CorrectionTypes.EXCHANGE_CORRECTION in input_config.spectra_config.corrections) and (wf_handler_init != None) and (wf_handler_final != None):
        build_exchange_correction(
            wf_handler_init, wf_handler_final, input_config.spectra_config.nuclear_radius
        )

    if (ph.CorrectionTypes.RADIATIVE_CORRECTION in input_config.spectra_config.corrections) and (wf_handler_final != None) and (wf_handler_final.scattering_handler != None):
        e_values = ph.electron_mass + \
            wf_handler_final.scattering_handler.energy_grid
        rad_cor = np.ones_like(e_values)
        for i_e in range(len(e_values)-1):
            rad_cor[i_e] = math_stuff.r_radiative(e_values[i_e], e_values[-1])
            logger.debug("Radiative correction at E=%s: %s", e_values[i_e], rad_cor[i_e])

        if (eta_total is None):
            eta_total = rad_cor
        else:
            eta_total = eta_total*rad_cor

    if (e_values is not None):
        eta_total = interpolate.CubicSpline(
            e_values-ph.electron_mass, eta_total)

    return eta_total


def _write_outputs(input_config: RunConfig, spectra_psfs: dict[str, SpectrumBase | dict]) -> None:
    """Write spectra/PSF outputs according to output configuration.

    Parameters
    ----------
    input_config:
        Run configuration with output settings.
    spectra_psfs:
        Computed spectrum objects grouped by Fermi-function type.
    """
    if input_config.output_config is None:
        logger.info("Did not receive output configuration. Skip writing files.")
        return

    output_dir_name = input_config.output_config.location
    try:
        os.mkdir(output_dir_name)
        logger.info("Directory %s created successfully.", output_dir_name)
    except FileExistsError:
        logger.warning("Directory %s already exists. Contents will be overwritten.", output_dir_name)
    except Exception as e:
        logger.error("An error occurred while preparing output directory: %s", e)
        return

    write_spectra = getattr(input_config.output_config, "spectra", False)
    write_psfs = getattr(input_config.output_config, "psfs", False)
    logger.debug("Output config: %s", input_config.output_config)
    if not (write_spectra or write_psfs):
        logger.info("Skipping spectra/PSF writing by configuration.")
        return

    spectrum_writer = SpectrumWriter(write_spectra=write_spectra,
                                     write_psfs=write_psfs)
    logger.info("Writing output file")
    for key in spectra_psfs:
        if isinstance(spectra_psfs[key], SpectrumBase):
            spectrum_writer.add_spectrum(spectra_psfs[key], key)
        elif isinstance(spectra_psfs[key], dict):
            for key1 in spectra_psfs[key]:
                spectrum_writer.add_spectrum(
                    spectra_psfs[key][key1], f"{key}_{key1}")
    spectrum_writer.write(f"{output_dir_name}/spectra.json")


def main(argv=None):
    """Run end-to-end CLI workflow: parse input, compute spectra/PSFs, write outputs.

    Parameters
    ----------
    argv:
        Optional CLI argument list. If ``None``, ``sys.argv`` is used.
    """
    input_config, initial_atom, final_atom = parse_input(argv)

    wf_handler_init, wf_handler_final = find_wavefunctions(
        input_config, initial_atom, final_atom)

    energy_grid_1D, e1_grid_2D, e2_grid_2D = build_energy_grids(input_config)

    eta_total = build_corrections(
        input_config, wf_handler_init, wf_handler_final)

    # compute the spectra and psfs
    spectra_psfs = compute_spectra_and_psfs(input_config=input_config,
                                            wf_handler_init=wf_handler_init,
                                            wf_handler_final=wf_handler_final,
                                            eta_total=eta_total,
                                            final_atom=final_atom,
                                            energy_grid_1D=energy_grid_1D,
                                            e1_grid_2D=e1_grid_2D,
                                            e2_grid_2D=e2_grid_2D)
    _write_outputs(input_config, spectra_psfs)


if __name__ == "__main__":
    main()

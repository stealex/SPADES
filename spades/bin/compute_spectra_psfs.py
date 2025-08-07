#!/usr/bin/env python

from multiprocessing import Value, process
from typing import Callable, Type, final
from matplotlib import pyplot as plt
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
import spades.spectra.twobeta
import spades.spectra.ecbeta
import spades.spectra.twoec
from spades.wavefunctions import WaveFunctionsHandler
from spades.spectra.spectrum_writer import SpectrumWriter

logger = logging.getLogger(__name__)


def parse_input():
    print("Parsing input arguments")
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

    args = parser.parse_args()
    ph.verbose = args.verbose
    logging.basicConfig(level=10*(5-ph.verbose))
    # logging.basicConfig(level=logging.DEBUG)
    ph.user_distance_unit_name = args.distance_unit
    ph.user_energy_unit_name = args.energy_unit
    ph.user_distance_unit = ph.__dict__[
        args.distance_unit]
    ph.user_energy_unit = ph.__dict__[args.energy_unit]
    ph.delta_m_files = args.qvalues_file

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
    initial_atom = AtomicSystem(**input_config.initial_atom_dict)
    initial_atom.print()

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

    final_atom.print()
    return (initial_atom, final_atom)


def find_wave_functions(input_config: RunConfig, initial_atom: AtomicSystem, final_atom: AtomicSystem):
    wf_handler_init = None
    if input_config.bound_config != None:
        wf_handler_init = WaveFunctionsHandler(
            initial_atom, input_config.bound_config)
        wf_handler_init.find_all_wavefunctions()

    wf_handler_final = None
    if (input_config.scattering_config != None) and (final_atom != None):
        wf_handler_final = WaveFunctionsHandler(
            final_atom, input_config.bound_config, input_config.scattering_config
        )
        wf_handler_final.find_all_wavefunctions()

    return (wf_handler_init, wf_handler_final)


def build_exchange_correction(wf_handler_init: WaveFunctionsHandler, wf_handler_final: WaveFunctionsHandler, nuclear_radius: float):
    print(f"Computing exchange correction")
    ex_corr = exchange.ExchangeCorrection(wf_handler_init,
                                          wf_handler_final,
                                          nuclear_radius)
    start_time = time.time()
    p_new, q_new = ex_corr.transform_scattering_wavefunctions()
    wf_handler_final.scattering_handler.p_grid = p_new
    wf_handler_final.scattering_handler.q_grid = q_new
    # ex_corr.compute_eta_total()
    stop_time = time.time()
    print(f"... took {stop_time-start_time: .2f} seconds")
    return ex_corr


def build_energy_grids(input_config: RunConfig):
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
    pass


def create_spectrum(input_config: RunConfig, fermi_functions: fermi_functions.FermiFunctions, eta_total: Callable | None,
                    final_atom: AtomicSystem, wf_handler_init: WaveFunctionsHandler | None, e1_grid_2D: np.ndarray | None = None, e2_grid_2D: np.ndarray | None = None) -> dict[str, SpectrumBase] | SpectrumBase:
    # prepare for all options
    atilde = 1.12*(final_atom.mass_number**0.5)
    if ("enei" in input_config.spectra_config.method):
        if (isinstance(input_config.spectra_config.method["enei"], float)):
            enei = input_config.spectra_config.method["enei"]
        elif isinstance(input_config.spectra_config.method["enei"], str):
            if (input_config.spectra_config.method["enei"] != "auto"):
                raise ValueError("Cannot interpret enei option")
            enei = atilde - 0.5 * input_config.spectra_config.ei_ef

    if ("orders" in input_config.spectra_config.method):
        if (isinstance(input_config.spectra_config.method["orders"], list)):
            orders = []
            for key in input_config.spectra_config.method["orders"]:
                orders.append(ph.TAYLOR_ORDER_NAMES_MAP[str(key)])
        elif (isinstance(input_config.spectra_config.method["orders"], str)):
            if (input_config.spectra_config.method["orders"] != "auto"):
                raise ValueError("Cannot interpret orders option")
            if input_config.process.transition == ph.TransitionTypes.ZEROPLUS_TO_TWOPLUS:
                orders = [ph.TaylorOrders.TWOTWO, ph.TaylorOrders.SIX]
            else:
                orders = [ph.TaylorOrders.ZERO, ph.TaylorOrders.TWO,
                          ph.TaylorOrders.TWOTWO, ph.TaylorOrders.FOUR]

    if (input_config.process.type in [ph.ProcessTypes.TWONEUTRINO_TWOBMINUS, ph.ProcessTypes.TWONEUTRINO_TWOBPLUS]):
        if (input_config.spectra_config.method["name"] == "Closure"):
            print("EN-EI=", enei, " atilde=", atilde, " totalKE=", input_config.spectra_config.total_ke,
                  " R=", input_config.spectra_config.nuclear_radius)
            return spades.spectra.twobeta.ClosureSpectrum2nu(total_ke=input_config.spectra_config.total_ke,
                                                             ei_ef=input_config.spectra_config.ei_ef,
                                                             enei=enei,
                                                             fermi_functions=fermi_functions,
                                                             eta_total=eta_total,
                                                             transition=input_config.process.transition,
                                                             min_ke=input_config.spectra_config.min_ke,
                                                             n_ke_points=input_config.spectra_config.n_ke_points,
                                                             energy_grid_type=input_config.spectra_config.energy_grid_type,
                                                             e1_grid_2D=e1_grid_2D,
                                                             e2_grid_2D=e2_grid_2D)

        elif (input_config.spectra_config.method["name"] == "Taylor"):
            print("Computing with Taylor")
            spectra = {}
            for ord in orders:
                spectra[ord] = spades.spectra.twobeta.TaylorSpectrum2nu(total_ke=input_config.spectra_config.total_ke,
                                                                        ei_ef=input_config.spectra_config.ei_ef,
                                                                        fermi_functions=fermi_functions,
                                                                        eta_total=eta_total,
                                                                        taylor_order=ord,
                                                                        transition=input_config.process.transition,
                                                                        min_ke=input_config.spectra_config.min_ke,
                                                                        n_ke_points=input_config.spectra_config.n_ke_points,
                                                                        energy_grid_type=input_config.spectra_config.energy_grid_type,
                                                                        e1_grid_2D=e1_grid_2D,
                                                                        e2_grid_2D=e2_grid_2D)

            return spectra

        else:
            raise NotImplementedError()
    elif (input_config.process.type in [ph.ProcessTypes.NEUTRINOLESS_TWOBMINUS, ph.ProcessTypes.NEUTRINOLESS_TWOBPLUS]):
        if (input_config.spectra_config.method["name"] == "Closure"):
            if (input_config.process.mechanism == ph.NeutrinoLessModes.LIGHT_NEUTRINO_EXCHANGE):
                return spades.spectra.twobeta.ClosureSpectrum0nu_LNE(total_ke=input_config.spectra_config.total_ke,
                                                                     ei_ef=input_config.spectra_config.ei_ef,
                                                                     nuclear_radius=input_config.spectra_config.nuclear_radius,
                                                                     fermi_functions=fermi_functions,
                                                                     eta_total=eta_total,
                                                                     min_ke=input_config.spectra_config.min_ke,
                                                                     n_ke_points=input_config.spectra_config.n_ke_points,
                                                                     energy_grid_type=input_config.spectra_config.energy_grid_type)
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

    elif (input_config.process.type == ph.ProcessTypes.TWONEUTRINO_BPLUSEC):
        if (wf_handler_init is None):
            raise ValueError(
                "Received None for wf_handler_init. Cannot compute EC withouot it.")

        if input_config.spectra_config.method["name"] == "Closure":
            print(enei, atilde, input_config.spectra_config.total_ke,
                  input_config.spectra_config.nuclear_radius)
            return spades.spectra.ecbeta.ClosureSpectrum2nu(total_ke=input_config.spectra_config.total_ke,
                                                            ei_ef=input_config.spectra_config.ei_ef,
                                                            fermi_functions=fermi_functions,
                                                            bound_handler=wf_handler_init.bound_handler,
                                                            nuclear_radius=input_config.spectra_config.nuclear_radius,
                                                            enei=enei,
                                                            transition_type=input_config.process.transition,
                                                            min_ke=input_config.spectra_config.min_ke,
                                                            n_ke_points=input_config.spectra_config.n_ke_points,
                                                            energy_grid_type=input_config.spectra_config.energy_grid_type,
                                                            e1_grid_2D=e1_grid_2D,
                                                            e2_grid_2D=e2_grid_2D)
        else:
            raise NotImplementedError()
    elif (input_config.process.type == ph.ProcessTypes.NEUTRINOLESS_BPLUSEC):
        if (wf_handler_init is None):
            raise ValueError(
                "Received None for wf_handler_init. Cannot compute EC withouot it.")
        if input_config.spectra_config.method["name"] == "Closure":
            print(enei, atilde, input_config.spectra_config.total_ke,
                  input_config.spectra_config.nuclear_radius)
            return spades.spectra.ecbeta.ClosureSpectrum0nu_LNE(total_ke=input_config.spectra_config.total_ke,
                                                                ei_ef=input_config.spectra_config.ei_ef,
                                                                fermi_functions=fermi_functions,
                                                                bound_handler=wf_handler_init.bound_handler,
                                                                nuclear_radius=input_config.spectra_config.nuclear_radius,
                                                                enei=enei)
        else:
            raise NotImplementedError()

    elif (input_config.process.type == ph.ProcessTypes.TWONEUTRINO_TWOEC):
        if (wf_handler_init is None):
            raise ValueError(
                "Received None for wf_handler_init. Cannot compute EC withouot it.")
        if input_config.spectra_config.method["name"] == "Closure":
            print(enei, atilde, input_config.spectra_config.total_ke,
                  input_config.spectra_config.nuclear_radius)

            return spades.spectra.twoec.TwoECSpectrumClosure(total_ke=input_config.spectra_config.total_ke,
                                                             ei_ef=input_config.spectra_config.ei_ef,
                                                             bound_handler=wf_handler_init.bound_handler,
                                                             nuclear_radius=input_config.spectra_config.nuclear_radius,
                                                             enei=enei,
                                                             transition_type=input_config.process.transition)
        elif input_config.spectra_config.method["name"] == "Taylor":
            spectra = {}
            for ord in orders:
                spectra[ord] = spades.spectra.twoec.TwoECSpectrumTaylor(total_ke=input_config.spectra_config.total_ke,
                                                                        ei_ef=input_config.spectra_config.ei_ef,
                                                                        bound_handler=wf_handler_init.bound_handler,
                                                                        nuclear_radius=input_config.spectra_config.nuclear_radius,
                                                                        transition_type=input_config.process.transition,
                                                                        order=ord)
            return spectra
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError()


def compute_spectra_and_psfs(input_config: RunConfig,
                             wf_handler_init: WaveFunctionsHandler | None,
                             wf_handler_final: WaveFunctionsHandler | None,
                             eta_total: Callable | None,
                             final_atom: AtomicSystem,
                             energy_grid_1D: np.ndarray,
                             e1_grid_2D: np.ndarray | None = None,
                             e2_grid_2D: np.ndarray | None = None):
    spectra = {}
    for ff_type in input_config.spectra_config.fermi_function_types:
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
            for sp_type in input_config.spectra_config.types:
                spectrum.compute_spectrum(sp_type)
                if (input_config.spectra_config.compute_2d):
                    spectrum.compute_2D_spectrum(sp_type)

            if not (input_config.process.type in [ph.ProcessTypes.TWONEUTRINO_TWOEC, ph.ProcessTypes.NEUTRINOLESS_BPLUSEC]):
                spectrum.integrate_spectrum()
            spectrum.compute_psf()

            print("These are the PSFS ", spectrum.psfs)
            spectra[ph.FERMIFUNCTIONS_MAP_REV[ff_type]] = spectrum
        elif isinstance(spectrum, dict):
            print("Doing this")
            spectra[ph.FERMIFUNCTIONS_MAP_REV[ff_type]] = {}
            for key in spectrum:
                for sp_type in input_config.spectra_config.types:
                    spectrum[key].compute_spectrum(sp_type)
                    if (input_config.spectra_config.compute_2d):
                        spectrum[key].compute_2D_spectrum(sp_type)

                if not (input_config.process.type in [ph.ProcessTypes.TWONEUTRINO_TWOEC, ph.ProcessTypes.NEUTRINOLESS_BPLUSEC]):
                    spectrum[key].integrate_spectrum()
                spectrum[key].compute_psf()

                print(f"These are the PSFs for {key} ", spectrum[key].psfs)
                spectra[ph.FERMIFUNCTIONS_MAP_REV[ff_type]] = spectrum
    return spectra


def build_corrections(input_config: RunConfig, wf_handler_init: WaveFunctionsHandler | None, wf_handler_final: WaveFunctionsHandler | None):
    eta_total = None
    e_values = None
    if (ph.CorrectionTypes.EXCHANGE_CORRECTION in input_config.spectra_config.corrections) and (wf_handler_init != None) and (wf_handler_final != None):
        exchange_correction = build_exchange_correction(
            wf_handler_init, wf_handler_final, input_config.spectra_config.nuclear_radius
        )
        # e_values = ph.electron_mass + \
        #     wf_handler_final.scattering_handler.energy_grid
        # eta_total = 1.+exchange_correction.eta_total
        # print(eta_total)

    if (ph.CorrectionTypes.RADIATIVE_CORRECTION in input_config.spectra_config.corrections) and (wf_handler_final != None) and (wf_handler_final.scattering_handler != None):
        e_values = ph.electron_mass + \
            wf_handler_final.scattering_handler.energy_grid
        rad_cor = np.ones_like(e_values)
        for i_e in range(len(e_values)-1):
            rad_cor[i_e] = math_stuff.r_radiative(e_values[i_e], e_values[-1])
            print(e_values[i_e], rad_cor[i_e])

        if (eta_total is None):
            eta_total = rad_cor
        else:
            eta_total = eta_total*rad_cor

    if (e_values is not None):
        eta_total = interpolate.CubicSpline(
            e_values-ph.electron_mass, eta_total)

    return eta_total


def main(argv=None):
    input_config, initial_atom, final_atom = parse_input()

    wf_handler_init, wf_handler_final = find_wave_functions(
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
    # write the output
    if (input_config.output_config is None):
        print("Did not receive output configuration. Skip writting files...")
    else:
        output_dir_name = input_config.output_config.location
        try:
            os.mkdir(output_dir_name)
            print(f"Directory {output_dir_name} created successfully.")
        except FileExistsError:
            print(
                f"Directory {output_dir_name} already exists. Contents will be overwritten")
        except Exception as e:
            print(f"An error occured {e}")
            return

        write_spectra = getattr(input_config.output_config, "spectra", False)
        write_psfs = getattr(input_config.output_config, "psfs", False)
        print(input_config.output_config)
        if (write_spectra or write_psfs):
            spectrum_writer = SpectrumWriter(write_spectra=write_spectra,
                                             write_psfs=write_psfs)
            if spectra_psfs is None:
                raise NotImplementedError
            else:
                print("Writing output file")
                for key in spectra_psfs:
                    if isinstance(spectra_psfs[key], SpectrumBase):
                        spectrum_writer.add_spectrum(spectra_psfs[key], key)
                    elif isinstance(spectra_psfs[key], dict):
                        for key1 in spectra_psfs[key]:
                            spectrum_writer.add_spectrum(
                                spectra_psfs[key][key1], f"{key}_{key1}")
                spectrum_writer.write(f"{output_dir_name}/spectra.json")
        else:
            print("Will not write spectra and psfs")


if __name__ == "__main__":
    main()

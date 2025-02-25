#!/usr/bin/env python

from multiprocessing import Value, process
from typing import Callable, Type, final
from matplotlib import pyplot as plt
import yaml
from argparse import ArgumentParser
import logging
import time
import numpy as np

from spades import fermi_functions, ph, exchange
from spades.config import RunConfig
from spades.dhfs import AtomicSystem, create_ion
from spades.spectra.base import BetaSpectrumBase, SpectrumBase
import spades.spectra.twobeta
import spades.spectra.ecbeta
import spades.spectra.twoec
from spades.wavefunctions import WaveFunctionsHandler

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
                        default=ph.q_values_file)
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
    ph.q_values_file = args.qvalues_file

    with open(args.config_file, 'r') as f:
        conf = yaml.safe_load(f)

    input_config = RunConfig(conf)
    initial_atom, final_atom = create_atoms(input_config)
    input_config.resolve_nuclear_radius(initial_atom)
    input_config.resolve_q_value(initial_atom, final_atom)
    input_config.create_spectra_config()
    input_config.create_scattering_config()
    input_config.resolve_bound_config(initial_atom)
    input_config.create_bound_config()

    return (input_config, initial_atom, final_atom)


def create_atoms(input_config: RunConfig):
    initial_atom = AtomicSystem(**input_config.initial_atom_dict)
    initial_atom.print()

    proc = input_config.process.type
    if (proc in [ph.TWONEUTRINO_TWOBMINUS, ph.NEUTRINOLESS_TWOBMINUS,
                 ph.TWONEUTRINO_TWOBPLUS, ph.NEUTRINOLESS_TWOBPLUS]):
        final_atom = create_ion(
            initial_atom, z_nuc=initial_atom.Z+ph.PROCESS_IONISATION[proc])
    elif (proc in [ph.TWONEUTRINO_BPLUSEC, ph.NEUTRINOLESS_BPLUSEC]):
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


def build_exchange_correction(wf_handler_init: WaveFunctionsHandler, wf_handler_final: WaveFunctionsHandler):
    print(f"Computign exchange correction")
    ex_corr = exchange.ExchangeCorrection(wf_handler_init,
                                          wf_handler_final)
    start_time = time.time()
    ex_corr.compute_eta_total()
    stop_time = time.time()
    print(f"... took {stop_time-start_time: .2f} seconds")
    return ex_corr


def build_energy_grids(input_config: RunConfig):
    if (input_config.spectra_config.energy_grid_type == "lin"):
        energy_grid_1D = np.linspace(input_config.spectra_config.min_ke,
                                     input_config.spectra_config.q_value-input_config.spectra_config.min_ke,
                                     input_config.spectra_config.n_ke_points)
    elif (input_config.spectra_config.energy_grid_type == "log"):
        energy_grid_1D = np.logspace(
            np.log10(input_config.spectra_config.min_ke),
            np.log10(input_config.spectra_config.q_value -
                     input_config.spectra_config.min_ke),
            input_config.spectra_config.n_ke_points
        )
    else:
        raise ValueError("Could not build 1D energy grid")

    return energy_grid_1D


def create_fermi_functions(ff_type: int, input_config: RunConfig, wf_handler_final: WaveFunctionsHandler | None, final_atom: AtomicSystem | None,
                           energy_grid_1D: np.ndarray | None):
    if ff_type == ph.NUMERIC_FERMIFUNCTIONS:
        if wf_handler_final == None:
            raise ValueError(
                "No configuration for numeric fermi functions. Check input")
        else:
            # TODO: add density function
            return fermi_functions.Numeric(wf_handler_final.scattering_handler, input_config.spectra_config.nuclear_radius)
    elif ff_type == ph.POINTLIKE_FERMIFUNCTIONS:
        if final_atom == None:
            raise ValueError(
                "Cannot build pointlike fermi functions without final atom")
        return fermi_functions.PointLike(final_atom.Z, input_config.spectra_config.nuclear_radius, energy_grid_1D)
    elif ff_type == ph.CHARGEDSPHERE_FERMIFUNCTIONS:
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
                    final_atom: AtomicSystem, wf_handler_init: WaveFunctionsHandler | None) -> SpectrumBase:
    if (input_config.process.type in [ph.TWONEUTRINO_TWOBMINUS, ph.TWONEUTRINO_TWOBPLUS]):
        if (input_config.spectra_config.method["name"] == "Closure"):
            atilde = 1.12*(final_atom.mass_number**0.5)
            if ("enei" in input_config.spectra_config.method):
                if (isinstance(input_config.spectra_config.method["enei"], float)):
                    enei = input_config.spectra_config.method["enei"]
                elif isinstance(input_config.spectra_config.method["enei"], str):
                    if (input_config.spectra_config.method["enei"] != "auto"):
                        raise ValueError("Cannot interpret enei option")
                    enei = atilde - 0.5 * \
                        (input_config.spectra_config.q_value + 2.0*ph.electron_mass)

            print(enei, atilde, input_config.spectra_config.q_value,
                  input_config.spectra_config.nuclear_radius)
            return spades.spectra.twobeta.ClosureSpectrum2nu(q_value=input_config.spectra_config.q_value,
                                                             enei=enei,
                                                             fermi_functions=fermi_functions,
                                                             eta_total=eta_total,
                                                             transition=input_config.process.transition,
                                                             min_ke=input_config.spectra_config.min_ke,
                                                             n_ke_points=input_config.spectra_config.n_ke_points,
                                                             energy_grid_type=input_config.spectra_config.energy_grid_type)
        else:
            raise NotImplementedError()
    elif (input_config.process.type in [ph.NEUTRINOLESS_TWOBMINUS, ph.NEUTRINOLESS_TWOBPLUS]):
        if (input_config.spectra_config.method["name"] == "Closure"):
            if (input_config.process.mechanism == ph.LIGHT_NEUTRINO_EXCHANGE):
                return spades.spectra.twobeta.ClosureSpectrum0nu_LNE(q_value=input_config.spectra_config.q_value,
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

    elif (input_config.process.type == ph.TWONEUTRINO_BPLUSEC):
        if input_config.spectra_config.method["name"] == "Closure":
            atilde = 1.12*(final_atom.mass_number**0.5)
            if ("enei" in input_config.spectra_config.method):
                if (isinstance(input_config.spectra_config.method["enei"], float)):
                    enei = input_config.spectra_config.method["enei"]
                elif isinstance(input_config.spectra_config.method["enei"], str):
                    if (input_config.spectra_config.method["enei"] != "auto"):
                        raise ValueError("Cannot interpret enei option")
                    enei = atilde - 0.5 * \
                        (input_config.spectra_config.q_value)

            print(enei, atilde, input_config.spectra_config.q_value,
                  input_config.spectra_config.nuclear_radius)
            return spades.spectra.ecbeta.ClosureSpectrum2nu(q_value=input_config.spectra_config.q_value,
                                                            fermi_functions=fermi_functions,
                                                            bound_handler=wf_handler_init.bound_handler,
                                                            nuclear_radius=input_config.spectra_config.nuclear_radius,
                                                            enei=enei,
                                                            min_ke=input_config.spectra_config.min_ke,
                                                            n_ke_points=input_config.spectra_config.n_ke_points,
                                                            energy_grid_type=input_config.spectra_config.energy_grid_type)

    elif (input_config.process.type == ph.TWONEUTRINO_TWOEC):
        if input_config.spectra_config.method["name"] == "Closure":
            atilde = 1.12*(final_atom.mass_number**0.5)
            if ("enei" in input_config.spectra_config.method):
                if (isinstance(input_config.spectra_config.method["enei"], float)):
                    enei = input_config.spectra_config.method["enei"]
                elif isinstance(input_config.spectra_config.method["enei"], str):
                    if (input_config.spectra_config.method["enei"] != "auto"):
                        raise ValueError("Cannot interpret enei option")
                    enei = atilde - 0.5 * \
                        (input_config.spectra_config.q_value-2.0*ph.electron_mass)

            print(enei, atilde, input_config.spectra_config.q_value,
                  input_config.spectra_config.nuclear_radius)

            return spades.spectra.twoec.TwoECSpectrumClosure(q_value=input_config.spectra_config.q_value,
                                                             bound_handler=wf_handler_init.bound_handler,
                                                             nuclear_radius=input_config.spectra_config.nuclear_radius,
                                                             enei=enei)
        else:
            raise NotImplementedError()

    else:
        raise NotImplementedError()


def compute_spectra_and_psfs(input_config: RunConfig, wf_handler_init: WaveFunctionsHandler | None, wf_handler_final: WaveFunctionsHandler | None, eta_total: Callable | None,
                             final_atom: AtomicSystem, energy_grid_1D: np.ndarray):
    for ff_type in input_config.spectra_config.fermi_function_types:
        fermi_functions = create_fermi_functions(ff_type,
                                                 input_config,
                                                 wf_handler_final,
                                                 final_atom,
                                                 energy_grid_1D)

        spectrum = create_spectrum(input_config,
                                   fermi_functions,
                                   eta_total, final_atom,
                                   wf_handler_init)

        for sp_type in input_config.spectra_config.types:
            spectrum.compute_spectrum(sp_type)

        if (input_config.process.type != ph.TWONEUTRINO_TWOEC):
            integrals = spectrum.integrate_spectrum()
            print(f"These are the integrals {integrals}")

            for key in integrals:
                if isinstance(integrals[key], dict):
                    for kk in integrals[key]:
                        print(
                            f"{key} {kk} {spectrum.compute_psf(integrals[key][kk])}")
                else:
                    return
        else:
            print(f"Integrals {spectrum.spectrum_integrals}")
            print(f"PSFs {spectrum.compute_psf()}")


def spectra_and_psfs_factory(input_config: RunConfig, wf_handler_init: WaveFunctionsHandler | None, wf_handler_final: WaveFunctionsHandler | None, eta_total,
                             initial_atom: AtomicSystem | None, final_atom: AtomicSystem, energy_grid_1D: np.ndarray):
    if input_config.process == ph.TWONEUTRINO_TWOEC:
        if (wf_handler_init == None):
            raise ValueError("Logic error for 2nu2EC")
        return compute_two_ec_psfs(input_config, wf_handler_init)
    else:
        return compute_spectra_and_psfs(input_config, wf_handler_init, wf_handler_final, eta_total,
                                        final_atom, energy_grid_1D)


def main(argv=None):
    input_config, initial_atom, final_atom = parse_input()

    wf_handler_init, wf_handler_final = find_wave_functions(
        input_config, initial_atom, final_atom)

    # check if we need to build exchange correction
    eta_total = None
    if (ph.EXCHANGE_CORRECTION in input_config.spectra_config.corrections) and (wf_handler_init != None) and (wf_handler_final != None):
        exchange_correction = build_exchange_correction(
            wf_handler_init, wf_handler_final
        )
        eta_total = exchange_correction.eta_total

    energy_grid_1D = build_energy_grids(input_config)

    spectra_psfs = spectra_and_psfs_factory(
        input_config, wf_handler_init, wf_handler_final, eta_total,
        initial_atom, final_atom, energy_grid_1D)


if __name__ == "__main__":
    main()

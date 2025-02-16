#!/usr/bin/env python
import os
import scipy as sp
import yaml
from argparse import ArgumentParser
from spades import ph, fermi_functions, exchange, spectra
from spades.wavefunctions import wavefunctions_handler, bound_config, scattering_config
from spades.dhfs import AtomicSystem, create_ion
from spades import io_handler
import matplotlib.pyplot as plt
import numpy as np
from spades.spectra import Spectra2DConfig, SpectraConfig
import time


class RunConfig:
    def __init__(self, config: dict) -> None:
        self.task_name = config["task"]
        self.process_name = config["process"]
        self.process = ph.BBTYPES[self.process_name]

        # atoms
        self.initial_atom = AtomicSystem(config["initial_atom"])
        self.final_atom = create_ion(
            self.initial_atom, self.initial_atom.Z + 2)

        # technicals
        method = config["spectra_computation"]["method"]
        wavefunction_eval = config["spectra_computation"]["wavefunction_evaluation"]
        nuclear_radius = config["spectra_computation"]["nuclear_radius"]
        print(type(nuclear_radius))
        if isinstance(nuclear_radius, str):
            print(nuclear_radius)
            if nuclear_radius == "auto":
                nuclear_radius = 1.2*(self.initial_atom.mass_number)**(1./3.)
            else:
                raise ValueError(
                    f"Unknown option {nuclear_radius} for nuclear_radius")
        elif isinstance(nuclear_radius, float):
            if (nuclear_radius < 0):
                raise ValueError("Nuclear radius cannot be < 0")
        else:
            raise ValueError("Cannot interpret nuclear_radius option")

        types = config["spectra_computation"]["types"]
        energy_grid_type = config["spectra_computation"]["energy_grid_type"]
        corrections = config["spectra_computation"]["corrections"]
        fermi_functions = config["spectra_computation"]["fermi_functions"]
        q_value = config["spectra_computation"]["q_value"]
        if q_value == "auto":
            q_values = ph.read_qvalues(ph.q_values_file)
            q_value = q_values[self.initial_atom.name_nice]
        elif (q_value > 0):
            pass
        else:
            raise ValueError("Cannot interpret q_value option")

        min_ke = float(config["spectra_computation"]["min_ke"])
        n_ke_points = -1
        ke_step = -1.
        try:
            n_ke_points = config["spectra_computation"]["n_ke_points"]
        except KeyError:
            ke_step = config["spectra_computation"]["ke_step"]
        if (n_ke_points > 0) & (ke_step > 0):
            raise ValueError(
                "Cannot have both n_ke_points and ke_step specified for spectra_computation")
        if (ke_step > 0) & (energy_grid_type != "lin"):
            raise ValueError(
                "Cannot have ke_step specified for log energy grid in spectra_computation")

        if "bound_states" in config:
            if (config["bound_states"]["n_values"] == "auto"):
                n_values = list(set(self.initial_atom.n_values.tolist()))
            else:
                n_values = config["bound_states"]["n_values"]

            if (config["bound_states"]["k_values"] == "auto"):
                k_values = {}
                for i_s in range(len(self.initial_atom.electron_config)):
                    n = self.initial_atom.n_values[i_s]
                    l = self.initial_atom.l_values[i_s]
                    j = 0.5*self.initial_atom.jj_values[i_s]
                    k = int((l-j)*(2*j+1))

                    if not (self.initial_atom.n_values[i_s] in k_values):
                        k_values[n] = []
                    k_values[n].append(k)
            else:
                k_values = config["bound_states"]["k_values"]
            self.bound_config = bound_config(max_r=config["bound_states"]["max_r"]*ph.user_distance_unit/ph.fm,
                                             n_radial_points=config["bound_states"]["n_radial_points"],
                                             n_values=n_values,
                                             k_values=k_values)
        else:
            self.bound_config = None

        if "scattering_states" in config:
            k_values = config["scattering_states"]["k_values"]
            if (config["scattering_states"]["k_values"] == "auto"):
                k_values = [-1, 1]
            if (config["scattering_states"]["n_ke_points"] == "auto"):
                n_ke_points_scattering = 100
            else:
                n_ke_points_scattering = config["scattering_states"]["n_ke_points"]
            self.scattering_config = scattering_config(max_r=config["scattering_states"]["max_r"]*ph.user_distance_unit/ph.fm,
                                                       n_radial_points=config["scattering_states"]["n_radial_points"],
                                                       min_ke=min_ke * ph.user_energy_unit/ph.MeV,
                                                       max_ke=q_value * ph.user_energy_unit/ph.MeV,
                                                       n_ke_points=n_ke_points_scattering,
                                                       k_values=k_values)
        else:
            self.scattering_config = None

        orders = None
        try:
            orders = config["spectra_computation"]["orders"]
        except KeyError:
            pass

        self.spectra_config = SpectraConfig(method=method, wavefunction_evaluation=wavefunction_eval, nuclear_radius=nuclear_radius,
                                            types=types, energy_grid_type=energy_grid_type, fermi_functions=fermi_functions, q_value=q_value,
                                            min_ke=min_ke, corrections=corrections, n_ke_points=n_ke_points, ke_step=ke_step, orders=orders)

        self.spectra_2d_config = Spectra2DConfig(n_points_log=config["spectra_computation"]["2d_config"]["n_points_log"],
                                                 n_points_lin=config["spectra_computation"]["2d_config"]["n_points_lin"],
                                                 e_max_log=config["spectra_computation"]["2d_config"]["e_max_log"])
        self.output_prefix = config["output"]["file_prefix"]


def main(argv=None):
    '''Command line arguments'''
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
    ph.user_distance_unit_name = args.distance_unit
    ph.user_energy_unit_name = args.energy_unit
    ph.user_distance_unit = ph.__dict__[
        args.distance_unit]
    ph.user_energy_unit = ph.__dict__[args.energy_unit]
    ph.q_values_file = args.qvalues_file

    with open(args.config_file, 'r') as f:
        run_conf = yaml.safe_load(f)

    input_config = RunConfig(run_conf)
    has_bound_state_config = True
    if not (input_config.bound_config is None):
        input_config.bound_config.print()
    else:
        print("No bound states configuration for RADIAL")
        has_bound_state_config = False

    has_scattering_state_config = True
    if not (input_config.scattering_config is None):
        input_config.scattering_config.print()
    else:
        print("No scattering states configuration for RADIAL")
        has_scattering_state_config = False

    if (has_bound_state_config):
        wf_handler_initial = wavefunctions_handler(
            input_config.initial_atom, input_config.bound_config)

        # run dhfs
        print("Computing wavefunctions for initial atom")
        start_time = time.time()
        wf_handler_initial.find_all_wavefunctions()
        stop_time = time.time()
        print(f"... took {stop_time-start_time: .2f} seconds")

    if has_scattering_state_config:
        print("Computing wavefunctions for final atom")

        wf_handler_final = wavefunctions_handler(
            input_config.final_atom, input_config.bound_config, input_config.scattering_config)

        start_time = time.time()
        wf_handler_final.find_all_wavefunctions()
        stop_time = time.time()
        print(f"... took {stop_time-start_time: .2f} seconds")

    if has_bound_state_config & has_scattering_state_config & (ph.EXCHANGE_CORRECTION in input_config.spectra_config.corrections):
        print(f"Computign exchange correction")
        ex_corr = exchange.ExchangeCorrection(wf_handler_initial,
                                              wf_handler_final)
        start_time = time.time()
        ex_corr.compute_eta_total()
        stop_time = time.time()
        eta_total = ex_corr.eta_total_func
        print(f"... took {stop_time-start_time: .2f} seconds")
    else:
        eta_total = None

    if type(input_config.spectra_config.nuclear_radius) == str:
        nuclear_radius = 1.2*(input_config.initial_atom.mass_number**(1./3.))
    elif type(input_config.spectra_config.nuclear_radius) == float:
        nuclear_radius = input_config.spectra_config.nuclear_radius

    if (input_config.spectra_config.energy_grid_type == "lin"):
        if input_config.spectra_config.n_ke_points == -1:
            spectrum_energy_grid = np.arange(
                input_config.spectra_config.min_ke,
                input_config.spectra_config.q_value -
                input_config.spectra_config.min_ke,
                input_config.spectra_config.ke_step)
        else:
            spectrum_energy_grid = np.linspace(input_config.spectra_config.min_ke,
                                               input_config.spectra_config.q_value-input_config.spectra_config.min_ke,
                                               input_config.spectra_config.n_ke_points,
                                               retstep=True)
    elif input_config.spectra_config.energy_grid_type == "log":
        spectrum_energy_grid = np.logspace(
            np.log10(input_config.spectra_config.min_ke),
            np.log10(input_config.spectra_config.q_value -
                     input_config.spectra_config.min_ke),
            input_config.spectra_config.n_ke_points)

    # 2D grid
    spectrum_energy_grid_2d = []
    e1_log = np.logspace(
        np.log10(input_config.spectra_config.min_ke),
        np.log10(input_config.spectra_2d_config.e_max_log),
        input_config.spectra_2d_config.n_points_log
    )
    e1_lin = np.linspace(
        input_config.spectra_2d_config.e_max_log,
        input_config.spectra_config.q_value-input_config.spectra_config.min_ke,
        input_config.spectra_2d_config.n_points_lin
    )
    e1_final = np.concatenate((e1_log, e1_lin[1:]))
    e2_final = e1_final.copy()
    e1_final, e2_final = np.meshgrid(e1_final, e2_final, indexing="ij")

    if input_config.spectra_config.method == ph.CLOSUREMETHOD:
        if input_config.process == ph.TWONEUTRINO_TWOBMINUS:
            atilde = 1.12*(input_config.initial_atom.mass_number**0.5)
            enei = atilde - 0.5 * \
                (input_config.spectra_config.q_value + 2.0*ph.electron_mass)
            spectrum = spectra.ClosureSpectrum2nubb(
                q_value=input_config.spectra_config.q_value, energy_points=spectrum_energy_grid, enei=enei)
        elif input_config.process == ph.NEUTRINOLESS_TWOBMINUS:
            spectrum = spectra.spectrum_0nubb(
                q_value=input_config.spectra_config.q_value, energy_points=spectrum_energy_grid, nuclear_radius=nuclear_radius)
    elif input_config.spectra_config.method == ph.TAYLORMETHOD:
        if input_config.process == ph.TWONEUTRINO_TWOBMINUS:
            spectrum = spectra.TaylorSpectrum2nubb(
                q_value=input_config.spectra_config.q_value, energy_points=spectrum_energy_grid, orders=input_config.spectra_config.orders
            )

    print("Computing spectra:")
    psf_collection = {}
    spectra_collection = {}
    spectra_2d_collection = {}
    eta1_grid = None
    eta2_grid = None

    ff0_vals = {}
    ff1_vals = {}

    for ff_type in input_config.spectra_config.fermi_functions:
        ff_type_nice = list(
            filter(lambda x: ph.FERMIFUNCTIONS[x] == ff_type, ph.FERMIFUNCTIONS))[0]
        psf_collection[ff_type_nice] = {}
        spectra_collection[ff_type_nice] = {}
        spectra_2d_collection[ff_type_nice] = {}

        print("\t"*1, f"- {ff_type_nice}")
        if ff_type == ph.NUMERIC_FERMIFUNCTIONS & (not has_scattering_state_config):
            print("ERROR: Numeric fermi functions requested, but no configuration was given.\n"
                  "Spectrum will not be computed with numeric fermi functions")
            continue

        if ff_type == ph.NUMERIC_FERMIFUNCTIONS:
            ff = fermi_functions.Numeric(
                wf_handler_final.scattering_handler, nuclear_radius)
        elif ff_type == ph.POINTLIKE_FERMIFUNCTIONS:
            ff = fermi_functions.PointLike(
                input_config.final_atom.Z, nuclear_radius, spectrum_energy_grid
            )
        elif ff_type == ph.CHARGEDSPHERE_FERMIFUNCTIONS:
            ff = fermi_functions.ChargedSphere(
                input_config.final_atom.Z, nuclear_radius
            )

        ff0_vals[ff_type] = np.zeros_like(spectrum_energy_grid)
        ff1_vals[ff_type] = np.zeros_like(spectrum_energy_grid)
        for ike in range(len(spectrum_energy_grid)):
            ke = spectrum_energy_grid[ike]
            ff0_vals[ff_type][ike] = ff.ff0_eval(ke)
            ff1_vals[ff_type][ike] = ff.ff1_eval(ke)

        for sp_type in input_config.spectra_config.types:
            sp_type_nice = ph.SPECTRUM_TYPES_NICE[sp_type]
            if sp_type == ph.ANGULARSPECTRUM:
                if input_config.process == ph.TWONEUTRINO_TWOBMINUS:
                    spectrum_vals = spectrum.compute_spectrum(
                        sp_type, ff.ff1_eval,
                        eta_total if ff_type == ph.NUMERIC_FERMIFUNCTIONS else None)
                else:
                    spectrum_vals = spectrum.compute_spectrum(
                        sp_type, 1, ff.ff1_eval,
                        eta_total if ff_type == ph.NUMERIC_FERMIFUNCTIONS else None)
            else:
                if input_config.process == ph.TWONEUTRINO_TWOBMINUS:
                    spectrum_vals = spectrum.compute_spectrum(
                        sp_type, ff.ff0_eval,
                        eta_total if ff_type == ph.NUMERIC_FERMIFUNCTIONS else None)
                else:
                    spectrum_vals = spectrum.compute_spectrum(
                        sp_type, 1, ff.ff0_eval,
                        eta_total if ff_type == ph.NUMERIC_FERMIFUNCTIONS else None)

            spectrum_integral = spectrum.integrate_spectrum(spectrum_vals)
            spectra_collection[ff_type_nice][sp_type_nice] = spectra.normalize_spectra(
                spectrum_vals, spectrum_integral)

            psf = spectrum.compute_psf(spectrum_integral)
            psf_type_nice = ph.PSF_TYPES_NICE[sp_type]
            psf_collection[ff_type_nice][psf_type_nice] = psf

            if not args.compute_2d_spectrum:
                continue
            if sp_type == ph.SUMMEDSPECTRUM:
                continue

            if (sp_type == ph.ANGULARSPECTRUM):
                spectrum_vals = spectrum.compute_2d_spectra(
                    sp_type, ff.ff1_eval, e1_final, e2_final,
                    eta_total if ff_type == ph.NUMERIC_FERMIFUNCTIONS else None)
            else:
                spectrum_vals = spectrum.compute_2d_spectra(
                    sp_type, ff.ff0_eval, e1_final, e2_final,
                    eta_total if ff_type == ph.NUMERIC_FERMIFUNCTIONS else None)

            if type(spectrum_vals) == dict:
                for ord in spectrum_vals:
                    for ie in range(len(spectrum_vals[ord])):
                        for je in range(len(spectrum_vals[ord][ie])):
                            spectrum_vals[ord][ie, je] = spectrum_vals[ord][ie, je] / \
                                spectrum_integral[ord]
            else:
                for ie in range(len(spectrum_vals)):
                    for je in range(len(spectrum_vals[ie])):
                        spectrum_vals[ie, je] = spectrum_vals[ie, je] / \
                            spectrum_integral

            spectra_2d_collection[ff_type_nice][sp_type_nice] = spectrum_vals

    print(psf_collection)
    io_handler.write_spectra(
        f"{input_config.output_prefix}_spectra.dat", input_config.initial_atom.name_nice, input_config.process_name, spectrum_energy_grid, spectra_collection, psf_collection)
    if spectra_2d_collection is not None:
        io_handler.write_2d_spectra(
            f"{input_config.output_prefix}_spectra_2d.dat",
            input_config.initial_atom.name_nice,
            input_config.process_name,
            e1_final,
            e2_final,
            spectrum_energy_grid[0],
            spectra_2d_collection,
            psf_collection)

    io_handler.write_fermi_functions(
        f"{input_config.output_prefix}_fermi_functions.dat", input_config.initial_atom.name_nice, input_config.process_name, spectrum_energy_grid, ff0_vals, ff1_vals)
    return
    print("Evaluating fermi functions")
    ff = fermi_functions.Numeric(
        wf_handler_final.scattering_handler, nuclear_radius)

    plot_phase_shifts(wf_handler_final)

    ff2 = fermi_functions.ChargedSphere(
        input_config.final_atom.Z, nuclear_radius)
    spectrum_energy_grid = np.logspace(np.log10(wf_handler_final.scattering_handler.energy_grid[0]*ph.hartree_energy),
                                       np.log10(wf_handler_final.scattering_handler.energy_grid[-1] *
                                       ph.hartree_energy),
                                       100)
    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.zeros_like(spectrum_energy_grid)
    y_ana = np.zeros_like(spectrum_energy_grid)
    for i in range(len(y)):
        y[i] = np.abs(ff.ff0_eval(spectrum_energy_grid[i]))
        y_ana[i] = np.abs(ff2.ff0_eval(spectrum_energy_grid[i]))

    ax.plot(spectrum_energy_grid, y)
    ax.plot(spectrum_energy_grid, y_ana)
    ax.set_xlim(0.9*spectrum_energy_grid[0],
                1.1*spectrum_energy_grid[-1])
#    ax.set_ylim(0.9*np.min(y), 1.1*np.max(y))
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_title("ff0")

    fig, ax = plt.subplots(figsize=(8, 6))
    y1 = np.zeros_like(y)
    y1_ana = np.zeros_like(y)
    for i in range(len(y1)):
        y1[i] = ff.ff1_eval(spectrum_energy_grid[i])
        y1_ana[i] = ff2.ff1_eval(spectrum_energy_grid[i])
    ax.plot(spectrum_energy_grid,
            y1)
    ax.plot(spectrum_energy_grid, y1_ana)
    ax.set_title("ff1")
    ax.set_xscale('log')

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].scatter(wf_handler_final.scattering_handler.energy_grid*ph.hartree_energy/ph.MeV,
                  np.abs(ff.gm1))
    ax[1].scatter(wf_handler_final.scattering_handler.energy_grid*ph.hartree_energy/ph.MeV,
                  np.abs(ff.fp1))
    ax[0].set_title("gm1")
    ax[1].set_title("fp1")
    ax[0].set_xscale('log')
    ax[1].set_xscale('log')

    plt.show()
    exit(0)
    atilde = 1.12*(input_config.initial_atom.mass_number**0.5)
    q_val = wf_handler_final.scattering_handler.energy_grid[-1] * \
        ph.hartree_energy
    enei = atilde - 0.5*(q_val + 2.0*ph.electron_mass)
    closure_spectrum = spectra.ClosureSpectrum2nubb(
        q_value=spectrum_energy_grid[-1], energy_points=spectrum_energy_grid, enei=enei)

    # fig, ax = plt.subplots()
    # ax.plot(spectrum_energy_grid, ex_corr.eta_total_func(spectrum_energy_grid))
    # ax.scatter(wf_handler_final.scattering_handler.energy_grid *
    #            ph.hartree_energy, ex_corr.eta_total, alpha=0.5)
    print("Computing spectra and PSF")
    sp_type = ph.SUMMEDSPECTRUM
    sp = closure_spectrum.compute_spectrum(sp_type, ff.ff0_eval)
    sp_integral = closure_spectrum.integrate_spectrum(sp)
    psf = closure_spectrum.compute_psf(sp_integral)
    sp_ana = closure_spectrum.compute_spectrum(
        sp_type, ff.ff0_eval)
    sp_ana_integral = closure_spectrum.integrate_spectrum(sp_ana)
    psf_ana = closure_spectrum.compute_psf(sp_ana_integral)
    sp_angular = closure_spectrum.compute_spectrum(
        ph.ANGULARSPECTRUM, ff.ff1_eval)
    sp_angular_integral = closure_spectrum.integrate_spectrum(sp_angular)
    psf_angular = closure_spectrum.compute_psf(sp_angular_integral)
    sp_angular_ana = closure_spectrum.compute_spectrum(
        ph.ANGULARSPECTRUM, ff.ff1_eval)
    sp_angular_ana_integral = closure_spectrum.integrate_spectrum(
        sp_angular_ana)
    psf_angular_ana = closure_spectrum.compute_psf(sp_angular_ana_integral)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(spectrum_energy_grid, sp_angular/sp)
    ax.plot(spectrum_energy_grid, sp_angular_ana/sp)
    ax.set_title("angular correlation")

    print(f"PSF = {psf/1E-21:11.5e} y^-1 {psf_ana/1E-21:11.5e} y^-1")
    print(f"PSF_ANGULAR = "
          f"{psf_angular/1E-21:11.5e} y^-1 {psf_angular_ana/1E-21:11.5e} y^-1")
    print(f"K = {psf_angular/psf} {psf_angular_ana/psf_ana}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(spectrum_energy_grid, sp / sp_integral)
    ax.plot(spectrum_energy_grid, sp_ana / sp_ana_integral)
    ax.set_xlim(0., spectrum_energy_grid[-1])
    # ax.set_ylim(0., 1.8)
    ax.set_title("single spectrum")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(spectrum_energy_grid, sp_angular / sp_angular_integral)
    ax.plot(spectrum_energy_grid, sp_angular_ana / sp_angular_ana_integral)
    ax.set_xlim(0., spectrum_energy_grid[-1])
    # ax.set_ylim(0., 1.8)
    ax.set_title("angular spectrum")

    plt.show()


if __name__ == "__main__":
    main()

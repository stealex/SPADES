#!/usr/bin/env python
import os
import yaml
from argparse import ArgumentParser
from src.wavefunctions_config import run_config
from src.output_config import output_config
from src.output_handler_old import output_handler
from src.wavefunctions import wavefunctions_handler
from src import io_handler
import matplotlib.pyplot as plt
import numpy as np
from src import ph, math_stuff, fermi_functions, exchange, spectra
import time
from grid_strategy import strategies


def next_plot_indices(i_row, i_col, n_rows, n_cols):
    if i_col < n_cols-1:
        return (i_row, i_col+1)
    else:
        return (i_row+1, 0)


def plot_asymptotic(wf_handler_final: wavefunctions_handler, e_values: list):
    a = strategies.SquareStrategy()
    k_values = wf_handler_final.scattering_config.k_values
    # print(a.get_grid_arrangement(2*len(e_values)*len(k_values)))
    n_rows, n_cols = a.get_grid_arrangement(2*len(k_values))

    r_grid = wf_handler_final.scattering_handler.r_grid
    for i_e in range(len(e_values)):
        fig, axs = plt.subplots(n_rows, n_cols,
                                figsize=(6*n_cols, 4*n_rows))
        i_row = 0
        i_col = 0

        index_e_current = np.abs(
            wf_handler_final.scattering_handler.energy_grid*ph.hartree_energy - e_values[i_e]).argmin()

        e = wf_handler_final.scattering_handler.energy_grid[index_e_current] * \
            ph.hartree_energy
        momentum = np.sqrt(e*(e+2*ph.electron_mass))
        norm = np.sqrt((e + 2.0*ph.electron_mass) / (2.0*(e+ph.electron_mass)))
        eta = math_stuff.sommerfeld_param(
            wf_handler_final.scattering_handler.z_inf, 1., e)
        for i_k in range(len(k_values)):
            k = k_values[i_k]
            l = k if k > 0 else -k-1
            # print(k, l)
            kr = momentum*(r_grid*ph.bohr_radius/ph.fermi) / ph.hc
            # print("momentum = ", momentum)
            # print("r_grid = ", r_grid)
            # print("ph.bohr_radius/ph.fermi = ", ph.bohr_radius/ph.fermi)
            # print(kr)

            p = wf_handler_final.scattering_handler.p_grid[k][index_e_current]
            q = wf_handler_final.scattering_handler.q_grid[k][index_e_current]

            delta = wf_handler_final.scattering_handler.phase_grid[k][index_e_current]
            coulomb_delta = math_stuff.coulomb_phase_shift(
                e, wf_handler_final.scattering_handler.z_inf, k)
            # print(e, eta, wf_handler_final.scattering_handler.z_inf,
            #       delta, coulomb_delta)
            axs[i_row, i_col].plot(r_grid, p)
            axs[i_row, i_col].plot(r_grid, np.sin(
                kr-l*np.pi/2.-eta*np.log(2*kr)+delta+coulomb_delta))
            # axs[i_row, i_col].set_xscale('log')
            axs[i_row, i_col].set_xlabel("r [bohr]")
            axs[i_row, i_col].set_ylabel("P(r)")
            axs[i_row, i_col].set_title(f"E = {e} [MeV]; kappa = {k}")
            # print(i_row, i_col)
            i_row, i_col = next_plot_indices(i_row, i_col, n_rows, n_cols)

            axs[i_row, i_col].plot(r_grid, q)
            # axs[i_row, i_col].plot(r_grid, np.cos(
            #     kr-l*np.pi/2.-eta*np.log(2*kr)+delta+coulomb_delta))
            # axs[i_row, i_col].set_xscale('log')
            axs[i_row, i_col].set_xlabel("r [bohr]")
            axs[i_row, i_col].set_ylabel("Q(r)")
            axs[i_row, i_col].set_title(f"E = {e} [MeV]; kappa = {k}")

            # print(i_row, i_col)
            i_row, i_col = next_plot_indices(i_row, i_col, n_rows, n_cols)


def plot_phase_shifts(wf_handler_final: wavefunctions_handler):

    a = strategies.SquareStrategy()
    k_values = wf_handler_final.scattering_config.k_values
    n_rows, n_cols = a.get_grid_arrangement(len(k_values)+2)

    i_row = 0
    i_col = 0
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    total_phase = np.zeros(
        (2, len(wf_handler_final.scattering_handler.phase_grid[-1])))
    for i_k in range(len(k_values)):
        k = k_values[i_k]
        coul_phase = np.zeros_like(total_phase[0])
        for i_e in range(len(coul_phase)):
            energy = wf_handler_final.scattering_handler.energy_grid[i_e] * \
                ph.hartree_energy
            coul_phase[i_e] = math_stuff.coulomb_phase_shift(
                energy, wf_handler_final.scattering_handler.z_inf, k)

        axs[i_row, i_col].scatter(
            wf_handler_final.scattering_handler.energy_grid*ph.hartree_energy,
            coul_phase,
            label=r"$\Delta_C$"
        )

        axs[i_row, i_col].scatter(
            wf_handler_final.scattering_handler.energy_grid*ph.hartree_energy,
            wf_handler_final.scattering_handler.phase_grid[k]-coul_phase,
            label=r"$\delta$"
        )

        total_phase[i_k] = wf_handler_final.scattering_handler.phase_grid[k]

        axs[i_row, i_col].scatter(
            wf_handler_final.scattering_handler.energy_grid*ph.hartree_energy,
            total_phase[i_k],
            label=r"$\delta + \Delta_C$"
        )

        axs[i_row, i_col].set_xlabel("Energy [MeV]")
        axs[i_row, i_col].set_ylabel("Phase")
        axs[i_row, i_col].set_xscale('log')
        axs[i_row, i_col].set_title(r"$\kappa = $"+f"{k}")

        axs[i_row, i_col].legend(loc=2)

        if i_col+1 < n_cols:
            i_col = i_col+1
        else:
            i_col = 0
            i_row = i_row+1

    axs[1, 0].scatter(
        wf_handler_final.scattering_handler.energy_grid*ph.hartree_energy,
        total_phase[1] - total_phase[0]
    )
    axs[1, 0].set_xlabel("E [MeV]")
    axs[1, 0].set_ylabel(r"$\Delta_{1} - \Delta_{-1}$")
    axs[1, 0].set_xscale('log')

    axs[1, 1].scatter(
        wf_handler_final.scattering_handler.energy_grid*ph.hartree_energy,
        np.cos(total_phase[1] - total_phase[0])
    )
    axs[1, 1].set_xlabel("E [MeV]")
    axs[1, 1].set_ylabel(r"$cos(\Delta_{1} - \Delta_{-1})$")
    axs[1, 1].set_xscale('log')


def main(argv=None):
    '''Command line arguments'''
    parser = ArgumentParser(
        description="Compute spectra and PSFs for 2nubb decay")
    parser.add_argument("config_file",
                        help="path to yaml configuration file")
    parser.add_argument("--verbose",
                        help="verbosity level: 0 = lowest; 5 = highest",
                        action="store",
                        choices=range(0, 6),
                        default=0)

    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        run_conf = yaml.safe_load(f)

    input_config = run_config(run_conf)
    has_bound_state_config = True
    try:
        input_config.bound_config.print()
    except AttributeError:
        print("No bound states configuration for RADIAL")
        has_bound_state_config = False

    has_scattering_state_config = True
    try:
        input_config.scattering_config.print()
    except AttributeError:
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

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(wf_handler_initial.dhfs_handler.rad_grid,
                wf_handler_initial.dhfs_handler.rv_modified)
        ax.set_xscale('log')

    if has_scattering_state_config:
        print("Computing wavefunctions for final atom")

        wf_handler_final = wavefunctions_handler(
            input_config.final_atom, input_config.bound_config, input_config.scattering_config)

        start_time = time.time()
        wf_handler_final.find_all_wavefunctions()
        stop_time = time.time()
        print(f"... took {stop_time-start_time: .2f} seconds")

    if has_bound_state_config & has_scattering_state_config & (ph.EXCHANGECORRECTION in input_config.spectra_config.corrections):
        print(f"Computign exchange correction")
        ex_corr = exchange.exchange_correction(wf_handler_initial,
                                               wf_handler_final)
        start_time = time.time()
        ex_corr.compute_eta_total()
        stop_time = time.time()
        eta_total = ex_corr.eta_total_func
        print(f"... took {stop_time-start_time: .2f} seconds")
    else:
        eta_total = None

    # out_conf = output_config(location=run_conf["output"]["location"])
    # out_handler = output_handler(out_conf, input_config)
    # out_handler.output_dhfs(wf_handler_initial.dhfs_handler)

    # out_handler.plot_scattering_wf(
    #     wf_handler_final.scattering_handler,
    #     np.array([2.0-ph.electron_mass])*ph.MeV/ph.hartree_energy,
    #     np.array([-1, 1]))

    # plot_asymptotic(wf_handler_final, [1E-4, 2.0])

    if type(input_config.spectra_config.nuclear_radius) == str:
        nuclear_radius = 1.2*(input_config.initial_atom.mass_number**(1./3.))
    elif type(input_config.spectra_config.nuclear_radius) == float:
        nuclear_radius = input_config.spectra_config.nuclear_radius

    if has_scattering_state_config:
        q_value = input_config.scattering_config.max_ke*ph.hartree_energy
        min_ke = input_config.scattering_config.min_ke*ph.hartree_energy
        n_ke_points = input_config.scattering_config.n_ke_points
    elif type(input_config.spectra_config.q_value) == float:
        q_value = input_config.spectra_config.q_value
        if type(input_config.spectra_config.min_ke) == float:
            min_ke = input_config.spectra_config.min_ke
        else:
            raise ValueError(
                "Could not determine spectrum start point."
                "Either the minimum ke for scattering states or for spectra has to be given")
        if type(input_config.spectra_config.n_ke_points) == int:
            n_ke_points = input_config.spectra_config.n_ke_points
        else:
            raise ValueError(
                "Could not determine number of energy points for spectrum."
                "Either the number of points for scattering states or for spectra has to be given"
            )
    else:
        raise ValueError(
            "Could not determine spectrum end point."
            "Either the maximum electron ke or the q-value has to be given")

    if (input_config.spectra_config.energy_grid_type == "lin"):
        spectrum_energy_grid = np.linspace(min_ke, q_value, n_ke_points)
    elif input_config.spectra_config.energy_grid_type == "log":
        spectrum_energy_grid = np.logspace(
            np.log10(min_ke), np.log10(q_value), n_ke_points)

    if input_config.spectra_config.method == ph.CLOSUREMETHOD:
        atilde = 1.12*(input_config.initial_atom.mass_number**0.5)
        enei = atilde - 0.5*(q_value + 2.0*ph.electron_mass)
        spectrum = spectra.closure_spectrum(
            q_value=q_value, energy_points=spectrum_energy_grid, enei=enei)

    print("Computing spectra:")
    psf_collection = {}
    spectra_collection = {}

    for ff_type in input_config.spectra_config.fermi_functions:
        ff_type_nice = list(
            filter(lambda x: ph.FERMIFUNCTIONS[x] == ff_type, ph.FERMIFUNCTIONS))[0]
        psf_collection[ff_type_nice] = {}
        spectra_collection[ff_type_nice] = {}

        print("\t"*1, f"- {ff_type_nice}")
        if ff_type == ph.NUMERICFERMIFUNCTIONS & (not has_scattering_state_config):
            print("ERROR: Numeric fermi functions requested, but no configuration was given.\n"
                  "Spectrum will not be computed with numeric fermi functions")
            continue

        if ff_type == ph.NUMERICFERMIFUNCTIONS:
            ff = fermi_functions.numeric(
                wf_handler_final.scattering_handler, nuclear_radius)

        elif ff_type == ph.POINTLIKEFERMIFUNCTIONS:
            ff = fermi_functions.point_like(
                input_config.final_atom.Z, nuclear_radius, spectrum_energy_grid
            )
        elif ff_type == ph.CHARGEDSPHEREFERMIFUNCTIONS:
            ff = fermi_functions.charged_sphere(
                input_config.final_atom.Z, nuclear_radius
            )

        for sp_type in input_config.spectra_config.types:
            sp_type_nice = ph.SPECTRUM_TYPES_NICE[sp_type]
            if sp_type == ph.ANGULARSPECTRUM:
                spectrum_vals = spectrum.compute_spectrum(
                    sp_type, ff.ff1_eval, eta_total)
            else:
                spectrum_vals = spectrum.compute_spectrum(
                    sp_type, ff.ff0_eval, eta_total)

            spectrum_integral = spectrum.integrate_spectrum(spectrum_vals)
            spectra_collection[ff_type_nice][sp_type_nice] = spectrum_vals / \
                spectrum_integral
            psf = spectrum.compute_psf(spectrum_integral)
            psf_collection[ff_type_nice][sp_type_nice] = psf

    io_handler.write_spectra(
        "spectra.dat", spectrum_energy_grid, spectra_collection, psf_collection)

    return
    print("Evaluating fermi functions")
    ff = fermi_functions.numeric(
        wf_handler_final.scattering_handler, nuclear_radius)

    plot_phase_shifts(wf_handler_final)

    ff2 = fermi_functions.charged_sphere(
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
    closure_spectrum = spectra.closure_spectrum(
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

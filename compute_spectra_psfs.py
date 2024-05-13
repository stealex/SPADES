#!/usr/bin/env python
import os
import yaml
from argparse import ArgumentParser
from configs.wavefunctions_config import run_config
from configs.output_config import output_config
from handlers.output_handler import output_handler
from wavefunctions_computation.wavefunctions import wavefunctions_handler
import matplotlib.pyplot as plt
import numpy as np
from utils import ph, math_stuff
from spectra import fermi_functions, closure

from grid_strategy import strategies


def plot_phase_shifts(wf_handler_final: wavefunctions_handler):

    a = strategies.SquareStrategy()
    k_values = wf_handler_final.scattering_config.k_values
    n_rows, n_cols = a.get_grid_arrangement(len(k_values)+2)

    i_row = 0
    i_col = 0
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    total_phase = np.zeros_like(wf_handler_final.scattering_handler.phase_grid)
    for i_k in range(len(k_values)):
        k = k_values[i_k]
        axs[i_row, i_col].scatter(
            wf_handler_final.scattering_handler.energy_grid*ph.hartree_energy,
            wf_handler_final.scattering_handler.phase_grid[i_k],
            label=r"$\delta$"
        )

        coul_phase = np.zeros_like(total_phase[i_k])
        for i_e in range(len(coul_phase)):
            energy = wf_handler_final.scattering_handler.energy_grid[i_e]
            coul_phase[i_e] = math_stuff.coulomb_phase_shift(
                energy*ph.hartree_energy, wf_handler_final.scattering_handler.z_inf, k)

        axs[i_row, i_col].scatter(
            wf_handler_final.scattering_handler.energy_grid*ph.hartree_energy,
            coul_phase,
            label=r"$\Delta_C$"
        )

        total_phase[i_k] = wf_handler_final.scattering_handler.phase_grid[i_k]+coul_phase

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
    input_config.bound_config.print()
    input_config.scattering_config.print()

    wf_handler_initial = wavefunctions_handler(
        input_config.initial_atom, input_config.bound_config)

    # run dhfs
    print("Computing wavefunctions for initial atom")
    wf_handler_initial.find_all_wavefunctions()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(wf_handler_initial.dhfs_handler.rad_grid,
            wf_handler_initial.dhfs_handler.rv_modified)
    ax.set_xscale('log')

    print("Computing wavefunctions for final atom")
    wf_handler_final = wavefunctions_handler(
        input_config.final_atom, input_config.bound_config, input_config.scattering_config)
    wf_handler_final.find_all_wavefunctions()

    out_conf = output_config(location=run_conf["output"]["location"])
    out_handler = output_handler(out_conf, input_config)
    out_handler.output_dhfs(wf_handler_initial.dhfs_handler)

    out_handler.plot_scattering_wf(
        wf_handler_final.scattering_handler,
        np.array([2.0-ph.electron_mass])*ph.MeV/ph.hartree_energy,
        np.array([-1, 1]))

    print("Evaluating fermi functions")
    ff = fermi_functions.numeric(wf_handler_final.scattering_handler)
    ff.eval_fg(1.2*(input_config.initial_atom.mass_number**(1./3.)))
    ff.build_fermi_functions()

    plot_phase_shifts(wf_handler_final)
    plt.show()
    exit(0)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(wf_handler_final.scattering_handler.energy_grid*ph.hartree_energy,
            wf_handler_final.scattering_handler.phase_grid[1])

    coulomb_phase_shift = np.zeros_like(
        wf_handler_final.scattering_handler.phase_grid[1])
    for i_e in range(len(wf_handler_final.scattering_handler.energy_grid)):
        coulomb_phase_shift[i_e] = math_stuff.coulomb_phase_shift(
            wf_handler_final.scattering_handler.energy_grid[i_e] *
            ph.hartree_energy,
            wf_handler_final.scattering_handler.z_inf,
            1
        )
    ax.plot(wf_handler_final.scattering_handler.energy_grid*ph.hartree_energy,
            coulomb_phase_shift)
    ax.set_title("phase")

    ff2 = fermi_functions.point_like(
        input_config.final_atom.Z, 1.2*(input_config.initial_atom.mass_number**(1./3.)))
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
            np.abs(y1))
    ax.plot(spectrum_energy_grid, np.abs(y1_ana))
    ax.set_title("ff1")

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].scatter(wf_handler_final.scattering_handler.energy_grid*ph.hartree_energy/ph.MeV,
                  np.abs(ff.gm1))
    ax[1].scatter(wf_handler_final.scattering_handler.energy_grid*ph.hartree_energy/ph.MeV,
                  np.abs(ff.fp1))
    ax[0].set_title("gm1")
    ax[1].set_title("fp1")
    ax[0].set_xscale('log')
    ax[1].set_xscale('log')

    # plt.show()
    # exit(0)

    atilde = 1.12*(input_config.initial_atom.mass_number**0.5)
    q_val = wf_handler_final.scattering_handler.energy_grid[-1] * \
        ph.hartree_energy
    enei = atilde - 0.5*(q_val + 2.0*ph.electron_mass)
    closure_spectrum = closure.closure_spectrum(
        q_value=spectrum_energy_grid[-1], energy_points=spectrum_energy_grid, enei=enei)

    print("Computing spectra and PSF")
    sp_type = ph.SINGLESPECTRUM
    sp = closure_spectrum.compute_spectrum(sp_type, ff.ff0_eval)
    sp_integral = closure_spectrum.integrate_spectrum(sp)
    psf = closure_spectrum.compute_psf(sp)
    sp_ana = closure_spectrum.compute_spectrum(sp_type, ff2.ff0_eval)
    sp_ana_integral = closure_spectrum.integrate_spectrum(sp_ana)
    psf_ana = closure_spectrum.compute_psf(sp_ana)
    sp_angular = closure_spectrum.compute_spectrum(
        ph.ANGULARSPECTRUM, ff.ff1_eval)
    sp_angular_integral = closure_spectrum.integrate_spectrum(sp_angular)
    psf_angular = closure_spectrum.compute_psf(sp_angular)
    sp_angular_ana = closure_spectrum.compute_spectrum(
        ph.ANGULARSPECTRUM, ff2.ff1_eval)
    sp_angular_ana_integral = closure_spectrum.integrate_spectrum(
        sp_angular_ana)
    psf_angular_ana = closure_spectrum.compute_psf(sp_angular_ana)

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

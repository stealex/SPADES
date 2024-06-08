import os
import numpy as np
import matplotlib.pyplot as plt

from src.dhfs import atomic_system, create_ion
from src import wavefunctions, ph, exchange, math_stuff


def test_exchange():
    dir_name = os.path.dirname(__file__)
    initial_atom = atomic_system({
        "name": "45Ca",
        "electron_config": "auto"
    })
    initial_atom.print()

    final_atom = create_ion(
        initial_atom, initial_atom.Z+1)
    ke_min = 2E-4
    ke_max = 1E-1
    k_values = {}
    for i_s in range(len(initial_atom.electron_config)):
        n = initial_atom.n_values[i_s]
        l = initial_atom.l_values[i_s]
        j = 0.5*initial_atom.jj_values[i_s]
        k = int((l-j)*(2*j+1))
        if not (n in k_values):
            k_values[n] = []
        k_values[n].append(k)

    n_values = list(set(initial_atom.n_values.tolist()))
    bound_config = wavefunctions.bound_config(
        100, 5000, n_values, k_values)
    bound_config.print()
    scattering_config = wavefunctions.scattering_config(
        100, 5000, ke_min/ph.hartree_energy, ke_max/ph.hartree_energy,
        100, [-1, 1])
    scattering_config.print()
    wf_handler_initial = wavefunctions.wavefunctions_handler(
        initial_atom, bound_conf=bound_config)

    wf_handler_final = wavefunctions.wavefunctions_handler(
        final_atom, bound_config, scattering_config
    )
    print("computing wf for initial atom")
    wf_handler_initial.find_all_wavefunctions()

    print("computing wf for final atom")
    wf_handler_final.find_all_wavefunctions()
    r_nuc = 1.2 * \
        ((wf_handler_initial.atomic_system.mass_number)**(1./3.))
    ir_nuc = np.abs(r_nuc*ph.fermi/ph.bohr_radius -
                    wf_handler_final.bound_handler.r_grid).argmin()
    exchange_correction = exchange.exchange_correction(
        wf_handler_initial, wf_handler_final)
    print("computing exchange")

    i_k = 0
    eta = {}
    tn = {}

    e_values = ph.electron_mass + \
        wf_handler_final.scattering_handler.energy_grid/ph.hartree_energy

    norm = np.sqrt((e_values+ph.electron_mass)/(2*e_values))

    gm1 = norm*wf_handler_final.scattering_handler.p_grid[-1][:, ir_nuc]
    fp1 = norm*wf_handler_final.scattering_handler.q_grid[1][:, ir_nuc]
    fs = gm1*gm1/(gm1*gm1+fp1*fp1)

    tn = exchange_correction.compute_t()
    eta = exchange_correction.compute_eta()
    eta_total = exchange_correction.compute_eta_total()
    eta_s = exchange_correction.eta_s
    eta_p = exchange_correction.eta_p
    for k in [-1, 1]:
        fig, ax = plt.subplots(2, 2)
        for i_n in range(len(n_values)):
            n = n_values[i_n]
            if not (k in k_values[n]):
                continue
            i_row, i_col = divmod(i_n, 2)
            ax[i_row, i_col].plot(1E3*wf_handler_final.scattering_handler.energy_grid*ph.hartree_energy,
                                  tn[n][k])
            ax[i_row, i_col].set_xscale('log')
            ax[i_row, i_col].set_xlim(1E3*ke_min, 1E3*ke_max)
            if k == -1:
                if (n == 1):
                    ax[i_row, i_col].set_ylim(-0.005, 0.085)
                if (n == 2):
                    ax[i_row, i_col].set_ylim(-0.004, 0.074)
                if (n == 3):
                    ax[i_row, i_col].set_ylim(-0.0015, 0.0315)
                if (n == 4):
                    ax[i_row, i_col].set_ylim(-0.0002, 0.0042)

        i_k = i_k+1

    fig, ax = plt.subplots()
    eta_total = eta_s+eta_p
    for n in n_values:
        ax.plot(1E3*wf_handler_final.scattering_handler.energy_grid*ph.hartree_energy,
                eta[n][-1])
    ax.plot(1E3*wf_handler_final.scattering_handler.energy_grid*ph.hartree_energy,
            eta_total)

    fig, ax = plt.subplots()
    for n in n_values:
        if not (1 in k_values[n]):
            continue
        ax.plot(1E3*wf_handler_final.scattering_handler.energy_grid*ph.hartree_energy,
                eta[n][1])

    plt.show()


test_exchange()

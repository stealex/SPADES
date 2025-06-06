import os
import numpy as np
import matplotlib.pyplot as plt

from spades.dhfs import AtomicSystem, create_ion
from spades import wavefunctions, ph, exchange, math_stuff


def test_exchange():
    dir_name = os.path.dirname(__file__)
    initial_atom = AtomicSystem(
        name="45Ca",
        electron_config="auto"
    )
    initial_atom.print()

    final_atom = create_ion(
        initial_atom, initial_atom.Z+1)
    ke_min = 2E-4  # MeV
    ke_max = 1E-1  # MeV
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
    bound_config = wavefunctions.BoundConfig(
        100*ph.bohr_radius/ph.fm, 5000, n_values, k_values)
    scattering_config = wavefunctions.ScatteringConfig(
        100*ph.bohr_radius/ph.fm, 5000, ke_min, ke_max,
        100, [-1, 1])
    wf_handler_initial = wavefunctions.WaveFunctionsHandler(
        initial_atom, bound_conf=bound_config)

    wf_handler_final = wavefunctions.WaveFunctionsHandler(
        final_atom, bound_config, scattering_config
    )

    print("computing wf for initial atom")
    wf_handler_initial.find_all_wavefunctions()

    print("computing wf for final atom")
    wf_handler_final.find_all_wavefunctions()

    print(
        f"Max diff = {np.argmax(np.abs(wf_handler_final.scattering_handler.r_grid - wf_handler_initial.bound_handler.r_grid))}")
    r_nuc = 1.2 * \
        ((wf_handler_initial.atomic_system.mass_number)**(1./3.))
    ir_nuc = np.abs(r_nuc -
                    wf_handler_final.bound_handler.r_grid).argmin()
    exchange_correction = exchange.ExchangeCorrection(
        wf_handler_initial, wf_handler_final)
    print("computing exchange")

    i_k = 0
    eta = {}
    tn = {}

    e_values = ph.electron_mass + \
        wf_handler_final.scattering_handler.energy_grid

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
            ax[i_row, i_col].plot(1E3*wf_handler_final.scattering_handler.energy_grid,
                                  tn[n][k])
            ax[i_row, i_col].set_xscale('log')
            ax[i_row, i_col].set_xlim(1E3*ke_min, 1E3*ke_max)
            ax[i_row, i_col].set_xlabel('E-m')
            ax[i_row, i_col].set_ylabel(f'T({n},{k})')
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
        ax.plot(1E3*wf_handler_final.scattering_handler.energy_grid,
                eta[n][-1])
    ax.plot(1E3*wf_handler_final.scattering_handler.energy_grid,
            eta_total)
    ax.set_xlabel('E-m')
    ax.set_ylabel("eta")

    fig, ax = plt.subplots()
    for n in n_values:
        if not (1 in k_values[n]):
            continue
        ax.plot(1E3*wf_handler_final.scattering_handler.energy_grid,
                eta[n][1])
    ax.legend()
    plt.show()


test_exchange()

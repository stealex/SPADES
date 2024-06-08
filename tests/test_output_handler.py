
import numpy as np

from src import io_handler
from src.wavefunctions import scattering_config, scattering_handler
from src.fermi_functions import numeric, charged_sphere, point_like

from src import ph

import matplotlib.pyplot as plt

import struct


def write_wf():
    z_nuc = 40
    r_nuc = 5.0  # fm

    rPoints = np.linspace(0., 100., 1000)
    rvValues = -z_nuc*np.ones_like(rPoints)

    min_ke = 1E-3  # 1 keV
    max_ke = 3.0  # 3 MeV
    scat_config = scattering_config(
        30., 2000, min_ke/ph.hartree_energy, max_ke/ph.hartree_energy, 100, [-1, 1])
    scat_handler = scattering_handler(z_nuc, z_nuc+2, scat_config)
    scat_handler.set_potential(rPoints, rvValues)
    scat_handler.compute_scattering_states()

    fig, ax = plt.subplots(ncols=2)
    ax[0].plot(scat_handler.r_grid, scat_handler.p_grid[-1][10])
    ax[0].set_xscale('log')
    ax[1].plot(scat_handler.energy_grid, scat_handler.phase_grid[-1])
    ax[1].set_xscale('log')
    io_handler.write_scattring_wf("test.dat",
                                  scat_config.k_values,
                                  scat_handler.energy_grid,
                                  scat_handler.phase_grid,
                                  scat_handler.coul_phase_grid,
                                  scat_handler.r_grid,
                                  scat_handler.p_grid,
                                  scat_handler.q_grid)


def read_wf():
    k_values, e_values, r_values, inner_phase_values, coulomb_phase_values, p_values, q_values = io_handler.read_scattering_wf(
        "test.dat")

    fig, ax = plt.subplots(ncols=2)
    ax[0].plot(r_values, p_values[-1][10])
    ax[0].set_xscale('log')
    ax[1].plot(e_values, inner_phase_values[-1])
    ax[1].set_xscale('log')

    print(f"k_values = {k_values}")
    print(f"e_values = {e_values}")
    print(f"r_values = {r_values}")
    print(f"p_values = {p_values[-1][10]}")


def write_spectra_psf():
    PSFs = {"Numeric": {"G": 1.28715E-4, "H": -2.781523E-4, "K": -0.55},
            "PointLike": {"G": 2.28715E-4, "H": -1.781523E-4, "K": -1.55},
            }

    e_grid = np.linspace(1E-3, 5., 100)
    sp1 = np.linspace(1., 10., 100)

    io_handler.write_spectra("test_spectra.dat", e_grid,
                             {"PointLike": {"dG/de": sp1, "dH/de": -sp1, "alpha": -sp1/sp1},
                              "Numeric": {"dG/de": sp1, "dH/de": -sp1, "alpha": -sp1/sp1}},
                             PSFs)


if __name__ == "__main__":
    # write_wf()
    # read_wf()
    write_spectra_psf()

    plt.show()

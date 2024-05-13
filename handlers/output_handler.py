from configs.output_config import output_config
from configs.wavefunctions_config import run_config
from handlers.wavefunction_handlers import dhfs_handler, scattering_handler
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from utils import ph
from grid_strategy import strategies


class output_handler:
    def __init__(self, out_conf: output_config, run_conf: run_config) -> None:
        self.output_config = out_conf
        self.run_config = run_conf
        self.prepare_directory()

    def prepare_directory(self):
        try:
            os.mkdir(self.output_config.location)
        except FileExistsError:
            pass

        self.final_dir = os.path.abspath(self.output_config.location)

    def output_dhfs(self, handler: dhfs_handler):
        base_name = f"{handler.dhfs_config.name}_dhfs.obj"

        with open(os.path.join(self.final_dir, base_name), 'wb') as f:
            pickle.dump(handler, f)

    def plot_scattering_wf(self, scatt_handler: scattering_handler, energy_values: np.ndarray, k_values: np.ndarray):
        available_energies = scatt_handler.energy_grid
        for i_e in range(len(energy_values)):
            a = strategies.SquareStrategy()
            n_rows, n_cols = a.get_grid_arrangement(2*len(k_values))
            fig, ax = plt.subplots(
                n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
            id_best = np.abs(available_energies-energy_values[i_e]).argmin()
            energy_true = available_energies[i_e]
            i_row = 0
            i_col = 0
            for i_k in range(len(k_values)):
                k = k_values[i_k]
                ax[i_row, i_col].plot(scatt_handler.r_grid*ph.bohr_radius /
                                      ph.fermi, np.abs(scatt_handler.g_grid[i_k][id_best]))
                ax[i_row, i_col].set_xlim(0., 8.)
                ax[i_row, i_col].set_title(f"g {k}")

                ax[i_row, i_col+1].plot(scatt_handler.r_grid*ph.bohr_radius/ph.fermi,
                                        np.abs(scatt_handler.f_grid[i_k][id_best]))
                ax[i_row, i_col+1].set_xlim(0., 8.)
                ax[i_row, i_col+1].set_title(f"f {k}")

                i_row = i_row+1
                i_col = 0

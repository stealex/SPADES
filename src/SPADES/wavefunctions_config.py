import re
import periodictable
import os
import yaml
import numpy as np
from copy import deepcopy
from . import ph
from .dhfs import atomic_system, create_ion
from .wavefunctions import bound_config, scattering_config
from .spectra import spectra_config


class run_config:
    def __init__(self, config: dict) -> None:
        self.task_name = config["task"]
        self.process_name = config["process"]

        ph.user_distance_unit_name = config["distance_unit"]
        ph.user_energy_unit_name = config["energy_unit"]
        ph.user_distance_unit = ph.__dict__[config["distance_unit"]]
        ph.user_energy_unit = ph.__dict__[config["energy_unit"]]

        # atoms
        self.initial_atom = atomic_system(config["initial_atom"])
        self.final_atom = create_ion(
            self.initial_atom, self.initial_atom.Z + 2)

        # technicals

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

            self.scattering_config = scattering_config(max_r=config["scattering_states"]["max_r"]*ph.user_distance_unit/ph.bohr_radius,
                                                       n_radial_points=config["scattering_states"]["n_radial_points"],
                                                       min_ke=config["scattering_states"]["min_ke"] *
                                                       ph.user_energy_unit,
                                                       max_ke=config["scattering_states"]["max_ke"] *
                                                       ph.user_energy_unit,
                                                       n_ke_points=config["scattering_states"]["n_ke_points"],
                                                       k_values=k_values)
        else:
            self.scattering_config = None

        method = config["spectra_computation"]["method"]
        wavefunction_eval = config["spectra_computation"]["wavefunction_evaluation"]
        nuclear_radius = config["spectra_computation"]["nuclear_radius"]
        types = config["spectra_computation"]["types"]
        energy_grid_type = config["spectra_computation"]["energy_grid_type"]
        corrections = config["spectra_computation"]["corrections"]
        fermi_functions = config["spectra_computation"]["fermi_functions"]
        q_value = config["spectra_computation"]["q_value"]
        min_ke = config["spectra_computation"]["min_ke"]
        n_ke_points = config["spectra_computation"]["n_ke_points"]
        self.spectra_config = spectra_config(
            method, wavefunction_eval, nuclear_radius, types, energy_grid_type, fermi_functions, corrections, q_value, min_ke, n_ke_points)

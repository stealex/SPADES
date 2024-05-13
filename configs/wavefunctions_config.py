import re
import periodictable
import os
import yaml
import numpy as np
from copy import deepcopy
from utils import ph


class atomic_system:
    def __init__(self, config: dict) -> None:
        self.name = config["name"]
        matches = re.match(r'(\d+)([A-Za-z]+)', self.name)
        if (type(matches) == re.Match):
            self.mass_number = int(matches.group(1))
            self.symbol = matches.group(2)
        else:
            raise ValueError(
                "Could not identify mass number and symbol for input atom. Format should be <A><Symbol>")

        if ("weight" in config):
            self.weight = float(config["weight"])
        else:
            self.weight = -1.

        if (self.weight < 0.):
            self.weight = float(self.mass_number)

        tmp = config["electron_config"]
        if (tmp == "auto"):
            z = periodictable.elements.isotope(self.symbol).number
            electron_configuration_filename = os.path.join(os.path.dirname(__file__),
                                                           f"../data/ground_state_config_Z{z:03d}.yaml")
        else:
            electron_configuration_filename = tmp

        with open(electron_configuration_filename, 'r') as f:
            config_tmp = yaml.safe_load(f)

        self.Z = config_tmp["Z"]
        self.name = config_tmp["Name"]
        self.electron_config = np.array(config_tmp["configuration"])

        self.n_values = self.electron_config[:, 0].astype(int)
        self.l_values = self.electron_config[:, 1].astype(int)
        self.jj_values = self.electron_config[:, 2].astype(int)
        self.k_values = (self.l_values-0.5*self.jj_values) * \
            (self.jj_values+1).astype(int)
        self.occ_values = self.electron_config[:, 3].astype(float)

    def print(self):
        text = f"""DHFS configuration:
Z: {self.Z:d}
name: {self.name:s}
configuration:[n,l,2j,occupation]
"""
        for c in self.electron_config:
            text = text + f"  - [{c[0]:d},{c[1]:d},{c[2]:d},{c[3]:f}]\n"

        print(text)


def create_ion(atom: atomic_system, z_nuc) -> atomic_system:
    ion = deepcopy(atom)
    ion.Z = z_nuc
    ion.name = f"{ion.mass_number:d}{periodictable.elements[ion.Z].symbol:s}"
    return ion


class radial_bound_config:
    def __init__(self, max_r: float, n_radial_points: int,
                 n_values: int | tuple[int, int] | list[int],
                 k_values: str | int | tuple[int, int] | dict[int, list[int]]) -> None:
        self.max_r = max_r
        self.n_radial_points = n_radial_points

        if (type(n_values) == int):
            self.n_values = [n_values]
        elif (type(n_values) == tuple):
            self.n_values = range(n_values[0], n_values[1]+1)
        elif (type(n_values) == list):
            self.n_values = n_values

        self.k_values = {}
        for i_n in range(len(self.n_values)):
            n = self.n_values[i_n]
            if (type(k_values) == str):
                if k_values != "auto":
                    raise ValueError("Cannot interpret k_values option")
                else:
                    k_tmp = range(-n, n)
            if (type(k_values) == int):
                if (k_values < -n) or (k_values >= n) or (k_values == 0):
                    continue
                self.k_values[n] = k_values
                continue
            elif type(k_values) == tuple:
                k_tmp = range(k_values[0], k_values[1]+1)
            elif type(k_values) == dict:
                self.k_values = k_values
                break

            for i_k in range(len(k_tmp)):
                if (k_tmp[i_k] < -n) or (k_tmp[i_k] >= n) or (k_tmp[i_k] == 0):
                    continue
                if (n not in self.k_values):
                    self.k_values[n] = []
                self.k_values[n].append(k_tmp[i_k])

    def print(self):
        print("Configuration for bound states")
        print(f"  - Maximum radial distance: {self.max_r: 8.3f} bohr")
        print(f"  - Number of radial points: {self.n_radial_points: d}")
        print(f"  - N and K values:")
        for i in range(len(self.n_values)):
            print(f"    - {self.n_values[i]}", self.k_values[self.n_values[i]])


class radial_scattering_config:
    def __init__(self, max_r: float, n_radial_points: int,
                 min_ke: float, max_ke: float, n_ke_points: int,
                 k_values: int | tuple[int, int] | list[int]) -> None:
        self.max_r = max_r
        self.n_radial_points = n_radial_points

        self.min_ke = min_ke
        self.max_ke = max_ke
        self.n_ke_points = n_ke_points

        self.k_values = []
        if type(k_values) == int:
            self.k_values.append(k_values)
        elif type(k_values) == tuple:
            for i_k in range(k_values[0], k_values[1]+1):
                if (k_values[i_k] == 0):
                    continue
                self.k_values.append(k_values[i_k])
        elif type(k_values) == list:
            for k in k_values:
                if (k == 0):
                    continue
                self.k_values.append(k)
        else:
            raise ValueError("Cannot interpret the type of k_values")

    def print(self):
        print("Configuration for scattering states")
        print(f"  - Maximum radial distance: {self.max_r: 8.3f} bohr")
        print(f"  - Number of radial points: {self.n_radial_points: d}")
        print("  - K values: ", self.k_values)


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
        self.bound_config = radial_bound_config(max_r=config["bound_states"]["max_r"]*ph.user_distance_unit/ph.bohr_radius,
                                                n_radial_points=config["bound_states"]["n_radial_points"],
                                                n_values=n_values,
                                                k_values=k_values)

        k_values = config["scattering_states"]["k_values"]
        if (config["scattering_states"]["k_values"] == "auto"):
            k_values = [-1, 1]

        self.scattering_config = radial_scattering_config(max_r=config["scattering_states"]["max_r"]*ph.user_distance_unit/ph.bohr_radius,
                                                          n_radial_points=config["scattering_states"]["n_radial_points"],
                                                          min_ke=config["scattering_states"]["min_ke"] *
                                                          ph.user_energy_unit/ph.hartree_energy,
                                                          max_ke=config["scattering_states"]["max_ke"] *
                                                          ph.user_energy_unit/ph.hartree_energy,
                                                          n_ke_points=config["scattering_states"]["n_ke_points"],
                                                          k_values=k_values)

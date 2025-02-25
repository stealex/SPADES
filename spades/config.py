from dataclasses import dataclass, field
from logging import config
from multiprocessing import Value
from optparse import Option
from tokenize import Double
from typing import Optional, Dict, Any
from spades import ph
from spades.dhfs import AtomicSystem, create_ion
import logging
logger = logging.Logger(__name__)


@dataclass
class DoubleBetaProcess:
    type: int
    transition: int = ph.ZEROPLUS_TO_ZEROPLUS
    mechanism: int = ph.LIGHT_NEUTRINO_EXCHANGE


@dataclass
class BoundConfig:
    max_r: float
    n_radial_points: int
    n_values: list[int]
    k_values: dict[int, list[int]]


@dataclass
class ScatteringConfig:
    max_r: float
    n_radial_points: int
    min_ke: float
    max_ke: float
    n_ke_points: int
    k_values: list[int]


@dataclass
class SpectraConfig:
    method: dict
    wave_function_evaluation: int
    nuclear_radius: float
    types: list[int]
    fermi_function_types: list[int]
    energy_grid_type: int
    n_ke_points: int
    min_ke: float
    q_value: float
    corrections: list[str] = field(default_factory=list)

    n_points_log_2d: int = 0
    n_points_lin_2d: int = 0
    e_max_log_2d: float = -1.


class RunConfig:
    def __init__(self, config: dict) -> None:
        self._raw_config = config  # store for later use
        # change input values to fm and MeV
        to_fm = ph.user_distance_unit/ph.fm
        to_MeV = ph.user_energy_unit/ph.MeV
        try:
            self._raw_config["bound_states"]["max_r"] = self._raw_config["bound_states"]["max_r"] * to_fm
        except KeyError:
            pass

        try:
            self._raw_config["scattering_states"]["max_r"] = self._raw_config["scattering_states"]["max_r"]*to_fm
        except KeyError:
            pass
        try:
            self._raw_config["spectra_computation"]["min_ke"] = self._raw_config["spectra_computation"]["min_ke"]*to_MeV
            self._raw_config["spectra_computation"]["e_max_log_2d"] = self._raw_config["spectra_computation"]["e_max_log_2d"]*to_MeV
        except KeyError:
            pass

        self.task_name = config["task"]

        proc_dict = {}
        proc_dict["type"] = ph.PROCESSES[config["process"]["name"]]
        if ("transition" in config["process"]):
            proc_dict["transition"] = ph.TRANSITION_NAMES[config["process"]["transition"]]
        if ("mechanism" in config["process"]):
            proc_dict["mechanism"] = ph.NEUTRINOLESS_MECHANISMS[config["process"]["mechanism"]]

        self.process = DoubleBetaProcess(**proc_dict)
        self.initial_atom_dict = config["initial_atom"]

    def resolve_nuclear_radius(self, initial_atom: AtomicSystem):
        if (self._raw_config["spectra_computation"]["nuclear_radius"] == "auto"):
            self._raw_config["spectra_computation"]["nuclear_radius"] = 1.2 * \
                (initial_atom.mass_number**(1./3.))
        elif (isinstance(self._raw_config["spectra_computation"]["nuclear_radius"], float)):
            if self._raw_config["spectra_computation"]["nuclear_radius"] < 0:
                raise ValueError("Negative nuclear radius not supported")
        else:
            raise ValueError("Cannot interpret nuclear_radius option")

    def resolve_q_value(self, initial_atom: AtomicSystem, final_atom: AtomicSystem):
        to_MeV = ph.user_energy_unit/ph.MeV
        if (type(self._raw_config["spectra_computation"]["q_value"]) is float):
            print("Received float q_value. All good, continue.")
            q_val = self._raw_config["spectra_computation"]["q_value"] * to_MeV
            return
        elif (type(self._raw_config["spectra_computation"]["q_value"]) is str):
            if (self._raw_config["spectra_computation"]["q_value"] != "auto"):
                raise ValueError(
                    "Cannot interpret q_value. Valid options: float or 'auto' ")
            # now we know q_value is "auto". Read files
            print("q_value is 'auto'. Will read from file")
            delta_m_map = ph.read_mass_difference(ph.q_values_file)
            delta_m_tmp = delta_m_map[initial_atom.name_nice]

            # if double beta minus
            if (self.process.type in [ph.TWONEUTRINO_TWOBMINUS, ph.NEUTRINOLESS_TWOBMINUS]):
                q_val = delta_m_tmp - 2.*ph.electron_mass
            elif (self.process.type in [ph.TWONEUTRINO_TWOBPLUS, ph.NEUTRINOLESS_TWOBPLUS]):
                q_val = delta_m_tmp - 4.*ph.electron_mass
            elif (self.process.type in [ph.TWONEUTRINO_BPLUSEC, ph.NEUTRINOLESS_BPLUSEC]):
                q_val = delta_m_tmp - 2.*ph.electron_mass
            elif (self.process.type in [ph.TWONEUTRINO_TWOEC]):
                q_val = delta_m_tmp
            else:
                raise ValueError("Could not resolve q_value option")

            self._raw_config["spectra_computation"]["q_value"] = q_val
        else:
            raise TypeError("Cannot interpret q_value")

    def resolve_bound_config(self, initial_atom: AtomicSystem):
        if not ("bound_states" in self._raw_config):
            return
        if self._raw_config["bound_states"]["n_values"] == "auto":
            self._raw_config["bound_states"]["n_values"] = list(
                set(initial_atom.n_values.tolist()))
        if self._raw_config["bound_states"]["k_values"] == "auto":
            k_values = {}
            for i_s in range(len(initial_atom.electron_config)):
                n = initial_atom.n_values[i_s]
                l = initial_atom.l_values[i_s]
                j = 0.5*initial_atom.jj_values[i_s]
                k = int((l-j)*(2*j+1))

                if not (initial_atom.n_values[i_s] in k_values):
                    k_values[n] = []
                k_values[n].append(k)
            self._raw_config["bound_states"]["k_values"] = k_values

    def create_spectra_config(self):
        for i in range(len(self._raw_config["spectra_computation"]["fermi_function_types"])):
            ft = self._raw_config["spectra_computation"]["fermi_function_types"][i]
            self._raw_config["spectra_computation"]["fermi_function_types"][i] = ph.FERMIFUNCTIONS[ft]
        for i in range(len(self._raw_config["spectra_computation"]["types"])):
            st = self._raw_config["spectra_computation"]["types"][i]
            self._raw_config["spectra_computation"]["types"][i] = ph.SPECTRUM_TYPES[st]
        self.spectra_config = SpectraConfig(
            **self._raw_config["spectra_computation"])

    def create_bound_config(self):
        if ("bound_states" in self._raw_config):
            self.bound_config = BoundConfig(
                **self._raw_config["bound_states"])
        else:
            self.bound_config = None

    def create_scattering_config(self):
        if ("scattering_states" in self._raw_config):
            self.scattering_config = ScatteringConfig(
                **self._raw_config["scattering_states"],
                min_ke=self.spectra_config.min_ke,
                max_ke=self.spectra_config.q_value)
        else:
            self.scattering_config = None

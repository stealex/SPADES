from dataclasses import dataclass, field
from logging import config
from multiprocessing import Value, process
from optparse import Option
from tokenize import Double
from typing import Optional, Dict, Any

from spades import ph
from spades.dhfs import AtomicSystem, create_ion
import logging
logger = logging.Logger(__name__)


@dataclass
class DoubleBetaProcess:
    type: ph.ProcessTypes
    transition: ph.TransitionTypes = ph.TransitionTypes.ZEROPLUS_TO_ZEROPLUS
    mechanism: ph.NeutrinoLessModes = ph.NeutrinoLessModes.LIGHT_NEUTRINO_EXCHANGE


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
    fermi_function_types: list[ph.FermiFunctionTypes]
    energy_grid_type: int
    n_ke_points: int
    min_ke: float
    total_ke: float
    ei_ef: float = -1.
    corrections: list[ph.CorrectionTypes] = field(default_factory=list)

    compute_2d: bool = False
    n_points_log_2d: int = 0
    n_points_lin_2d: int = 0
    e_max_log_2d: float = -1.


@dataclass
class OutputConfig:
    location: str
    what: list[str]
    file_prefix: str

    def solve_output_config(self):
        self.bound_wavefunctions = False
        self.scattering_wavefunctions = False
        self.spectra = False
        self.psfs = False
        self.binding_energies = False
        for elem in self.what:
            if elem == "bound wavefunctions":
                self.bound_wavefunctions = True
            elif elem == "scattering wavefunctions":
                self.scattering_wavefunctions = True
            elif elem == "spectra":
                self.spectra = True
            elif elem == "psfs":
                self.psfs = True
            elif elem == "binding energies":
                self.binding_energies = True
            else:
                raise NotImplementedError(f"Unknown output option {elem}")


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
        proc_dict["type"] = ph.PROCESS_NAMES_MAP[config["process"]["name"]]
        if ("transition" in config["process"]):
            proc_dict["transition"] = ph.TRANSITION_NAMES_MAP[config["process"]["transition"]]
        if ("mechanism" in config["process"]):
            proc_dict["mechanism"] = ph.NEUTRINOLESS_MECHANISMS_MAP[config["process"]["mechanism"]]

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

    def resolve_ei_ef(self, initial_atom: AtomicSystem, final_atom: AtomicSystem):
        to_MeV = ph.user_energy_unit/ph.MeV
        if (type(self._raw_config["spectra_computation"]["total_ke"]) is float):
            print("Received float total_ke. All good, continue.")
            total_ke = self._raw_config["spectra_computation"]["total_ke"] * to_MeV
        elif (type(self._raw_config["spectra_computation"]["total_ke"]) is str):
            if (self._raw_config["spectra_computation"]["total_ke"] != "auto"):
                raise ValueError(
                    "Cannot interpret total_ke. Valid options: float or 'auto' ")
            # now we know ei_ef is "auto". Read files
            print("total_ke is 'auto'. Will read from file")
            delta_m_map = ph.read_mass_difference(ph.delta_m_files)
            delta_m_tmp = delta_m_map[initial_atom.name_nice]

            if (self.process.type in [ph.ProcessTypes.TWONEUTRINO_TWOBMINUS, ph.ProcessTypes.NEUTRINOLESS_TWOBMINUS]):
                total_ke = delta_m_tmp
            elif (self.process.type in [ph.ProcessTypes.TWONEUTRINO_TWOBPLUS, ph.ProcessTypes.NEUTRINOLESS_TWOBPLUS]):
                total_ke = delta_m_tmp-4.0*ph.electron_mass
            elif (self.process.type in [ph.ProcessTypes.TWONEUTRINO_TWOEC]):
                total_ke = delta_m_tmp
            elif (self.process.type in [ph.ProcessTypes.TWONEUTRINO_BPLUSEC, ph.ProcessTypes.NEUTRINOLESS_BPLUSEC]):
                total_ke = delta_m_tmp-2.0*ph.electron_mass

            self._raw_config["spectra_computation"]["total_ke"] = total_ke
        else:
            raise TypeError("Cannot interpret total_ke")

        # now compute energy difference between nuclear states
        if (self.process.type in [ph.ProcessTypes.TWONEUTRINO_TWOBMINUS, ph.ProcessTypes.NEUTRINOLESS_TWOBMINUS,
                                  ph.ProcessTypes.TWONEUTRINO_TWOBPLUS, ph.ProcessTypes.NEUTRINOLESS_TWOBPLUS]):
            ei_ef = total_ke+2.0*ph.electron_mass
        elif (self.process.type in [ph.ProcessTypes.TWONEUTRINO_TWOEC]):
            ei_ef = total_ke-2.0*ph.electron_mass
        elif (self.process.type in [ph.ProcessTypes.TWONEUTRINO_BPLUSEC, ph.ProcessTypes.NEUTRINOLESS_BPLUSEC]):
            ei_ef = total_ke
        else:
            raise TypeError("Logic error, cannot determine ei_ef")
        self._raw_config["spectra_computation"]["ei_ef"] = ei_ef

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
            self._raw_config["spectra_computation"]["fermi_function_types"][i] = ph.FERMIFUNCTIONS_MAP[ft]
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
                max_ke=self.spectra_config.total_ke)
        else:
            self.scattering_config = None

    def create_output_config(self):
        if ("output" in self._raw_config):
            self.output_config = OutputConfig(**self._raw_config["output"])
            self.output_config.solve_output_config()
        else:
            self.output_config = None
